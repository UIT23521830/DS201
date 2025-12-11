# model/LSTM_Luong_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    """
    Implements Luong global-style attention (general score).
    score(h_t, H) = h_t^T * W_a * H
    After score -> softmax over source length -> context = a * H
    Then concat: attn_vector = tanh(W_c [context; h_t])
    """
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        # W_a: project encoder states to dec_dim (for general score)
        self.Wa = nn.Linear(enc_dim, dec_dim, bias=False)
        # W_c: combine context and decoder hidden
        self.Wc = nn.Linear(enc_dim + dec_dim, dec_dim, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        """
        dec_hidden: (batch, dec_dim)  -- current decoder hidden (last layer)
        enc_outputs: (batch, src_len, enc_dim)
        mask: (batch, src_len) with 1 for valid tokens, 0 for padding (optional)
        returns:
            attn_vector: (batch, dec_dim)  -- fused vector tanh(Wc[context; h_t])
            attn_weights: (batch, src_len)
        """
        # enc_proj: (batch, src_len, dec_dim)
        enc_proj = self.Wa(enc_outputs)  # (b, src_len, dec_dim)
        # dec_hidden unsqueeze for matmul
        # score = enc_proj dot dec_hidden  -> (b, src_len)
        # compute scores
        # dec_hidden: (b, dec_dim) -> (b, dec_dim, 1)
        scores = torch.bmm(enc_proj, dec_hidden.unsqueeze(-1)).squeeze(-1)  # (b, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (b, src_len)

        # context: weighted sum of enc_outputs
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (b, enc_dim)

        # concat context and dec_hidden, then project
        concat = torch.cat([context, dec_hidden], dim=-1)  # (b, enc_dim + dec_dim)
        attn_vector = torch.tanh(self.Wc(concat))  # (b, dec_dim)

        return attn_vector, attn_weights


class Seq2seqLSTMLuong(nn.Module):
    def __init__(self, d_model, vocab, n_encoder=3, n_decoder=3, dropout=0.1):
        """
        d_model: hidden size (encoder hidden per direction = d_model)
                 Note: encoder is bidirectional -> enc_dim = 2*d_model
        vocab: Vocab object with attributes pad_idx, bos_idx, eos_idx, total_*_tokens
        """
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model

        # embeddings
        self.src_embedding = nn.Embedding(vocab.total_src_tokens, d_model, padding_idx=vocab.pad_idx)
        self.tgt_embedding = nn.Embedding(vocab.total_tgt_tokens, d_model, padding_idx=vocab.pad_idx)

        # encoder: bidirectional
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_encoder,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # decoder: unidirectional
        # decoder takes embedded token concatenated? For Luong we feed embedding only, then attention uses decoder hidden.
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_decoder,
            dropout=dropout,
            batch_first=True
        )

        # Luong attention: enc_dim = 2*d_model, dec_dim = d_model
        self.attention = LuongAttention(enc_dim=2 * d_model, dec_dim=d_model)

        # final projection from attn_vector to vocab logits
        self.output_head = nn.Linear(d_model, vocab.total_tgt_tokens)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def _init_decoder_states(self, h, c):
        """
        Convert encoder (num_layers*2, b, d) into decoder initial (num_layers, b, d)
        We will sum the forward and backward states per layer to initialize decoder.
        """
        # h/c shapes: (num_layers*2, batch, d)
        num_layers_times_2 = h.size(0)
        batch = h.size(1)
        d = h.size(2)
        num_enc_layers = num_layers_times_2 // 2
        # reshape -> (num_layers, 2, batch, d)
        h_reshaped = h.view(num_enc_layers, 2, batch, d)
        c_reshaped = c.view(num_enc_layers, 2, batch, d)
        # sum forward and backward
        h_sum = h_reshaped.sum(dim=1)  # (num_layers, batch, d)
        c_sum = c_reshaped.sum(dim=1)
        # Now decoder has possibly different num_layers; if decoder has more layers, we will repeat last
        return h_sum, c_sum

    def forward(self, src, tgt, src_mask=None):
        """
        src: (batch, src_len) token ids
        tgt: (batch, tgt_len) token ids  -> typically we pass tgt shifted for teacher forcing (with BOS and without last EOS)
        src_mask: (batch, src_len) optional mask (1 for token, 0 for pad)
        """
        batch = src.size(0)
        # encode
        src_embed = self.src_embedding(src)  # (b, src_len, d)
        enc_outputs, (h_enc, c_enc) = self.encoder(src_embed)  # enc_outputs: (b, src_len, 2*d)

        # init decoder hidden states: sum forward/back per layer
        h_dec_init, c_dec_init = self._init_decoder_states(h_enc, c_enc)  # (n_enc_layers, b, d)

        # if decoder has different num_layers, adapt (repeat last layer)
        dec_num_layers = self.decoder.num_layers
        if h_dec_init.size(0) >= dec_num_layers:
            dec_h = h_dec_init[:dec_num_layers]
            dec_c = c_dec_init[:dec_num_layers]
        else:
            # repeat last layer to fill
            repeat = dec_num_layers - h_dec_init.size(0)
            last_h = h_dec_init[-1:].repeat(repeat, 1, 1)
            last_c = c_dec_init[-1:].repeat(repeat, 1, 1)
            dec_h = torch.cat([h_dec_init, last_h], dim=0)
            dec_c = torch.cat([c_dec_init, last_c], dim=0)

        # decode with teacher forcing
        tgt_embed = self.tgt_embedding(tgt)  # (b, tgt_len, d)
        logits = []
        # We'll maintain dec_h, dec_c updated each step
        for t in range(tgt_embed.size(1)):
            y_t = tgt_embed[:, t, :].unsqueeze(1)  # (b,1,d)
            out, (dec_h, dec_c) = self.decoder(y_t, (dec_h, dec_c))  # out: (b,1,d)
            dec_out = out.squeeze(1)  # (b, d)
            # attention using decoder last-layer hidden (take dec_h[-1])
            dec_last = dec_h[-1]  # (b, d)
            attn_vector, _ = self.attention(dec_last, enc_outputs, mask=src_mask)  # (b, d)
            logit = self.output_head(attn_vector)  # (b, vocab)
            logits.append(logit.unsqueeze(1))

        logits = torch.cat(logits, dim=1)  # (b, tgt_len, vocab)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return loss

    def predict(self, src, max_len=50, src_mask=None):
        """
        Greedy decode, return padded tensor (batch, out_len)
        """
        self.eval()
        with torch.no_grad():
            batch = src.size(0)
            src_embed = self.src_embedding(src)
            enc_outputs, (h_enc, c_enc) = self.encoder(src_embed)
            h_dec_init, c_dec_init = self._init_decoder_states(h_enc, c_enc)

            dec_num_layers = self.decoder.num_layers
            if h_dec_init.size(0) >= dec_num_layers:
                dec_h = h_dec_init[:dec_num_layers].clone()
                dec_c = c_dec_init[:dec_num_layers].clone()
            else:
                repeat = dec_num_layers - h_dec_init.size(0)
                dec_h = torch.cat([h_dec_init, h_dec_init[-1:].repeat(repeat, 1, 1)], dim=0)
                dec_c = torch.cat([c_dec_init, c_dec_init[-1:].repeat(repeat, 1, 1)], dim=0)

            y_t = torch.full((batch,), self.vocab.bos_idx, dtype=torch.long, device=src.device)
            outputs = [[] for _ in range(batch)]
            finished = torch.zeros(batch, dtype=torch.bool, device=src.device)

            for _ in range(max_len):
                embed = self.tgt_embedding(y_t).unsqueeze(1)  # (b,1,d)
                out, (dec_h, dec_c) = self.decoder(embed, (dec_h, dec_c))
                dec_out = out.squeeze(1)
                dec_last = dec_h[-1]
                attn_vector, _ = self.attention(dec_last, enc_outputs, mask=src_mask)
                logit = self.output_head(attn_vector)
                next_y = logit.argmax(dim=-1)

                for i in range(batch):
                    if not finished[i]:
                        if int(next_y[i].item()) == self.vocab.eos_idx:
                            finished[i] = True
                        else:
                            outputs[i].append(int(next_y[i].item()))

                y_t = next_y
                if finished.all():
                    break

            max_out = max((len(o) for o in outputs), default=0)
            if max_out == 0:
                return torch.zeros((batch, 0), dtype=torch.long, device=src.device)
            out_tensor = torch.full((batch, max_out), self.vocab.pad_idx, dtype=torch.long, device=src.device)
            for i in range(batch):
                if outputs[i]:
                    out_tensor[i, :len(outputs[i])] = torch.tensor(outputs[i], dtype=torch.long, device=src.device)

            return out_tensor
