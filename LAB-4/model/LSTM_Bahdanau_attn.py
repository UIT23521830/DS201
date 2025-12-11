# model/LSTM_Bahdanau_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, dec_dim)
        self.W2 = nn.Linear(dec_dim, dec_dim)
        self.V = nn.Linear(dec_dim, 1)

    def forward(self, encoder_outputs, hidden):
        hidden = hidden.unsqueeze(1)  # (batch,1,dec_dim)
        score = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        score = self.V(score).squeeze(-1)  # (batch, src_len)
        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Seq2seqLSTM(nn.Module):
    def __init__(self, d_model, vocab, n_encoder=3, n_decoder=3, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.enc_hidden = d_model
        self.dec_hidden = d_model

        self.src_embedding = nn.Embedding(vocab.total_src_tokens, d_model, padding_idx=vocab.pad_idx)
        self.encoder = nn.LSTM(d_model, d_model, num_layers=n_encoder, dropout=dropout, batch_first=True, bidirectional=True)

        self.tgt_embedding = nn.Embedding(vocab.total_tgt_tokens, d_model, padding_idx=vocab.pad_idx)
        self.decoder = nn.LSTM(d_model + 2 * d_model, d_model, num_layers=n_decoder, dropout=dropout, batch_first=True)

        self.attention = BahdanauAttention(enc_dim=2 * d_model, dec_dim=d_model)
        self.output_head = nn.Linear(d_model, vocab.total_tgt_tokens)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, tgt):
        # src: (bs, src_len), tgt: (bs, tgt_len)
        src_embed = self.src_embedding(src)
        encoder_outputs, (h, c) = self.encoder(src_embed)  # encoder_outputs: (bs, src_len, 2*d)
        # init decoder hidden from last forward layer
        dec_h = h[-1].unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        dec_c = c[-1].unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        tgt_embed = self.tgt_embedding(tgt)
        logits = []
        for t in range(tgt.size(1)):
            y_t = tgt_embed[:, t, :]  # (bs, d_model)
            context, _ = self.attention(encoder_outputs, dec_h[-1])
            dec_input = torch.cat([y_t, context], dim=-1).unsqueeze(1)
            out, (dec_h, dec_c) = self.decoder(dec_input, (dec_h, dec_c))
            logit = self.output_head(out.squeeze(1))
            logits.append(logit.unsqueeze(1))
        logits = torch.cat(logits, dim=1)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return loss

    def predict(self, src, max_len=50):
        self.eval()
        with torch.no_grad():
            batch = src.size(0)
            src_embed = self.src_embedding(src)
            encoder_outputs, (h, c) = self.encoder(src_embed)

            dec_h = h[-1].unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
            dec_c = c[-1].unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

            y_t = torch.full((batch,), self.vocab.bos_idx, dtype=torch.long, device=src.device)

            outputs = [[] for _ in range(batch)]
            finished = torch.zeros(batch, dtype=torch.bool, device=src.device)

            for _ in range(max_len):
                embed = self.tgt_embedding(y_t)  # (batch, d)
                context, _ = self.attention(encoder_outputs, dec_h[-1])  # (batch, 2d)
                dec_input = torch.cat([embed, context], dim=-1).unsqueeze(1)  # (batch,1,d+2d)
                out, (dec_h, dec_c) = self.decoder(dec_input, (dec_h, dec_c))  # out: (batch,1,d)
                logit = self.output_head(out.squeeze(1))  # (batch, vocab)
                next_y = logit.argmax(dim=-1)  # (batch,)

                for i in range(batch):
                    if not finished[i]:
                        if int(next_y[i].item()) == self.vocab.eos_idx:
                            finished[i] = True
                        else:
                            outputs[i].append(int(next_y[i].item()))

                y_t = next_y
                if finished.all():
                    break

            # pad outputs to tensor
            max_len_out = max((len(o) for o in outputs), default=0)
            if max_len_out == 0:
                return torch.zeros((batch, 0), dtype=torch.long, device=src.device)
            out_tensor = torch.full((batch, max_len_out), self.vocab.pad_idx, dtype=torch.long, device=src.device)
            for i in range(batch):
                if len(outputs[i]) > 0:
                    out_tensor[i, :len(outputs[i])] = torch.tensor(outputs[i], dtype=torch.long, device=src.device)
            return out_tensor
