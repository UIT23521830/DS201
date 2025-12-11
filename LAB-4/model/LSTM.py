import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2seqLSTM(nn.Module):
    def __init__(self, d_model, vocab, n_encoder=3, n_decoder=3, dropout=0.1):
        super().__init__()
        self.vocab = vocab

        # Embedding
        self.src_embedding = nn.Embedding(vocab.total_src_tokens, d_model, padding_idx=vocab.pad_idx)
        self.tgt_embedding = nn.Embedding(vocab.total_tgt_tokens, d_model, padding_idx=vocab.pad_idx)

        # Encoder LSTM 3-layer
        self.encoder = nn.LSTM(
            d_model, d_model, 
            num_layers=n_encoder,
            dropout=dropout,
            batch_first=True
        )

        # Decoder LSTM 3-layer
        self.decoder = nn.LSTM(
            d_model, d_model,
            num_layers=n_decoder,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_head = nn.Linear(d_model, vocab.total_tgt_tokens)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, tgt):
        # src: (bs, src_len), tgt: (bs, tgt_len)
        src_embed = self.src_embedding(src)
        encoder_outputs, (h, c) = self.encoder(src_embed)

        # init decoder from encoder last hidden state
        dec_h = h
        dec_c = c

        tgt_embed = self.tgt_embedding(tgt)
        logits = []

        # Teacher forcing loop
        for t in range(tgt.size(1)):
            y_t = tgt_embed[:, t, :].unsqueeze(1)  # (bs, 1, d)
            out, (dec_h, dec_c) = self.decoder(y_t, (dec_h, dec_c))
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

            dec_h, dec_c = h, c

            y_t = torch.full((batch,), self.vocab.bos_idx, dtype=torch.long, device=src.device)

            outputs = [[] for _ in range(batch)]
            finished = torch.zeros(batch, dtype=torch.bool, device=src.device)

            for _ in range(max_len):
                embed = self.tgt_embedding(y_t).unsqueeze(1)
                out, (dec_h, dec_c) = self.decoder(embed, (dec_h, dec_c))
                logit = self.output_head(out.squeeze(1))

                next_y = logit.argmax(dim=-1)

                for i in range(batch):
                    if not finished[i]:
                        if int(next_y[i]) == self.vocab.eos_idx:
                            finished[i] = True
                        else:
                            outputs[i].append(int(next_y[i]))

                y_t = next_y
                if finished.all():
                    break

            # pad output
            max_len_out = max(len(o) for o in outputs)
            out_tensor = torch.full((batch, max_len_out), self.vocab.pad_idx, device=src.device)

            for i in range(batch):
                out_tensor[i, :len(outputs[i])] = torch.tensor(outputs[i], device=src.device)

            return out_tensor
