import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, n_labels, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.bilstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size*2, n_labels)  # *2 vì BiLSTM

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)
        out, _ = self.bilstm(emb)  # [batch_size, seq_len, hidden*2]
        logits = self.classifier(out)  # [batch_size, seq_len, n_labels]
        return logits
