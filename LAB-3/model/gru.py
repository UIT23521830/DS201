import torch
from torch import nn

class GRUModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, n_labels, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=300,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, n_labels)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.gru(x)
        
        # Lấy hidden state của layer cuối + cả 2 directions
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)

        return self.fc(final_hidden)
