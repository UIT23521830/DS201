import torch
from torch import nn

from torch.nn import NLLLoss, CrossEntropyLoss

class LstmMoodel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_labels: int,
        padding_idx: int = 0
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layers,
        )

        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=n_labels
        )

    def forward(self, inputs: torch.Tensor):
        embedded_features = self.embedding(inputs)
        features, _ = self.lstm(embedded_features)

        # lấy hidden state cuối cùng
        features = features[:, -1]
        logits = self.classifier(features)

        return logits

