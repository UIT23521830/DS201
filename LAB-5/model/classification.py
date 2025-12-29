import torch.nn as nn
from model.transformer import TransformerEncoder

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids):
        enc_out = self.encoder(input_ids)
        pooled = enc_out.mean(dim=1)
        return self.classifier(pooled)
