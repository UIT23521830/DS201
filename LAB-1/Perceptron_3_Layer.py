import torch
import torch.nn as nn
import torch.nn.functional as F


class Perceptron_3_Layer(nn.Module):

    def __init__(self, image_size: tuple = (28, 28), hidden1: int = 128, hidden2: int = 64, num_labels: int = 10):
        super().__init__()
        w, h = image_size
        in_features = w * h
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
