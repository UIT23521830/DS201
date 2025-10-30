import torch
import torch.nn as nn
import torch.nn.functional as F

class Perceptron_1_Layer(nn.Module):
    def __init__(self, image_size: tuple, num_labels: int):
        super().__init__()

        w, h = image_size
        self.linear = nn.Linear(
            in_features=w * h,
            out_features=num_labels
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, -1) # bs, w*h
        x = self.linear(x)   # bs, num_labels
        x = F.softmax(x, dim=1)

        return x