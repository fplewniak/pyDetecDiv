import torch
from torch import nn


class NN_module(nn.Module):
    def __init__(self, n_classes):
        super(NN_module, self).__init__()

        self.expected_shape = ('Batch', 3, 64, 64)

        self.nn = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=1, padding=1),  # [B, 3, 64, 64] → [B, 16, 64, 64]
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # [B, 16, 64, 64] → [B, 16, 32, 32]
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [B, 16, 32, 32] → [B, 32, 16, 16]
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # [B, 32, 16, 16] → [B, 32, 8, 8]
                nn.Flatten(), # [B, 32, 8, 8] → [B, 2048]
                nn.Linear(2048, 512),
                nn.Linear(512, n_classes),
                nn.Softmax(dim=-1)
                )

    def forward(self, x):
        output = self.nn(x)
        return output
