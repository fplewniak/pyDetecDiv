import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torchvision import models
from torchvision.models import ResNet18_Weights


class NN_module(nn.Module):
    def __init__(self, n_classes):
        super(NN_module, self).__init__()

        self.expected_shape = ('Batch', 3, 60, 60)

        # Load ResNet50 (since PyTorch lacks ResNet50V2)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(8192, n_classes)  # BiLSTM output size = 2 * hidden_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
