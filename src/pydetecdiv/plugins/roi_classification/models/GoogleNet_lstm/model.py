import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import GoogLeNet_Weights


class SequenceFoldingLayer(nn.Module):
    """
    Folds a sequence of images into a batch to feed the ResNet.
    Maybe the nn.Fold layer would work here
    """

    def __init__(self, data_shape):
        super(SequenceFoldingLayer, self).__init__()
        self.data_shape = data_shape

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape  # Original shape
        x = x.view(-1, c, h, w)
        return x, torch.tensor(batch_size)


class SequenceUnfoldingLayer(nn.Module):
    """
    Unfolds the sequence after CNN processing to reshape for LSTM input.
    Maybe the nn.Unfold layer would work here
    """

    def __init__(self, data_shape):
        super(SequenceUnfoldingLayer, self).__init__()
        self.data_shape = data_shape  # (1, 1, feature_dim)

    def forward(self, x, batch_size):
        seq_len = x.shape[0] // batch_size  # Recover original sequence length
        x = x.view(batch_size, seq_len, *self.data_shape)  # Restore sequence dimension
        return x


class NN_module(nn.Module):
    def __init__(self, n_classes):
        super(NN_module, self).__init__()

        self.expected_shape = ('Batch', 'Sequence', 3, 60, 60)

        googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        # googlenet = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
        self.googlenet = nn.Sequential(*list(googlenet.children())[:-2])  # Remove FC layer

        self.folding = SequenceFoldingLayer((3, 60, 60))
        self.unfolding = SequenceUnfoldingLayer((1024, 1, 1))  # GoogleNet outputs 1024 features

        self.bilstm = nn.LSTM(input_size=1024, hidden_size=150, num_layers=1, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.classify = nn.Linear(300, n_classes)

    def forward(self, x):
        x, batch_size = self.folding(x)  # Fold sequence into batch form
        x = self.googlenet(x)  # CNN Feature Extraction
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.unfolding(x, batch_size)  # Unfold sequence
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten
        x, _ = self.bilstm(x)  # Pass through BiLSTM
        x = nn.functional.leaky_relu(x)
        # x = self.dropout(x)
        x = self.classify(x)
        return x
