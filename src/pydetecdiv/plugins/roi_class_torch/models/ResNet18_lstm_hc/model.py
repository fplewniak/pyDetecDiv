import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


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

        # Load ResNet18
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D

        self.folding = SequenceFoldingLayer((3, 60, 60))
        self.unfolding = SequenceUnfoldingLayer((512, 1, 1))  # ResNet18 outputs 512 features

        num_layers = 1
        hidden_size = 150
        self.bilstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.h0 = nn.Parameter(torch.randn(2 * num_layers, 1, hidden_size) * 0.01)
        self.c0 = nn.Parameter(torch.randn(2 *num_layers, 1, hidden_size) * 0.01)

        self.dropout = nn.Dropout(0.5)

        self.classify = nn.Linear(2 * hidden_size, n_classes)
        nn.init.xavier_uniform_(self.classify.weight)
        if self.classify.bias is not None:
            nn.init.zeros_(self.classify.bias)


    def forward(self, x):
        x, batch_size = self.folding(x)  # Fold sequence into batch form
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.unfolding(x, batch_size)  # Unfold sequence
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten
        # Expand learnable initial states to match the batch size
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        x, _ = self.bilstm(x, (h0, c0))  # Pass through BiLSTM
        x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        x = self.classify(x)
        return x
