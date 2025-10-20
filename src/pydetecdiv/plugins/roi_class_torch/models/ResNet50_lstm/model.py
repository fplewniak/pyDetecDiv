import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


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

        # Load ResNet50 (since PyTorch lacks ResNet50V2)
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.folding = SequenceFoldingLayer((3, 60, 60))
        self.unfolding = SequenceUnfoldingLayer((2048, 1, 1))  # ResNet50 outputs 2048 features
        # self.unfolding = SequenceUnfoldingLayer((1, 1, 2048))  # or should it be that order ? to be tested, which one works best ?

        self.bilstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, dropout=0.25,
                              batch_first=True, bidirectional=True)  # check that all parameters are OK, where are activation, etc.

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)  # BiLSTM output size = 2 * hidden_size
        self.fc2 = nn.Linear(128, n_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, batch_size = self.folding(x)  # Fold sequence into batch form
        x = self.resnet(x)  # CNN Feature Extraction
        # x = self.global_avg_pool(x)
        x = self.global_max_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)
        x = self.unfolding(x, batch_size)  # Unfold sequence
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten
        x = self.relu(x)
        # x = self.dropout(x)
        x, _ = self.bilstm(x)  # Pass through BiLSTM
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.softmax(x)

        return x
