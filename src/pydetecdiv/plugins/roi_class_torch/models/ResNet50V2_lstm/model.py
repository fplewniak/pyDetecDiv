import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SequenceFoldingLayer(nn.Module):
    """
    Folds a sequence of images into a batch to feed the ResNet.
    """

    def __init__(self, data_shape):
        super(SequenceFoldingLayer, self).__init__()
        self.data_shape = data_shape  # (H, W, C)

    def forward(self, x):
        batch_size, seq_len, h, w, c = x.shape  # Original shape
        x = x.view(-1, h, w, c).permute(0, 3, 1, 2)  # Reshape and reorder to (batch, C, H, W)
        return x, batch_size  # Return reshaped input and batch size


class SequenceUnfoldingLayer(nn.Module):
    """
    Unfolds the sequence after CNN processing to reshape for LSTM input.
    """

    def __init__(self, data_shape):
        super(SequenceUnfoldingLayer, self).__init__()
        self.data_shape = data_shape  # (1, 1, feature_dim)

    def forward(self, x, batch_size):
        seq_len = x.shape[0] // batch_size  # Recover original sequence length
        x = x.view(batch_size, seq_len, *self.data_shape)  # Restore sequence dimension
        return x


class ResNetLSTMModel(nn.Module):
    def __init__(self, n_classes):
        super(ResNetLSTMModel, self).__init__()

        # Load ResNet50 (since PyTorch lacks ResNet50V2)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Equivalent to GlobalAveragePooling2D

        self.folding = SequenceFoldingLayer((60, 60, 3))
        self.unfolding = SequenceUnfoldingLayer((1, 1, 2048))  # ResNet50 outputs 2048 features

        self.bilstm = nn.LSTM(input_size=2048, hidden_size=150, num_layers=1,
                              batch_first=True, bidirectional=True) # check that all parameters are OK, where are activation, etc.

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, n_classes)  # BiLSTM output size = 2 * hidden_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, batch_size = self.folding(x)  # Fold sequence into batch form
        x = self.resnet(x)  # CNN Feature Extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, features)

        x = self.unfolding(x, batch_size)  # Unfold sequence
        x = x.permute(0, 1, 3, 2)  # Equivalent to TimeDistributed(Permute(3,2,1))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten

        x, _ = self.bilstm(x)  # Pass through BiLSTM
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
