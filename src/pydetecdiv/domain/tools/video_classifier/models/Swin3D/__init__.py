from torch import nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights, Swin3D_S_Weights, swin3d_s, swin3d_b, Swin3D_B_Weights


class Swin3D_tiny(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(Swin3D_tiny, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.swin = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)

        self.swin.head = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                  nn.Linear(in_features=768, out_features=n_classes, bias=True)
                                  )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.swin(x)
        return x


class Swin3D_small(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(Swin3D_small, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.swin = swin3d_s(weights=Swin3D_S_Weights.DEFAULT)

        self.swin.head = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                  nn.Linear(in_features=768, out_features=n_classes, bias=True)
                                  )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.swin(x)
        return x


class Swin3D_base(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(Swin3D_base, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.swin = swin3d_b(weights=Swin3D_B_Weights.DEFAULT)

        self.swin.head = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                  nn.Linear(in_features=1024, out_features=n_classes, bias=True)
                                  )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.swin(x)
        return x
