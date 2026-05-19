from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18, r2plus1d_18, R2Plus1D_18_Weights


class R3D_18(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(R3D_18, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.r3d_18 = r3d_18(weights=R3D_18_Weights.DEFAULT)

        self.r3d_18.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                       nn.Linear(in_features=512, out_features=n_classes, bias=True)
                                       )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.r3d_18(x)
        return x


class MC3_18(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(MC3_18, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.mc3_18 = mc3_18(weights=MC3_18_Weights.DEFAULT)

        self.mc3_18.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                       nn.Linear(in_features=512, out_features=n_classes, bias=True)
                                       )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.mc3_18(x)
        return x


class R2Plus1d_18(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(R2Plus1d_18, self).__init__()

        self.expected_shape = ('Batch', 4, 3, 224, 224)

        self.r2plus1d_18 = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        self.linear = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        self.r2plus1d_18.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                            self.linear
                                            )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.r2plus1d_18(x)
        return x
