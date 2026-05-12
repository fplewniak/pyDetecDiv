from torch import nn
from torchvision.models.video import s3d, S3D_Weights


class S3D(nn.Module):
    def __init__(self, n_classes=6, dropout=0.5, **kwargs):
        super(S3D, self).__init__()

        self.expected_shape = ('Batch', 14, 3, 224, 224)

        self.s3d = s3d(weights=S3D_Weights.DEFAULT)

        self.s3d.head = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                  nn.Linear(in_features=1024, out_features=n_classes, bias=True)
                                  )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.s3d(x)
        return x
