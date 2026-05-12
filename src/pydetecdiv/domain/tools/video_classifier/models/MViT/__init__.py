import torch.nn as nn
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights, mvit_v2_s, MViT_V2_S_Weights

"""
Data should be provided as follows:

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

transform = Compose([
    Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
    Normalize(mean, std),
    CenterCrop(224),
])
dataset = ROIDataset(hdf5_file, [[1, 1]], targets=True, image_shape=(256, 256), seq2one=False, seqlen=15, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, (batch, targets) in enumerate(dataloader):
    print(f'batch {i}')
    pred = model(batch)
"""


class MViT_v1_b(nn.Module):
    def __init__(self, n_classes=400, dropout=0.5, **kwargs):
        super(MViT_v1_b, self).__init__()

        self.expected_shape = ('Batch', 16, 3, 224, 224)

        self.mvit = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

        self.mvit.head = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                  nn.Linear(in_features=768, out_features=n_classes, bias=True)
                                  )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.mvit(x)
        return x


class MViT_v2_s(nn.Module):
    def __init__(self, n_classes=400, dropout=0.5, **kwargs):
        super(MViT_v2_s, self).__init__()

        self.expected_shape = ('Batch', 16, 3, 224, 224)

        self.mvit = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)

        self.mvit.head = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                  nn.Linear(in_features=768, out_features=n_classes, bias=True)
                                  )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.mvit(x)
        return x
