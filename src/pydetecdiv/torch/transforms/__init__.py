import torch
from torchvision.transforms import v2


class toStandardizedFloat32(torch.nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return v2.ToDtype(torch.float32, scale=True)(img) / torch.max(img).item()
