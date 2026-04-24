"""
Custom v2 Torchvision transforms
"""
import torch
from torchvision.transforms import v2


class toStandardizedFloat32(torch.nn.Module):
    """
    A transformation to standardized float32 tensor, with values between 0 and 1.
    """
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the transform

        :param img: the image or video to transform
        :return: the transformed image or video
        """
        return v2.ToDtype(torch.float32, scale=True)(img) / torch.max(img).item()
