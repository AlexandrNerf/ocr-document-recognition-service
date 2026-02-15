import torch
from torch import Tensor, nn
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.v2 import RandomPhotometricDistort

RandomPhotometricDistort()

def apply_jpeg(x: Tensor, quality: int) -> Tensor:
    return decode_jpeg(encode_jpeg(x, quality))


class RandomApplyJpeg(nn.Module):
    def __init__(self, min_quality: int, max_quality: int) -> None:
        assert 1 <= min_quality <= max_quality <= 100
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality

    def forward(self, x: Tensor) -> Tensor:
        quality = torch.randint(self.min_quality, self.max_quality + 1, ()).item()
        return apply_jpeg(x, quality)