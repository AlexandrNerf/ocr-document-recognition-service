import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureWidthInterpolator(nn.Module):
    """Растягивает feature map по ширине до заданного числа таймстепов T"""

    def __init__(self, desired_T: int = 40):
        super().__init__()
        self.desired_T = desired_T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W
        _, _, h, w = x.shape
        if w < self.desired_T:
            x = F.interpolate(
                x, size=(h, self.desired_T), mode="bilinear", align_corners=False
            )
        return x
