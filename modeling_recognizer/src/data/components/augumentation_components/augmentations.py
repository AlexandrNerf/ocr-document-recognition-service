import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.data.components.augumentation_components.glare import Glare
from src.data.components.augumentation_components.scan_line import ScanLineAugmentation
from src.data.components.augumentation_components.zoom_blur import ZoomBlur
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomGrayscale,
    RandomPerspective,
    RandomPhotometricDistort,
)
from doctr import transforms as T

def augs_val(h, w, use_augmentation):
    return Compose([
        T.Resize((h, w), preserve_aspect_ratio=True),
        ])

def augs_train(h, w, use_augmentation):
     return Compose([
        T.Resize((h, w), preserve_aspect_ratio=True),
        # Augmentations
        T.RandomApply(T.ColorInversion(), 0.1),
        RandomGrayscale(p=0.1),
        RandomPhotometricDistort(p=0.1),
        T.RandomApply(T.RandomShadow(), p=0.4),
        T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
        RandomPerspective(distortion_scale=0.2, p=0.3),
    ])

def soft_augs(h, w, use_augmentation):
    return (
        A.Compose(
            [
                A.Compose(
                    [
                        A.SmallestMaxSize(max_size=h),
                        A.PadIfNeeded(
                            min_height=h, min_width=w, 
                            border_mode=0,
                            value=(0, 0, 0)
                        ),
                        A.Resize(height=h, width=w)
                    ]
                ),
                A.OneOf(
                    [
                        A.Morphological(operation="erosion", scale=(2, 3), p=0.2),
                        A.Morphological(operation="dilation", scale=(2, 3), p=0.2),
                    ],
                    p=1,
                ),
                Glare(p=0.2, max_flares=8),
                ScanLineAugmentation(
                    max_line_count=5,
                    thickness_range=(1, 2),
                    intensity_range=(50, 150),
                    p=0.6,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(
                            blur_limit=(3, 3), sigma_limit=(0.1, 5.0), p=0.3
                        ),
                        ZoomBlur(p=0.2, lvl=0),
                    ],
                    p=1,
                ),
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.05,
                            p=0.3,
                        ),
                        A.Solarize(threshold_range=(0.3, 0.7), p=0.1),
                    ],
                    p=1,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(std_range=(0.05, 0.1), p=0.6),
                        A.ISONoise(
                            color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.6
                        ),
                        A.MultiplicativeNoise(
                            multiplier=(0.7, 1.3), per_channel=True, p=0.3
                        ),
                    ],
                    p=1,
                ),
                ToTensorV2(),
            ]
            if use_augmentation
            else [
                A.Compose(
                    [
                        A.SmallestMaxSize(max_size=40),
                        A.Resize(height=40, width=200),
                    ]
                ),
                ToTensorV2(),
            ]
        ),

        A.Compose(
            [
                A.Compose(
                    [
                        A.SmallestMaxSize(max_size=40),
                        A.Resize(height=40, width=200),
                    ]
                ),
                ToTensorV2(),
            ]
        )
        
        )