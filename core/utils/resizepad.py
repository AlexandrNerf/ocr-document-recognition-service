import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


class ResizePad(ImageOnlyTransform):
    def __init__(self, target_height, max_width, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.target_height = target_height
        self.max_width = max_width

    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_width = min(int(w * (self.target_height / h)), self.max_width)
        img = cv2.resize(img, (new_width, self.target_height))

        pad_left = (self.max_width - new_width) // 2
        pad_right = self.max_width - new_width - pad_left

        img = cv2.copyMakeBorder(
            img, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0
        )
        return img
