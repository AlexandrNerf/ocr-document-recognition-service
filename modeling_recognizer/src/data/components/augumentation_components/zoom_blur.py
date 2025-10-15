import random
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class ZoomBlur(ImageOnlyTransform):
    def __init__(self, p=0.2, lvl=0, p_shift=1.0):
        super().__init__(p)
        '''
        Создает эффект размытия с масштабированием
        :param p (default=0.2) - вероятность применения аугментации
        :param lvl (default=0) - уровень эффекта блюра
        :param p_shift
        '''
        self.lvl = lvl
        self.p_shift = p_shift
        self.shift = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

    def apply(self, img, **params):
        if random.random() > self.p:
            return img

        shift_x = random.choice(self.shift) if random.random() < self.p_shift else 1.0
        shift_y = random.choice(self.shift) if random.random() < self.p_shift else 1.0

        h, w = img.shape[:2]

        c = [
            np.arange(1, 1.06, 0.01),
            np.arange(1, 1.12, 0.01),
            np.arange(1, 1.18, 0.01),
        ]
        if self.lvl < 0 or self.lvl >= len(c):
            raise ValueError(f'lvl zoom is invalid: {self.lvl}. ')
        scales = c[self.lvl]

        uint8_img = img.copy()
        img_float = img.astype(np.float32) / 255.0
        out = np.zeros_like(img_float)

        for zoom_factor in scales:
            zw = int(w * zoom_factor)
            zh = int(h * zoom_factor)

            resized = cv2.resize(uint8_img, (zw, zh), interpolation=cv2.INTER_CUBIC)

            x1 = int(((zw - w) // 2) * shift_x)
            y1 = int(((zh - h) // 2) * shift_y)
            x2 = min(x1 + w, zw)
            y2 = min(y1 + h, zh)

            crop = resized[y1:y2, x1:x2]

            padded = np.zeros_like(uint8_img, dtype=np.float32)
            ch, cw = crop.shape[:2]
            padded[:ch, :cw] = crop.astype(np.float32) / 255.0

            out += padded

        img_result = (img_float + out) / (len(scales) + 1)
        img_result = np.clip(img_result, 0, 1) * 255
        return img_result.astype(np.uint8)
