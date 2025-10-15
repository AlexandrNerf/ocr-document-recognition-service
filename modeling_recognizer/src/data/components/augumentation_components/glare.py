import random

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class Glare(ImageOnlyTransform):
    def __init__(self, p=0.5, max_flares=5):
        super().__init__(p)
        '''
        Добавления полупрозрачных белых пятен на изображение с заданной вероятностью.
        :param p (default=0.2) - вероятность применения аугментации.
        :param max_flares (default=5) - максимальное количество точек, которые могут быть добавлены на изображение.
        '''
        self.p = p
        self.max_flares = max_flares

    def apply(self, img, **params):
        if random.random() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]
        overlay = img_np.copy()
        num_flares = random.randint(1, self.max_flares)

        for _ in range(num_flares):
            center = (random.randint(5, w - 5), random.randint(5, h - 5))
            radius = random.randint(min(h, w) // 10, min(h, w) // 2)
            color = (240, 240, 240)
            alpha = random.uniform(0.4, 0.8)
            flare = overlay.copy()
            cv2.circle(flare, center, radius, color, -1)
            overlay = cv2.addWeighted(flare, alpha, overlay, 1 - alpha, 0)

        return overlay
