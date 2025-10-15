import random

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

class ScanLineAugmentation(ImageOnlyTransform):
    def __init__(self, max_line_count=3, thickness_range=(1, 3), intensity_range=(50, 200), p=0.5):
        super().__init__(p)
        '''
        Созздает линии на картинке.

        :param p (default=0.2) - вероятность применения аугментации.
        :param line_count (default=3) - количество линий на изображении.
        :param thickness_range (default=(1, 3)) - диапозон толщины линий.
        :param intensity_range (default=(50, 200)) - интенсивность серого цвета.
        '''
        self.max_line_count = max_line_count
        self.thickness_range = thickness_range
        self.intensity_range = intensity_range
        self.p = p

    def apply(self, img, **params):
        if random.random() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        for _ in range(random.randint(1, self.max_line_count)):
            # Параметры линии
            thickness = random.randint(*self.thickness_range)
            intensity = random.randint(*self.intensity_range)
            y_start = random.randint(0, h)
            y_end = h - y_start
            # Генерируем базовые координаты
            x_start = random.randint(-20, 20)  # Начало за пределами изображения
            x_end = random.randint(w - 20, w + 20)  # Конец за пределами изображения

            cv2.line(img_np,
                     (x_start, y_start),
                     (x_end, y_end),
                     (intensity, intensity, intensity),  # Серый цвет
                     thickness,
                     lineType=cv2.LINE_AA)

        return img_np
