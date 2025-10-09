import os
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from doctr.models import detection_predictor, fast_base, reparameterize
from doctr.models.builder import DocumentBuilder

from data.data_classes import Prediction
from pipelines.default.detector import Detector


class FASTDetector(Detector):
    def _init(
        self,
        cuda: bool,
        pretrained: bool,
        assume_straight_pages: Optional[bool] = False,
        weights_path: Optional[str] = None,
        preserve_aspect_ratio: Optional[bool] = True,
    ):
        """
        Детектор текста FAST из библиотеки docTR (fast_base).
        Args:
            cuda (bool): Использование видеокарты
            assume_straight_pages (bool): Когда True, боксы прямоугольные. По умолчанию False
            pretrained (bool): Использовать предобученную модель
            weights_path (str): Использовать свои веса
            preserve_aspect_ratio (bool): При увеличении изображения не приводить к формату 1:1 ()
        Returns:
            out (FASTDetector): модель FAST детектора
        Пример:
        ```python
        model = FASTDetector(cuda=True, pretrained=True)
        result = model.detect(image)
        ```
        """

        fast = fast_base(pretrained=pretrained)

        if weights_path:
            weights = torch.load(weights_path)
            if 'state_dict' in weights:
                # Загрузка весов из lightning
                new_weights = OrderedDict()
                for k, v in weights['state_dict'].items():
                    new_key = k.replace('net.', '')
                    new_weights[new_key] = v
                fast.load_state_dict(new_weights)
            else:
                fast.load_state_dict(weights)

        fast = reparameterize(fast)

        self.net = detection_predictor(
            arch=fast,
            pretrained=False,
            assume_straight_pages=assume_straight_pages,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )

        self.polygons = not assume_straight_pages

        if cuda and torch.cuda.is_available():
            self.net.cuda()

    def detect(self, image: np.array) -> tuple[List[Prediction], np.array]:
        boxes = self.net([image])
        img_h, img_w, _ = image.shape
        detected: List[Prediction] = []
        for page in boxes:
            for box in page['words']:
                if self.polygons:
                    top_left_x, top_left_y = box[0]
                    top_right_x, top_right_y = box[1]
                    bottom_right_x, bottom_right_y = box[2]
                    bottom_left_x, bottom_left_y = box[3]

                    _, conf = box[4]
                else:
                    top_left_x, top_left_y, bottom_right_x, bottom_right_y, conf = box
                    top_right_x, top_right_y = bottom_right_x, top_left_y
                    bottom_left_x, bottom_left_y = top_left_x, bottom_right_y

                absolute_box = [
                    (int(top_left_x * img_w), int(top_left_y * img_h)),
                    (int(top_right_x * img_w), int(top_right_y * img_h)),
                    (int(bottom_right_x * img_w), int(bottom_right_y * img_h)),
                    (int(bottom_left_x * img_w), int(bottom_left_y * img_h)),
                ]
                relative_box = [
                    (top_left_x, top_left_y),
                    (top_right_x, top_right_y),
                    (bottom_right_x, bottom_right_y),
                    (bottom_left_x, bottom_left_y),
                ]
                detected.append(
                    Prediction(
                        absolute_box=absolute_box,
                        relative_box=relative_box,
                        score=conf,
                    )
                )
        return detected
