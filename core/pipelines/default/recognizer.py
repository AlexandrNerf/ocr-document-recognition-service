from abc import ABC, abstractmethod
import numpy as np

from data.data_classes import Prediction
from pipelines.default.base import BasePipeline

class Recognizer(BasePipeline):

    def _run(self, data):
        return {
            'predictions': [self.recognize(post_detections) for post_detections in data['post_detections']]
        }

    @abstractmethod
    def recognize(self, post_detections: list[Prediction]) -> list[Prediction]:
        """Распознавание текста с детектированных областей
        Args:
            post_detections (list[Prediction]): Детекции с кропами

        Returns:
            out (list[Prediction]): Список распознанных текстов
        """
        raise NotImplementedError
