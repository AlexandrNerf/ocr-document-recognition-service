from abc import ABC, abstractmethod
import numpy as np

from data.data_classes import Prediction
from pipelines.default.base import BasePipeline

class Visualizer(BasePipeline):

    def _run(self, data):
        self.visualize(data['images'], data['predictions'])

    @abstractmethod
    def visualize(self, images: list[np.array], predictions: list[list[Prediction]]) -> None:
        """Вывод изображений с использованием визуалайзера (plotly и др.)
        Args:
            images (list[np.array]): Список изображений
            post_detections (list[Prediction]): Детекции с кропами

        Returns:
            out (None): Выведенные изображения
        """
        raise NotImplementedError
