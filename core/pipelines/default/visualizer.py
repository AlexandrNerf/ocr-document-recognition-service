from abc import abstractmethod

import numpy as np
from data.data_classes import Prediction
from pipelines.default.base import BasePipeline


class Visualizer(BasePipeline):

    def _run(self, data):
        self.visualize(data)

    @abstractmethod
    def visualize(self, data: dict) -> None:
        """Вывод изображений с использованием визуалайзера (plotly и др.)
        Args:
            data (dict): словарь с результатами работы модели

        Returns:
            out (None): Выведенные изображения
        """
        raise NotImplementedError
