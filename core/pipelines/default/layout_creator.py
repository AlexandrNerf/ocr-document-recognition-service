from abc import abstractmethod

import numpy as np
from data.data_classes import Prediction, Document
from pipelines.default.base import BasePipeline


class LayoutCreator(BasePipeline):
    def _run(self, data):
        return {"documents": [self.parse_layout(image, predictions) for image, predictions in zip(data["crop_images"], data['predictions'])]}

    @abstractmethod
    def parse_layout(self, image: np.array, predictions: list[Prediction]) -> list[Document]:
        """Парсинг структуры текста"""
        raise NotImplementedError