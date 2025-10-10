from abc import abstractmethod

import numpy as np
from data.data_classes import Prediction
from pipelines.default.base import BasePipeline


class Detector(BasePipeline):

    def _run(self, data):
        return {"detections": [self.detect(image) for image in data["images"]]}

    @abstractmethod
    def detect(self, image: np.array) -> list[Prediction]:
        """Детектирование текста"""
        raise NotImplementedError
