from abc import ABC, abstractmethod

import numpy as np

from data.data_classes import Prediction
from pipelines.default.base import BasePipeline

class Postprocessor(BasePipeline):
    def _run(self, data):
        return {
            'post_detections': [
                    self.postprocessing(detections, image) 
                    for detections, image in zip(data['detections'], data['images'])
                ]
            }

    @abstractmethod
    def postprocessing(
        self, detections: list[Prediction], image: np.ndarray
    ) -> list[Prediction]:
        """Постпроцессинг и вырезание кропов с правильным ориентированием"""
        raise NotImplementedError
