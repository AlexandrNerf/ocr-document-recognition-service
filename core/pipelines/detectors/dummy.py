import numpy as np

from data.data_classes import Prediction
from pipelines.default.detector import Detector


class DummyDetector(Detector):
    def detect(self, image: np.array) -> list[Prediction]:
        return []
