import numpy as np

from data.data_classes import Prediction
from pipelines.default.recognizer import Recognizer


class DummyRecognizer(Recognizer):
    def recognize(self, image: np.ndarray, detections: list[Prediction]) -> list[str]:
        pass
