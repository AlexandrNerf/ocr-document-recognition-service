import numpy as np

from pipelines.default.preprocessor import Preprocessor


class DummyPreprocessor(Preprocessor):
    def preprocessing(self, image: np.array) -> np.array:
        pass
