import numpy as np
from pipelines.default.preprocessor import Preprocessor


class SimplePreprocessor(Preprocessor):

    def preprocessing(self, image: np.array) -> np.array:
        return image
