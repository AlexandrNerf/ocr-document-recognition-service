import numpy as np

from pipelines.default.preprocessor import Preprocessor
from pipelines.default.base import BasePipeline

class SimplePreprocessor(Preprocessor):

    def preprocessing(self, image: np.array) -> np.array:
        return image
