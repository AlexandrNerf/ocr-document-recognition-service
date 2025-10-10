from abc import abstractmethod

import numpy as np
from pipelines.default.base import BasePipeline


class Preprocessor(BasePipeline):
    def _run(self, data):
        return {"images": [self.preprocessing(image) for image in data["images"]]}

    @abstractmethod
    def preprocessing(self, image: np.array) -> np.array:
        """Начальная обработка входной картинки"""
        raise NotImplementedError
