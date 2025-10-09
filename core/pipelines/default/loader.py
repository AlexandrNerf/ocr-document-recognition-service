from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
from pipelines.default.base import BasePipeline

class Loader(BasePipeline):
    def _run(self, data):
        return {'images': self.load()}
    
    @abstractmethod
    def load(self) -> list[np.array]:
        raise NotImplementedError

    @abstractmethod
    def end_stream(self) -> None:
        raise NotImplementedError
