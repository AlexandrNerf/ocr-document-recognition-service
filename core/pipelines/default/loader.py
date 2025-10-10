from abc import abstractmethod

import numpy as np
from pipelines.default.base import BasePipeline


class Loader(BasePipeline):
    def _run(self, data):
        return {"images": self.load(data)}

    @abstractmethod
    def load(self) -> list[np.array]:
        raise NotImplementedError

    @abstractmethod
    def end_stream(self) -> None:
        raise NotImplementedError
