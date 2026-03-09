from abc import abstractmethod

import numpy as np
from pipelines.default.base import BasePipeline


class DocUnwrapper(BasePipeline):
    def __init__(self, *args, **kwargs):
        self.run_no_wrap = kwargs.pop('run_no_wrap')
        super().__init__(*args, **kwargs)
        
    def _run(self, data):
        if self.run_no_wrap:
            return {}
        return {'crop_images': [self.unwrapping(image, warp) for image, warp in zip(data['crop_images'], data['warp_params'])]}

    @abstractmethod
    def unwrapping(self, image: np.array, warp: np.ndarray | None) -> np.array:
        """Выравнивание изображения страницы."""
        raise NotImplementedError
