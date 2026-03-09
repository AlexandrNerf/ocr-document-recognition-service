from abc import abstractmethod

import numpy as np
from pipelines.default.base import BasePipeline


class PageDetector(BasePipeline):
    def __init__(self, *args, **kwargs):
        self.run_no_page_det = kwargs.pop('run_no_page_det')
        super().__init__(*args, **kwargs)
        

    def _run(self, data):
        new_data = {
            'crop_images': [],
            'warp_params': [],
        }
        if self.run_no_page_det:
            return {
                'crop_images': data['images'],
                'warp_params': [None for _ in range(len(data['images']))]
            }
        for image in data['images']:
            crop_image, warp_param = self.detect_page(image)
            new_data['crop_images'].append(crop_image)
            new_data['warp_params'].append(warp_param)
        return new_data

    @abstractmethod
    def detect_page(self, image: np.array) -> tuple[np.ndarray, np.ndarray | None]:
        """Поиск кропов на изображении"""
        raise NotImplementedError
