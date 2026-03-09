from pipelines.default.doc_unwrapper import DocUnwrapper
from paddleocr import TextImageUnwarping
import numpy as np


class UVDocUnwrapper(DocUnwrapper):
    def _init(
        self,
        iterations: int,
    ):
        """
        Модель выравнивания документов с помощью 
    
        Args:
            iterations (int): количество итераций выравнивания
        """
        self.model = TextImageUnwarping(model_name="UVDoc")
        self.iterations = iterations


    def unwrapping(self, image: np.array, warp: np.ndarray | None) -> np.array:
        """Преобразует изображение на основе предыдущего пайплайна.
        
        Если был найден документ и было проведено афинное преобразование, то warp не является None.
        В таком случае количество итераций выравнивания равно 1, иначе значению iterations.
        """
        result = image
        if warp is not None:
            return self.model.predict(result, batch_size=1)[0]['doctr_img']
        for _ in range(self.iterations):
            result = self.model.predict(result, batch_size=1)[0]['doctr_img']
        return result
        