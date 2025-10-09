import os
from typing import Iterator

import cv2
import fitz
import numpy as np

from pipelines.default.loader import Loader


class ImageLoader(Loader):

    def _init(self, image_path: str, pdf_zoom: float = 1):
        self.img_dir = image_path
        self.pdf_zoom = pdf_zoom
        self.files = [
            os.path.join(self.img_dir, file) for file in os.listdir(self.img_dir)
        ]
        self.images = []
        self.stream = True

    def load(self) -> list[np.ndarray]:
        for file_path in self.files:
            if not self.stream:
                break
            ext = os.path.splitext(file_path)[1]
            if ext == '.pdf':
                image = self._pdf_to_img(file_path)
            else:
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.images.append(image)
        return self.images

    def _pdf_to_img(self, path):
        doc = fitz.open(path)
        page = doc.load_page(0)
        matrix = fitz.Matrix(self.pdf_zoom, self.pdf_zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        doc.close()
        return img_array

    def end_stream(self) -> None:
        self.stream = False
