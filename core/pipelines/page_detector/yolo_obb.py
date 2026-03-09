import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from pathlib import Path
from pipelines.default.page_detector import PageDetector



class YOLOPageDetector(PageDetector):
    def _init(
        self,
        weights_path: str,
        threshold: float,
        mode: str
    ):
        """
        Класс детекции страницы с помощью YOLO8-obb.
    
        Args:
            weights_path (str): Путь до весов.
            threshold (float): Порог выбора боксов.
            mode (str): Режим обрезки - 'mask' и 'warp' 
        """
        self.model = YOLO(model=weights_path)
        self.threshold = threshold
        self.mode = mode


    def detect_page(self, image: np.array) -> tuple[np.ndarray, np.ndarray | None]:
        """Поиск кропов на изображении"""
        warping_params = None
        prediction = self.model.predict(image, conf=self.threshold, verbose=False)[0]
        if prediction.obb is None or len(prediction.obb) == 0:
            return image, warping_params

        boxes = prediction.obb.xyxyxyxy.cpu().numpy()
        scores = prediction.obb.conf.cpu().numpy()

        idx = scores.argmax()

        pts = boxes[idx].reshape(4,2)

        if self.mode == "warp":
            out, warping_params = self.warp_document(image, pts)
        elif self.mode == "mask":
            out = self.mask_document(image, pts)
        return out, warping_params

    def order_points(self, pts: np.array):
        rect = np.zeros((4,2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def warp_document(self, img, pts):

        rect = self.order_points(pts)
        tl, tr, br, bl = rect

        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        maxW = int(max(wA, wB))

        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        maxH = int(max(hA, hB))

        dst = np.array([
            [0,0],
            [maxW-1,0],
            [maxW-1,maxH-1],
            [0,maxH-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(img, M, (maxW, maxH))
        return warped, M


    def mask_document(self, img, pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        result = cv2.bitwise_and(img, img, mask=mask)

        return result