from abc import ABC, abstractmethod
from typing import List

import torch
from fuzzywuzzy import fuzz
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics

from data.data_classes import Prediction


class Metrics(ABC):

    @abstractmethod
    def process(self, detections: List[Prediction], gt_detections: List[Prediction]):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class UltralyticsMetrics(Metrics):

    def __init__(self):
        self.metrics = DetMetrics()
        self.validator = DetectionValidator()

    def __str__(self):
        return 'UltralyticsMetrics: recall, mAP50, mAP50-95'

    def process(self, detections: List[Prediction], gt_detections: List[Prediction]):
        pred_conf = [detection.score for detection in detections]

        pred_boxes = [detection.absolute_box for detection in detections]

        # (tl_x, tl_y, br_x, br_y) in this approach
        pred_boxes = [
            (box[0][0], box[0][1], box[2][0], box[2][1]) for box in pred_boxes
        ]

        pred_boxes = [
            (*box, detection.score, 0) for detection, box in zip(detections, pred_boxes)
        ]

        gt_boxes = [detection.absolute_box for detection in gt_detections]
        gt_boxes = [(box[0][0], box[0][1], box[2][0], box[2][1]) for box in gt_boxes]

        pred_cls = torch.zeros(len(pred_boxes))
        gt_cls = torch.zeros(len(gt_boxes))

        gt_boxes = torch.tensor(gt_boxes)
        pred_boxes = torch.tensor(pred_boxes)
        pred_conf = torch.tensor(pred_conf)

        tp = self.validator._process_batch(pred_boxes, gt_boxes, gt_cls).int()
        self.metrics.process(tp, pred_conf, pred_cls, gt_cls)

    def get_metrics(self):
        result = ''
        for key, item in self.metrics.results_dict.items():
            result += f'{key}: {item} \n'
        return result


class OCRMetrics:
    """
    :param float iou_threshold: Порог IoU от которого мы считаем детекцию успешной. По умолчанию - 0.5.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.tp_words = 0
        self.total_words = 0
        self.tp_chars = 0
        self.total_chars = 0

    def __str__(self):
        return (
            "OCR metrics: WRR (Word Recognition Rate), CRR (Character Recognition Rate)"
        )

    def calculate_iou(self, pred_box, gt_box):
        """
        Рассчитывает IoU для двух bounding box.
        Формат bounding box: (tl_x, tl_y, br_x, br_y)
        """
        x1, y1, x2, y2 = pred_box
        x1_gt, y1_gt, x2_gt, y2_gt = gt_box

        xi1 = max(x1, x1_gt)
        yi1 = max(y1, y1_gt)
        xi2 = min(x2, x2_gt)
        yi2 = min(y2, y2_gt)

        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0

    def process(self, detections: List[Prediction], gt_detections: List[Prediction]):
        """
        Обрабатывает предсказанные и ground truth детекции для расчета метрик.
        """
        pred_boxes = [detection.absolute_box for detection in detections]

        pred_boxes = [
            (box[0][0], box[0][1], box[2][0], box[2][1]) for box in pred_boxes
        ]

        pred_texts = [det.text.lower() for det in detections]

        gt_boxes = [detection.absolute_box for detection in gt_detections]

        gt_boxes = [(box[0][0], box[0][1], box[2][0], box[2][1]) for box in gt_boxes]

        gt_texts = [gt.text.lower() for gt in gt_detections]

        self.total_words += len(gt_texts)
        self.total_chars += sum(len(gt_text) for gt_text in gt_texts)

        for gt_box, gt_text in zip(gt_boxes, gt_texts):
            max_iou = 0
            best_match_index = -1

            for i, pred_box in enumerate(pred_boxes):
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match_index = i

            if max_iou > self.iou_threshold and best_match_index != -1:
                pred_text = pred_texts[best_match_index]
                if pred_text == gt_text:
                    self.tp_words += 1

                # Расстояние Левенштейна
                distance = fuzz.ratio(pred_text, gt_text)
                self.tp_chars += (distance / 100) * len(gt_text)

    def get_wrr(self):
        """Возвращает Word Recognition Rate (WRR)."""
        return self.tp_words / self.total_words if self.total_words > 0 else 0

    def get_crr(self):
        """Возвращает Character Recognition Rate (CRR)."""
        return self.tp_chars / self.total_chars if self.total_chars > 0 else 0

    def get_metrics(self):
        """Возвращает строку с метриками WRR и CRR."""
        return f'WRR: {self.get_wrr():.2f}, CRR: {self.get_crr():.2f}'
