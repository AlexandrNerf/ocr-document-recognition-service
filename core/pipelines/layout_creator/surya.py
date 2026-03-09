from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from data.data_classes import Prediction, Document, Block
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

from pipelines.default.layout_creator import LayoutCreator

class LayoutParser(LayoutCreator):
    def _init(self):
        """
        Cоздание шаблона документа с помощью Surya LayoutPredictor.
        """
        self.layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))

    def parse_layout(self, image: np.array, predictions: list[Prediction]) -> Document:
        """
        Делает разметку документа на блоки
        возвращает объект Document с блоками (bbox и тип)
        """
        pil_image = Image.fromarray(image)
        layout_result = self.layout_predictor([pil_image])[0]

        return self.build_document(
            image=image,
            layout_result=layout_result,
            predictions=predictions
        )

    def build_document(
        self,
        image: np.ndarray,
        layout_result,
        predictions: List[Prediction],
    ) -> Document:

        doc = Document(image)

        blocks = self.layout_to_blocks(layout_result)
        doc.blocks.extend(blocks)
        entire_document = self.assign_predictions_to_blocks(
            predictions,
            doc.blocks,
        )

        if entire_document:
            h, w = image.shape[:2]
            doc.blocks.append(
                Block(
                    polygon=[
                        (0,0),
                        (w,0),
                        (w,h),
                        (0,h)
                    ],
                    type="not_marked_document",
                    predictions=entire_document
                )
            )

        return doc
    
    def layout_to_blocks(self, layout_result):
        blocks = []

        for layout_box in layout_result.bboxes:
            polygon = [tuple(map(lambda x: max(x, 0), map(int, poly))) for poly in layout_box.polygon]
            print(polygon)
            block = Block(
                polygon=polygon,
                type=layout_box.label,
                predictions=[]
            )

            blocks.append(block)

        return blocks

    def assign_predictions_to_blocks(
        self,
        predictions: List[Prediction],
        blocks: List[Block],
    ):

        entire_document = []

        for pred in predictions:
            center = pred.center
            assigned = False

            for block in blocks:
                if self._point_in_bbox(center, block.bbox):
                    pred.relative_polygon(
                        block.bbox
                    )
                    block.predictions.append(pred)
                    assigned = True
                    break

            if not assigned:
                entire_document.append(pred)

        return entire_document
    
    def _point_in_bbox(self, point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
