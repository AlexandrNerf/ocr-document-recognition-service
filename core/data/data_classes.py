from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
import cv2
from pydantic import ConfigDict


@dataclass
class Prediction:  # noqa: WPS210
    absolute_box: list[Tuple[int, int]]  # Формат (tl_x, tl_y, br_x, br_y)
    score: Optional[float]  # Конфиденс
    relative_box: list[float] = None
    crop: Optional[np.array] = None  # Нарезанные кропы
    text: Optional[str] = None
    text_score: Optional[float] = None

    @property
    def center(self, format="absolute") -> Tuple[float, float]:
        box = self.relative_box if format == 'relative' else self.absolute_box
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        return center_x, center_y
    
    def relative_polygon(self, block_bbox):
        bx1, by1, bx2, by2 = block_bbox

        bw = bx2 - bx1
        bh = by2 - by1

        if bw == 0 or bh == 0:
            return None

        relative = []

        for x, y in self.absolute_box:
            rx = (x - bx1) / bw
            ry = (y - by1) / bh
            relative.append((rx, ry))

        return relative

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )



from dataclasses import dataclass

@dataclass
class Line:
    predictions: list[Prediction]
    polygon: list[tuple[int, int]] | None

    def to_json(self):
        return {
            "polygon": self.polygon,
            "words": [
                {
                    "polygon": to_int_tuples(p.absolute_box),
                    "text": p.text,
                    "confidence": p.text_score
                }
                for p in self.predictions
            ]
        }
@dataclass
class TextStructure:
    lines: list[Line] | None

    def build_from_predictions(self, predictions):
        def create_lines_from_sorted(predictions, threshold=10):
            lines = []
            current_line = [predictions[0]]
            prev_x, _ = predictions[0].center

            for pred in predictions[1:]:

                x, _ = pred.center

                if x < prev_x - threshold:
                    lines.append(Line(predictions=current_line, polygon=[]))
                    current_line = []

                current_line.append(pred)
                prev_x = x
            lines.append(Line(predictions=current_line, polygon=[]))
            return lines
        
        line_groups = create_lines_from_sorted(predictions)

        for group in line_groups:

            polygon = self._merge_polygons(
                [p.absolute_box for p in group.predictions]
            )

            self.lines.append(
                Line(
                    predictions=group.predictions,
                    polygon=polygon
                )
            )

    def _merge_polygons(self, polygons):

        xs = []
        ys = []

        for poly in polygons:
            for x, y in poly:
                xs.append(x)
                ys.append(y)

        return [
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        ]

    def to_json(self):

        return {
            "type": "text",
            "lines": [line.to_json() for line in self.lines]
        }
    
@dataclass
class TableCell:
    polygon: list[tuple[int, int]] | None = None
    predictions: list[Prediction] | None = None

    def text(self):
        return " ".join(p.text for p in self.predictions if p.text)
    
    def compute_polygon(self):

        if self.polygon:
            return self.polygon
        
        if not self.predictions:
            return []

        xs = []
        ys = []

        for p in self.predictions:
            for x, y in p.absolute_box:
                xs.append(x)
                ys.append(y)

        return [
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        ]


@dataclass
class TableRow:
    cells: list[TableCell] | None

@dataclass
class TableStructure:
    rows: list[TableRow] | None

    def to_json(self):

        return {
            "type": "table",
            "rows": [
                {
                    "cells": [
                        {
                            "polygon": to_int_tuples(cell.compute_polygon()),
                            "text": cell.text()
                        }
                        for cell in row.cells
                    ]
                }
                for row in self.rows
            ]
        }

@dataclass
class Block:
    polygon: list[tuple[int, int]]
    predictions: list[Prediction] | None = None
    type: str = 'Text'

    block_image: np.array = None
    structure: TextStructure | TableStructure | None = None

    @property
    def bbox(self):
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]

        return [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ]

    def calculate_relative_polygons(self):
        for prediction in self.predictions:
            prediction.relative_polygon(self.bbox)

    def set_image(self, doc_image: np.ndarray):
        x1, y1, x2, y2 = self.bbox
        self.block_image = doc_image[y1:y2, x1:x2]
    
    def to_structure(self):
        if self.type in ["Text", "SectionHeader", "Caption"]:

            structure = TextStructure(lines=[])
            structure.build_from_predictions(self.predictions)
            return structure

        elif self.type == "Table":
            self.structure = create_table_structure(self.predictions)

    def to_json(self):
        data = {
            "type": self.type,
            "polygon": self.polygon,
            "bbox": self.bbox
        }

        if self.structure:
            data["structure"] = self.structure.to_json()
        elif self.type == 'Figure':
            data["structure"] = {
                "type": "image",
                "image_array": self.block_image
            }
        elif self.predictions:
            # fallback: flat predictions
            data["structure"] = {
                "type": "predictions",
                "predictions": [
                    {
                        "polygon": to_int_tuples(p.absolute_box),
                        "text": p.text
                    } for p in self.predictions
                ]
            }

        return data
    
def to_int_tuples(poly: list[tuple[float, float]]) -> list[tuple[int,int]]:
    return [(int(x), int(y)) for x, y in poly]
class Document:
    """Документ с пайплайном выравнивания, детекции и OCR"""
    def __init__(self, image: np.ndarray):
        self.image = image
        self.blocks: list[Block] = []

    def to_json(self):
        for block in self.blocks:
            block.to_structure()
        return {
            "type": "document",
            "blocks": [block.to_json() for block in self.blocks],
            "width": self.image.shape[1],
            "height": self.image.shape[0]
        }

def get_bbox(pred):
    xs = [p[0] for p in pred.absolute_box]
    ys = [p[1] for p in pred.absolute_box]
    return min(xs), min(ys), max(xs), max(ys)

def assign_to_cells(row_preds, columns):

    cells = [TableCell(polygon=[], predictions=[]) for _ in columns]

    for pred in row_preds:

        x1, _, x2, _ = get_bbox(pred)
        center_x = (x1 + x2) / 2

        closest_col = min(
            range(len(columns)),
            key=lambda i: abs(center_x - columns[i])
        )

        cells[closest_col].predictions.append(pred)

    return cells

def detect_columns(predictions, x_threshold=20):

    columns = []

    for pred in predictions:

        center_x, _ = pred.center

        placed = False

        for col in columns:
            if abs(center_x - col) < x_threshold:
                placed = True
                break

        if not placed:
            columns.append(center_x)

    columns.sort()
    return columns

def group_rows(predictions, y_threshold=15):

    rows = []

    for pred in predictions:

        _, center_y = pred.center

        placed = False

        for row in rows:
            if abs(center_y - row["y"]) < y_threshold:
                row["preds"].append(pred)
                placed = True
                break

        if not placed:
            rows.append({
                "y": center_y,
                "preds": [pred]
            })

    return [r["preds"] for r in rows]

def create_table_structure(predictions):

    if not predictions:
        return TableStructure()

    rows_preds = group_rows(predictions)

    columns = detect_columns(predictions)

    rows = []

    for row_preds in rows_preds:

        # сортируем внутри строки
        row_preds = sorted(row_preds, key=lambda p: get_bbox(p)[0])

        cells = assign_to_cells(row_preds, columns)

        rows.append(TableRow(cells=cells))

    return TableStructure(rows=rows)