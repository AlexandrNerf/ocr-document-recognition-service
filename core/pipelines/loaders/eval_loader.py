import logging
import os
from typing import Iterator

import chardet
import cv2
import numpy as np
from pipelines.default.loader import Loader
from utils.read_detection import read_detections


def get_file_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result["encoding"] or "utf-8"


class EvalLoader(Loader):

    def _init(self, image_path: str, label_path: str, labels_format: str):
        self.img_dir = image_path
        self.label_dir = label_path
        self.labels_format = labels_format

        self.files = [
            os.path.join(self.img_dir, file) for file in os.listdir(self.img_dir)
        ]
        self.labels = [
            os.path.join(
                self.label_dir, os.path.splitext(file)[0] + ".txt"
            )  # noqa: WPS221
            for file in os.listdir(self.img_dir)
        ]

        assert len(self.files) == len(
            self.labels
        ), "Number of images and labels must be equal"

        self.stream = True

    def __len__(self) -> int:
        return len(self.files)

    def load(self) -> Iterator[tuple[np.ndarray, list[str]]]:  # noqa: WPS234
        for file_path, label_path in zip(self.files, self.labels):
            if not self.stream:
                break
            image = cv2.imread(file_path)
            if os.path.exists(label_path):
                encoding = get_file_encoding(label_path)
                with open(label_path, "r", encoding=encoding) as f:
                    labels = f.readlines()
                    labels = list(map(str.strip, labels))
                    try:
                        _, gt_detections = read_detections(
                            [image],
                            labels,
                            labels_from_file=False,
                            box_format=self.labels_format,
                        )
                    except Exception as e:
                        logging.error(f"Error with {file_path} - {e}! Skipping...")
                        continue
                    if len(gt_detections) == 0:
                        continue
                    gt_detections = gt_detections[0]
            else:
                logging.warning(
                    f"Label {label_path} doesn`t exist! Skipping {file_path}..."
                )

            yield image, gt_detections

    def end_stream(self) -> None:
        self.stream = False
