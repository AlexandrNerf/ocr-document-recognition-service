import pathlib
from itertools import islice

import cv2
import numpy as np
import pytest
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-roots", pythonpath=True)

from shift_ocr.loaders.eval_loader import EvalLoader
from shift_ocr.loaders.image_loader import ImageLoader
from shift_ocr.shift_ocr.data_classes import Detection
from shift_ocr.utils.read_detection import read_detections


@pytest.mark.parametrize(
    'img_dir, label_dir',
    [(rf'{root}/assets/images', rf'{root}/assets/data')],
)
def test_read_detections_one_image(img_dir, label_dir):
    img_path = pathlib.Path(img_dir)
    label_path = pathlib.Path(label_dir)

    assert img_path.is_dir()
    assert label_path.is_dir()

    for image_file in islice(pathlib.Path.glob(img_path, '*.png'), 10):
        image = cv2.imread(image_file)
        assert image is not None
        image_file_name = image_file.stem
        label_path = label_path.joinpath(image_file_name + '.txt')

        if not label_path.exists():
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
            labels = list(map(str.strip, labels))

        gt_img, gt_detections = read_detections(
            [image],
            labels,
            labels_from_file=False,
            box_format='yolo',
        )

        assert np.allclose(gt_img[0], image)
        assert all(isinstance(det, Detection) for det in gt_detections[0])
