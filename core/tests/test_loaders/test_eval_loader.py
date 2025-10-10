from itertools import islice

import numpy as np
import pytest
import rootutils

root = rootutils.setup_root(__file__, indicator=".core-root", pythonpath=True)

from data.data_classes import Predictions
from pipelines.loaders.eval_loader import EvalLoader


@pytest.mark.parametrize(
    "img_path, label_path",
    [(rf"{root}/assets/images", rf"{root}/assets/data")],
)
def test_image_loader_load(img_path, label_path):
    loader = EvalLoader(img_path, label_path, "yolo")
    for img, labels in islice(loader.load(), 10):
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[2] == 3
        assert isinstance(labels, list)
        assert all(isinstance(el, Predictions) for el in labels)
    loader.end_stream()
