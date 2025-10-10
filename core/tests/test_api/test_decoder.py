import base64

import cv2
import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-roots", pythonpath=True)

from shift_ocr.utils.base64utils import decode_image


@pytest.fixture
def encode_image():
    img_encoded = cv2.imread("../../assets/images/DDI_0.png")
    assert img_encoded is not None
    _, img_encoded = cv2.imencode(".png", img_encoded)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def test_decode_image(encode_image):
    img_decoded = decode_image(encode_image)
    assert isinstance(img_decoded, np.ndarray)
    assert img_decoded.dtype == np.uint8
    assert img_decoded.ndim == 3 and img_decoded.shape[2] == 3
