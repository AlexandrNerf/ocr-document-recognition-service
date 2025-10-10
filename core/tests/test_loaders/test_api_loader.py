import numpy as np
import pytest
from pipelines.loaders.api_loader import APILoader
from utils.base64utils import encode_image


@pytest.fixture
def api_loader():
    loader = APILoader()
    yield loader
    loader.end_stream()


@pytest.fixture
def encoded_image():
    with open("image_base64_test.txt", "r") as file:
        content = file.read()
    assert content is not None
    return content


def test_api_loader_load(api_loader, encoded_image):
    img_decoded = api_loader.load(base64=(encoded_image, "image"))

    assert isinstance(img_decoded, np.ndarray)
    assert img_decoded.dtype == np.uint8
    assert img_decoded.ndim == 3 and img_decoded.shape[2] == 3

    encoded_image_once_again = encode_image(img_decoded)
    with open("res.txt", "w") as f:
        f.write(encoded_image_once_again)
