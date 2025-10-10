import numpy as np
import pytest
import rootutils

root = rootutils.setup_root(__file__, indicator=".core-root", pythonpath=True)

from pipelines.loaders.image_loader import ImageLoader


@pytest.mark.parametrize(
    'path',
    [
        f'{root}/assets/images',
        f'{root}/assets/pdf',
    ],
)
def test_image_loader_load(path):

    loader = ImageLoader(path)

    for i, img in enumerate(loader.load()):
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[2] == 3

    loader.end_stream()
