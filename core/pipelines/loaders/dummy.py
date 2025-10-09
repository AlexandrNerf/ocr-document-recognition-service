from typing import Iterator

import numpy as np

from shift_ocr.shift_ocr.loader import Loader


class DummyLoader(Loader):
    def load(self) -> Iterator[np.array]:
        pass

    def end_stream(self) -> None:
        pass
