from typing import List

import numpy as np

from data.data_classes import Prediction
from pipelines.default.postprocessor import Postprocessor


class DummyPostprocessor(Postprocessor):
    def postprocessing(
        self, detections: List[Prediction], image: np.array
    ) -> List[Prediction]:
        pass
