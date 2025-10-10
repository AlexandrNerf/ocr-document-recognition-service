import cv2
import numpy as np

from data.data_classes import Prediction
from pipelines.default.detector import Detector
from pipelines.default.loader import Loader
from pipelines.default.postprocessor import Postprocessor
from pipelines.default.preprocessor import Preprocessor
from pipelines.default.recognizer import Recognizer
from pipelines.default.visualizer import Visualizer
from utils.BBoxVisualizer import BoundingBoxVisualizer
from plotly.subplots import make_subplots

class CorePipeline:
    def __init__(
        self,
        loader: Loader,
        detector: Detector,
        recognizer: Recognizer,
        preprocessor: Preprocessor,
        postprocessor: Postprocessor,
        visualizer: Visualizer,
    ): 
        self._data: dict = {}
        self._loader = loader
        self._detector = detector
        self._recognizer = recognizer
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._visualizer = visualizer

        self._pipelines = [
            self._loader,
            self._preprocessor,
            self._detector,
            self._postprocessor,
            self._recognizer,
            self._visualizer
        ]
        

    def run(self) -> None:
        """Запускает инференс модели на загруженных из лоадера фотках. 
            Итоговый результат выводится в виде plotly страницы.
        """
        for pipeline in self._pipelines:
            self._data = pipeline.run(self._data)
        

