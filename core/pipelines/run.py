from pipelines.default.detector import Detector
from pipelines.default.loader import Loader
from pipelines.default.postprocessor import Postprocessor
from pipelines.default.preprocessor import Preprocessor
from pipelines.default.recognizer import Recognizer
from pipelines.default.visualizer import Visualizer


class CorePipeline:
    def __init__(  # noqa: WPS211
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
            self._visualizer,
        ]

    def run(self) -> None:
        """Запускает инференс модели на загруженных из лоадера фотках.
        Итоговый результат выводится в виде plotly страницы.
        """
        for pipeline in self._pipelines:
            self._data = pipeline.run(self._data)

    def predict(self, image) -> dict:
        """Предсказание для одного изображения (без визуализатора).
        
        Args:
            image: numpy array изображения
            
        Returns:
            dict с ключами:
                - images: list[np.array] - список изображений
                - predictions: list[list[Prediction]] - список предсказаний
        """
        self._data = {"images": [image] if not isinstance(image, list) else image}
        
        # Пропускаем loader, т.к. изображение уже загружено
        pipelines_without_loader = [
            self._preprocessor,
            self._detector,
            self._postprocessor,
            self._recognizer,
        ]
        
        for pipeline in pipelines_without_loader:
            self._data = pipeline.run(self._data)
        
        return {
            "images": self._data.get("images", []),
            "predictions": self._data.get("predictions", []),
        }