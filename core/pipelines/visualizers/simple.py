import cv2
import numpy as np
import plotly.io as pio
import tempfile
import webbrowser
import panel as pn
from rich.console import Console
from data.data_classes import Prediction
from plotly.subplots import make_subplots
from pipelines.default.visualizer import Visualizer
from utils.BBoxVisualizer import BoundingBoxVisualizer

console = Console()

class SimpleVisualizer(Visualizer):
    def _init(self):
        self.figs: list = []
        self.max_width: int | None = 1000
        pn.extension('plotly')

    def update_figures(self, images: list[np.array], predictions: list[list[Prediction]]):
        self.max_width = min(self.max_width, max(image.shape[1] for image in images))
        for image, prediction in zip(images, predictions):
            self.figs.append(BoundingBoxVisualizer.show_image(image, prediction, self.max_width))

    def visualize(self, images: list[np.array], predictions: list[list[Prediction]]) -> None:
        """Вывод изображений с использованием BoundingBoxVisualizer
        Args:
            images (list[np.array]): Список изображений
            post_detections (list[Prediction]): Детекции с кропами

        Returns:
            out (None): Выведенные изображения
        """
        print(images)
        print(predictions)
        self.update_figures(images, predictions)

        if len(self.figs) < 1:
            return Exception('Not founded images to visualize')
            
        panel = pn.Column()
        for fig in self.figs:
            panel.append(pn.pane.Plotly(fig, config={"responsive": True}))
        
        panel.show()

        
        
            
