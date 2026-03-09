import numpy as np
import panel as pn
import json
from data.data_classes import Prediction
from pipelines.default.visualizer import Visualizer
from utils.BBoxVisualizer import BoundingBoxVisualizer

pn.extension("plotly")


class SimpleVisualizer(Visualizer):  # noqa: WPS338
    def _init(self):
        self.figs: list = []
        self.max_width: int | None = 1000

    def update_figures(
        self, images: list[np.array], predictions: list[list[Prediction]]
    ):
        self.max_width = min(
            self.max_width, max(image.shape[1] for image in images)
        )  # noqa: WPS221, E501
        for image, prediction in zip(images, predictions):
            self.figs.append(
                BoundingBoxVisualizer.show_image(image, prediction, self.max_width)
            )

    import plotly.graph_objects as go

    def create_structure_figures(self, images: list[np.array], documents):
        doc_figs = []
        for img, document in zip(images, documents):
            doc_figs.append(BoundingBoxVisualizer.visualize_full_document(img, document))
        return doc_figs


    def visualize(self, data: dict) -> None:
        """Вывод изображений с использованием BoundingBoxVisualizer
        Args:
            data (dict): словарь с результатом работы модели

        Returns:
            out (None): Выведенные изображения
        """
        self.update_figures(data['crop_images'], data['predictions'])

        for i, doc in enumerate(data['documents']):
            with open(f'assets/results/{i}.json', 'w+', encoding='utf-8') as f:
                json.dump(doc.to_json(), f)


        if len(self.figs) < 1:
            return Exception("Not founded images to visualize")

        panel = pn.Column()
        for fig in self.figs:
            panel.append(pn.pane.Plotly(fig, config={"responsive": True}))
        for fig in self.create_structure_figures(data['crop_images'], data['documents']):
            panel.append(pn.pane.Plotly(fig, config={"responsive": True}))

        panel.show()
