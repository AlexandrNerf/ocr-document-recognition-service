import copy
from collections import OrderedDict
from typing import list

import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from data.data_classes import Detection


class ImgDetectionVisualize:
    def __init__(
        self,
        images: list[np.ndarray],
        detections: list[list[Detection]],
        cache_size: int = 10,
    ):
        """Класс для визуализации детекций на изображениях с возможностью переключения между изображениями
            и выбора детекций для отображения.

        Args:
            images (List(np.ndarray)): - Список изображений.
            param detections (List(List(Detection))): - Список детекций для каждого изображения.
            cache_size (int): - Размер кеша для хранения обработанных изображений и чекбоксов.
            type_bbox: (str): - Формат представления ограничивающих рамок в Detection ('pascal_voc' или 'yolo'), по умолчанию 'pascal'.
        """
        self.images = images
        self.detections = copy.deepcopy(detections)  # Для избежания влияния на оригинал
        self.cache_size = cache_size
        self.cur_ind = 0

        self.checkboxes_cache = OrderedDict()
        self.image_cache = OrderedDict()

        self.output = widgets.Output()
        self.prev_button = widgets.Button(description='Предыдущее')
        self.next_button = widgets.Button(description='Следующее')
        self.checkboxes_box = widgets.GridBox()
        self.checkboxes_box.layout.grid_template_columns = 'repeat(4, auto)'
        self.index_label = widgets.Label(value=str(self.cur_ind))

        self.prev_button.on_click(lambda _: self.prev_image())
        self.next_button.on_click(lambda _: self.next_image())

        self.update_checkboxes()
        self.update_index_label()
        self.draw_detection()

    def paint(self, detections: list[Detection], image: np.array) -> np.array:
        img_with_boxes = image.copy()
        for detection in detections:
            cv2.polylines(
                img_with_boxes,
                np.array(detection.absolute_box, dtype=np.uint32),
                isClosed=True,
                color=(0, 255, 0),
                thickness=1,
            )
        return img_with_boxes

    def visualize(self, image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise Exception('img is not type np.ndarray')

        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


    def update_checkboxes(self):
        """Обновляет чекбоксы, соответствующие детекциям текущего изображения."""
        if len(self.checkboxes_cache) > self.cache_size:
            self.checkboxes_cache.pop(next(iter(self.checkboxes_cache)))

        if self.cur_ind in self.checkboxes_cache:
            self.checkboxes_box.children = self.checkboxes_cache[self.cur_ind]

        else:
            checkboxes = [
                widgets.Checkbox(value=True, description=f'Detection {i}')
                for i in range(len(self.detections[self.cur_ind]))
            ]
            for checkbox in checkboxes:
                checkbox.observe(lambda _: self.draw_detection(), names='value')

            self.checkboxes_cache[self.cur_ind] = checkboxes
            self.checkboxes_box.children = checkboxes

    def update_index_label(self):
        """Обновляет отображаемый номер текущего изображения."""
        self.index_label.value = f'Картинка номер {self.cur_ind + 1}'

    def draw_detection(self):
        """Отображает текущее изображение с выбранными детекциями."""

        if len(self.image_cache) > self.cache_size:
            self.image_cache.pop(next(iter(self.checkboxes_cache)))

        selected_detections = [
            det
            for det, chbox in zip(
                self.detections[self.cur_ind], self.checkboxes_box.children
            )
            if chbox.value
        ]

        if (
            self.cur_ind in self.image_cache
            and self.image_cache[self.cur_ind][0] == selected_detections
        ):
            image_result = self.image_cache[self.cur_ind][1]
        else:

            image = self.images[self.cur_ind]
            image_result = self.paint(selected_detections, image)
            self.image_cache[self.cur_ind] = (selected_detections, image_result)

        with self.output:
            self.output.clear_output(wait=True)
            self.visualize(image_result)

    def next_image(self):
        """Переключает отображение на следующее изображение, если возможно."""
        if self.cur_ind < len(self.images) - 1:
            self.cur_ind += 1
            self.update_checkboxes()
            self.update_index_label()
            self.draw_detection()

    def prev_image(self):
        """Переключает отображение на предыдущее изображение, если возможно."""
        if self.cur_ind > 0:
            self.cur_ind -= 1
            self.update_checkboxes()
            self.update_index_label()
            self.draw_detection()

    def display(self):
        """Отображает элементы управления и визуализации в Jupyter Notebook."""
        control_box = widgets.HBox([self.prev_button, self.next_button])
        display(
            widgets.VBox(
                [control_box, self.index_label, self.checkboxes_box, self.output]
            )
        )
