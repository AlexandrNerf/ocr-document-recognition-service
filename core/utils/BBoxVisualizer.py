import os
from random import randint as rand

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from data.data_classes import Prediction

from .read_detection import crop_masked_rectangle, read_detections


class BoundingBoxVisualizer:
    def __init__(
        self,
        dataset_path: list[np.ndarray] | list[Image.Image],
        predictions: list[list[Prediction]],
        max_crop_images: int = -1,
    ):
        """
        Визуализатор детекций на основе plotly.

        Args:
            dataset_path (list[np.ndarray] | list[Image.Image]): Путь к изображениям (или список с картинками)
            predictions (list[list[Prediction]]): Список предсказаний для каждой картинки
            max_crop_images (int): Допустимое число выводимых кропов
            box_format (str): Тип боксов (если вводятся пути)

        Пример 1 (Со списками):
        ```python

        images = 'path/to/images'
        datas = 'path/to/data'
        imgs, predictions = read_detections(images, detections, box_format='yolo')
        visualizer = BoundingBoxVisualizer(imgs, predictions)
        visualizer.visualize(index=0)
        ```

        """
        self.dataset_path = dataset_path
        self.image_files = dataset_path

        self.predictions = predictions
        self.max_crop_images = max_crop_images

        self.current_index = 0

        self.image_params = {  # Настройки image_layout
            'source': 0,
            'x': 0,
            'y': 0,
            'sizex': 0,
            'sizey': 0,
            'xref': 'x',
            'yref': 'y',
            'xanchor': 'left',
            'yanchor': 'top',
            'layer': 'below',
        }

    # Данные методы повторяют основные функции, однако статически определены
    @staticmethod
    def show_image(
        image_path: str | np.ndarray | Image.Image, predictions: list[Prediction], max_width: int
    ):
        """
        Отображает изображения с боксами, а также отображает legend с подписанным номером детекции
        и возможностью изменения отображения детекции.

        Args:
            image_path (str | np.ndarray | Image.Image): - Путь к изображению.
            predictions (List(Detection)): - Список предсказанных объектов.
        """
        

        if isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = Image.open(image_path)

        width, height = image.size
        scale = max_width / width    
        width, height = int(width * scale), int(height * scale)
        image = image.resize(
            (int(width), int(height))
        )
        image_params = {  # Настройки image_layout
                    'x': 0,
                    'y': 1,
                    'sizex': 1,
                    'sizey': 1,
                    'xref': 'x domain',
                    'yref': 'y domain',
                    'sizing': 'stretch', 
                    'layer': 'below',
                }
        fig = go.Figure(go.Image(z=image))

        fig.update_layout(
            width=width,
            height=height,
            xaxis=dict(scaleanchor='y', constrain='domain'),
            yaxis=dict(scaleanchor='x', constrain='domain', autorange='reversed')
        )
        num_detection = 0
        for pred in predictions:
            num_detection += 1
            text = pred.text
            det_confidence = pred.score
            ocr_confidence = pred.text_score
            color = f'rgba({rand(0, 255)}, {rand(0, 255)}, {rand(0, 255)}, 1)'

            bbox_x = [x * scale for x, _ in pred.absolute_box]
            bbox_y = [y * scale for _, y in pred.absolute_box]
            bbox_x.append(bbox_x[0])
            bbox_y.append(bbox_y[0])
            text_x, text_y = bbox_x[0], bbox_y[0]

            fig.add_trace(  # Отрисовка детекций на изображении.
                go.Scatter(
                    x=bbox_x,
                    y=bbox_y,
                    text=f'{text}',
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'Box {num_detection}: {text}',
                )
            )
            fig.add_trace(  # Отрисовка confidence модели на изображении.
                go.Scatter(
                    x=[text_x],
                    y=[text_y],
                    text=[f'({det_confidence:.2f}, {ocr_confidence:.2f})'],
                    mode='text',
                    textfont=dict(color='black', size=8),
                    name=f'confidences: ({det_confidence:.2f}, {ocr_confidence:.2f})',
                )
            )

        fig.update_layout(  # Отрисовка легенды с информацией о детекциях и confidence.
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(116, 116, 116, 0.61)',
            ),
            margin=dict(l=0, r=200, t=0, b=0),
        )
        return fig

    @staticmethod
    def show_crop(image_path: str | np.ndarray | Image.Image, prediction: Prediction):
        """Отображает обрезанное изображение вокруг предсказанного объекта. Статический метод

        Args:
            image_path (str | np.ndarray | Image.Image): - Путь к изображению.
            prediction (Detection): - Список предсказанных объектов.
        """
        image_params = {  # Настройки image_layout
            'source': 0,
            'x': 0,
            'y': 0,
            'sizex': 0,
            'sizey': 0,
            'xref': 'x',
            'yref': 'y',
            'xanchor': 'left',
            'yanchor': 'top',
            'layer': 'below',
        }

        if isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = Image.open(image_path)

        crop_image = crop_masked_rectangle(image, prediction.absolute_box)
        crop_width, crop_height = crop_image.size
        image_params.update(
            dict(
                zip(
                    ['source', 'x', 'y', 'sizex', 'sizey'],
                    [crop_image, 0, crop_height, crop_width, crop_height],
                )
            )
        )

        fig = go.Figure()
        fig.add_layout_image(**image_params)

        fig.update_xaxes(visible=False, range=[0, crop_width])
        fig.update_yaxes(visible=False, range=[0, crop_height])

        fig.update_layout(
            width=500,
            height=crop_height * (500 // crop_width),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.show()

    def _draw_image_with_bboxes(
        self, image_path: str | Image.Image | np.ndarray, predictions: list[Prediction]
    ):
        """Отображает изображения с боксами, а также отображает legend с подписанным номером детекции
        и возможностью изменения отображения детекции.

        Args:
            image_path (str | np.ndarray | Image.Image): - Путь к изображению.
            predictions (List(Detection)): - Список предсказанных объектов.
        """
        if isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = Image.open(image_path)

        width, height = image.size
        offset = (
            0.15 * width
        )  # Параметр нужен для создания смещения, чтобы легенда не перекрывала изображение.
        self.image_params.update(
            dict(
                zip(
                    ['source', 'x', 'y', 'sizex', 'sizey'],
                    [image, offset, height, width, height],
                )
            )
        )
        fig = go.Figure()

        fig.add_layout_image(**self.image_params)

        num_detection = 0
        for pred in predictions:
            num_detection += 1
            confidence = pred.score
            color = f'rgba({rand(0, 255)}, {rand(0, 255)}, {rand(0, 255)}, 1)'

            bbox_x = [
                x + offset for x, _ in pred.absolute_box
            ]
            bbox_y = [height - y for _, y in pred.absolute_box]
            bbox_x.append(bbox_x[0])
            bbox_y.append(bbox_y[0])
            text_x, text_y = bbox_x[0], bbox_y[0]

            fig.add_trace(  # Отрисовка детекций на изображении.
                go.Scatter(
                    x=bbox_x,
                    y=bbox_y,
                    mode='lines',
                    line=dict(color=color, width=1),
                    name=f'Detection: {num_detection}',
                )
            )
            fig.add_trace(  # Отрисовка confidence модели на изображении.
                go.Scatter(
                    x=[text_x],
                    y=[text_y],
                    text=[f'{confidence:.2f}'],
                    mode='text',
                    textfont=dict(color='black', size=12),
                    name=f'{confidence:.2f}',
                )
            )
        fig.update_xaxes(visible=False, range=[0, width + offset])
        fig.update_yaxes(visible=False, range=[0, height])

        fig.update_layout(  # Отрисовка легенды с информацией о детекциях и confidence.
            legend=dict(
                x=0.01,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(116, 116, 116, 0.61)',
            ),
            width=width + offset,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.show()

    def _draw_crop_images(self, image_path: str | np.ndarray | Image.Image, prediction):
        """Отображает обрезанное изображение вокруг предсказанного объекта.

        Args:
            image_path (str | np.ndarray | Image.Image): - Путь к изображению.
            prediction (Detection): - Список предсказанных объектов.
        """
        if isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = Image.open(image_path)

        crop_image = crop_masked_rectangle(image, prediction.absolute_box)
        crop_width, crop_height = crop_image.size
        self.image_params.update(
            dict(
                zip(
                    ['source', 'x', 'y', 'sizex', 'sizey'],
                    [crop_image, 0, crop_height, crop_width, crop_height],
                )
            )
        )

        fig = go.Figure()
        fig.add_layout_image(**self.image_params)

        fig.update_xaxes(visible=False, range=[0, crop_width])
        fig.update_yaxes(visible=False, range=[0, crop_height])

        fig.update_layout(
            width=500,
            height=crop_height * (500 // crop_width),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.show()

    def visualize(self, index=None, crop_and_visualize=False):
        """
        Визуализирует изображение с рамками или обрезанными объектами.

        Параметры:
        :param index: int(optional) - Индекс изображения для визуализации.
        :param crop_and_visualize: bool(optional) - Отображать ли обрезанные объекты.
        """

        if index is not None:
            self.current_index = index
        else:
            index = self.current_index

        if index < 0 or index >= len(self.image_files):
            print('Индекс вне диапазона.')
            return

        image_name = self.image_files[index]

        if crop_and_visualize:
            count_crop_images = 0
            for pred in self.predictions[self.current_index]:
                if count_crop_images >= self.max_crop_images >= 0:
                    print(
                        f'Максимальное число выводимых картинок достигунто - {self.max_crop_images}'
                    )
                    break
                else:
                    self._draw_crop_images(image_name, pred)
                    count_crop_images += 1
        else:
            self._draw_image_with_bboxes(
                image_name, self.predictions[self.current_index]
            )

    def next_image(self):
        """Переходит к следующему изображению и визуализирует его."""
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.visualize()

    def previous_image(self):
        """Переходит к предыдущему изображению и визуализирует его."""
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.visualize()
