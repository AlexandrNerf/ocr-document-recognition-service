from typing import List

import cv2
import numpy as np

from data.data_classes import Prediction
from pipelines.default.postprocessor import Postprocessor


import numpy as np
from functools import cmp_to_key

# polygons = np.array([box.absolute_box for box in detections])[:, :4]

        # centers = polygons.mean(axis=1)
        # y_centers = centers[:, 1]
        # x_centers = centers[:, 0]

        # initial_order = np.argsort(y_centers)
        # y_centers = y_centers[initial_order]
        # x_centers = x_centers[initial_order]

        # lines = []
        # current_line = [0]
        # for i in range(1, len(polygons)):
        #     if abs(y_centers[i] - y_centers[current_line[0]]) < y_thresh:
        #         current_line.append(i)
        #     else:
        #         lines.append(current_line)
        #         current_line = [i]
        # lines.append(current_line)

        # final_indices = []
        # for line in lines:
        #     sorted_line = sorted(line, key=lambda idx: x_centers[idx])
        #     final_indices.extend(initial_order[sorted_line])

        # indices = np.array(final_indices)

        # return [detections[i] for i in indices]

class SimplePostprocessor(Postprocessor):
    """
    Класс простого постпроцессора.

    Функционал: `сортировка боксов`, `создание кропов`.
    """

    def rotate_polygon_boxes(self, polygons: np.array) -> list[tuple[int, np.array]]:
        """
        Вращает боксы относительно центра, приводя их к прямому виду
        """
        def angle_top_edge(poly: np.array):
            """
            Возвращает угол (радианы), на который нужно повернуть polygon,
            чтобы сторона top-left → top-right была горизонтальной
            """
            # Найдем top-left
            top_left, top_right = poly[0], poly[1]

            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]
            angle = np.arctan2(dy, dx)
            return angle
        
        def rotate_points_image_coords(
            points: np.array, 
            angle: float, 
            center: tuple[float, float] = (0,0)
        ):
            """
            Поворот точек на angle (радианы) для координат изображения (Y вниз)
            """
            cos_angle, sin_angle = np.cos(angle), np.sin(angle)
            cx, cy = center
            points = np.array(points)
            x, y = points[:,0], points[:,1]

            x_new = cos_angle*(x-cx) + sin_angle*(y-cy) + cx
            y_new = -sin_angle*(x-cx) + cos_angle*(y-cy) + cy
            
            return np.stack([x_new, y_new], axis=1)
        
        angles = [angle_top_edge(poly) for poly in polygons]
        mean_angle = np.mean(angles)

        # 2. центр документа для вращения
        
        doc_center = np.mean(np.vstack(polygons), axis=0)
        straighten_polygons = [
            (i, rotate_points_image_coords(poly, mean_angle, doc_center)) for i, poly in enumerate(polygons)
        ]
        return straighten_polygons

        
    def sort_polygon_box_indices(
        self, detections: List[Prediction], y_thresh=0.6
    ) -> List[Prediction]:
        """
        Сортировка индексов боксов

        Args:
            detections (List[Detection]): (N, 4, 2) — Список детекций
            y_thresh (float): порог для объединения боксов в одну строку 
                                (по умолчанию равен 60% вхождения по высоте)

        Returns:
            Отсортированные оригинальные полигоны
        """
        polygons = np.array([det.absolute_box for det in detections])
        straighten_polygons = self.rotate_polygon_boxes(polygons)
        
        heights = np.array([self.poly_height(p) for _, p in straighten_polygons])
        median_height = np.median(heights)
        threshold = y_thresh * median_height
            
        straighten_polygons = sorted(straighten_polygons, key=lambda p: self.poly_center(p[1])[0])

        length = len(straighten_polygons)-1
        swapped = True
        while swapped:
            swapped = False
            for i in range(length):
                if self.sort_two_boxes(straighten_polygons[i], straighten_polygons[i+1], threshold):
                    straighten_polygons[i], straighten_polygons[i+1] = straighten_polygons[i+1], straighten_polygons[i]
                    swapped = True
        
        return [detections[i] for i, _ in straighten_polygons]
    
    def postprocessing(
        self, detections: List[Prediction], image: np.ndarray
    ) -> List[Prediction]:
        """
        Создание кропов по детекциям и картинкам
        Сортирует координаты бокса, сортирует сами боксы.

        Args:
            detections (List[Detection]): Детекции
            image (np.ndarray): Изображение (исходник)

        Returns:
            out (List[Detection]): Детекции с кропами с правильным порядком вершин и самих боксов
        """

        for box in detections:
            box.absolute_box = self.sort_vertices_order(box.absolute_box)
            bbox = np.array(box.absolute_box)

            # Определяем размеры результирующего прямоугольника
            width = int(
                max(
                    np.linalg.norm(bbox[0] - bbox[1]), np.linalg.norm(bbox[2] - bbox[3])
                )
            )
            height = int(
                max(
                    np.linalg.norm(bbox[0] - bbox[3]), np.linalg.norm(bbox[1] - bbox[2])
                )
            )

            dst = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            ).reshape(4, 2)

            #assert bbox.shape == (4, 2), f"Unexpected box shape: {bbox.shape}"
            assert dst.shape == (4, 2), f"Unexpected dst shape: {dst.shape}"

            warped = cv2.warpPerspective(
                image, cv2.getPerspectiveTransform(bbox, dst), (width, height)
            )
            box.crop = warped
        return self.sort_polygon_box_indices(detections)

    def sort_vertices_order(self, box) -> list[tuple[int, int]]:
        bbox = np.array(box, dtype=np.float32).reshape(4, 2)

        s = bbox.sum(axis=1)  # x + y
        diff = np.diff(bbox, axis=1)  # x - y

        # Определяем углы
        top_left = bbox[np.argmin(s)]
        bottom_right = bbox[np.argmax(s)]
        top_right = bbox[np.argmin(diff)]
        bottom_left = bbox[np.argmax(diff)]

        bbox = [
            tuple(top_left), 
            tuple(top_right), 
            tuple(bottom_right), 
            tuple(bottom_left),
        ]
        
        return bbox
    
    def poly_center(self, poly):
        return np.mean(poly, axis=0)

    def poly_height(self, poly):
        y_coords = poly[:,1]
        return y_coords.max() - y_coords.min()

    def sort_two_boxes(self, first_poly, second_poly, threshold):
            _, first_poly = first_poly
            _, second_poly = second_poly

            center_ax, center_ay = self.poly_center(first_poly)
            center_bx, center_by = self.poly_center(second_poly)

            if abs(center_ay - center_by) > threshold:
                return center_ay > center_by
            return center_ax > center_bx