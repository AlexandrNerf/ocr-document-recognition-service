import ast
import os
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageFile

from data.data_classes import Prediction


def read_detections(
    images: List[np.ndarray] | List[Image.Image] | List[str] | str,
    txt_paths: str | List[str],
    box_format='pascal_voc',
    labels_from_file=True,
) -> Tuple[List[np.ndarray], List[List[Prediction]]]:
    """
    Чтение детекций из изображений

    Args:
        images (List[np.ndarray] | List[Image.Image] | str | List[str]): Список изображений или путь к папке с ними
        txt_paths (List[str] | str): Путь до детекций (формат txt) или список детекций при labels_from_file=False
        box_format (str): Формат ('yolo' или 'pascal_voc')
        labels_from_file (bool): Если True, то txt_paths должен быть путь до папки с txt файлами, иначе txt_paths должен быть список детекций
    """
    detections = []
    imgs = []

    if isinstance(images, str):
        images = [os.path.join(images, pth) for pth in sorted(os.listdir(images))]
    else:
        imgs = images

    if isinstance(txt_paths, str) and labels_from_file:
        txt_paths = [
            os.path.join(txt_paths, pth) for pth in sorted(os.listdir(txt_paths))
        ]

    for image, txt_path in zip(images, txt_paths):
        if isinstance(image, str):
            width, height = Image.open(image).size
        else:
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                width, height = image.size

        if labels_from_file:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        else:
            lines = txt_paths

        image_detections = []

        for line in lines:
            try:
                text, coords_str = line.split(" ", 1)
                if box_format == 'yolo':
                    x_cn, y_cn, w_n, h_n = list(map(float, coords_str.split(' ')))

                    tl_x, tl_y, br_x, br_y = (
                        (x_cn - w_n / 2) * width,
                        (y_cn - h_n / 2) * height,
                        (x_cn + w_n / 2) * width,
                        (y_cn + h_n / 2) * height,
                    )

                    absolute_box = [
                        (int(tl_x), int(tl_y)),
                        (int(br_x), int(tl_y)),
                        (int(br_x), int(br_y)),
                        (int(tl_x), int(br_y)),
                    ]

                else:
                    tl_x, tl_y, br_x, br_y = ast.literal_eval(coords_str)
                    absolute_box = [
                        (int(tl_x), int(tl_y)),
                        (int(br_x), int(tl_y)),
                        (int(br_x), int(br_y)),
                        (int(tl_x), int(br_y)),
                    ]

                relative_box = [
                    (tl_x / width, tl_y / height),
                    (br_x / width, tl_y / height),
                    (br_x / width, br_y / height),
                    (tl_x / width, br_y / height),
                ]

                image_detections.append(
                    Prediction(
                        absolute_box=absolute_box,
                        score=1.0,
                        relative_box=relative_box,
                        text=text,
                        text_score=1.0,
                    )
                )
            except ValueError as _:
                continue

        detections.append(image_detections)

    return imgs, detections


def crop_masked_rectangle(
    image: Union[np.ndarray, Image.Image, str],
    box: Union[List[tuple[int, int]], List[tuple[float, float]]],
) -> np.ndarray | Image.Image:
    """
    Вырезает прямоугольную область и вращает изображение (постпроцессинг).

    Args:
        image (np.ndarray | Image.Image | str): Входное изображение в формате BGR или PIL Image, размер (H, W, 3), или путь к файлу.
        box (list[tuple[int, int]] | list[tuple[float, float]]): Координаты четырёхугольника [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

    Returns:
        out (np.ndarray | Image.Image): Вырезанное изображение в формате BGR, размер (h, w, 3),
        где h и w — размеры ограничивающего прямоугольника.
    """

    if isinstance(image, Image.Image):
        img = np.asarray(image)
    elif isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    bbox = np.array(box, dtype=np.float32).reshape(4, 2)

    s = bbox.sum(axis=1)  # x + y
    diff = np.diff(box, axis=1)  # x - y

    # Определяем углы
    top_left = bbox[np.argmin(s)]
    bottom_right = bbox[np.argmax(s)]
    top_right = bbox[np.argmin(diff)]
    bottom_left = bbox[np.argmax(diff)]

    bbox = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    # Определяем размеры результирующего прямоугольника
    width = int(
        max(np.linalg.norm(bbox[0] - bbox[1]), np.linalg.norm(bbox[2] - bbox[3]))
    )
    height = int(
        max(np.linalg.norm(bbox[0] - bbox[3]), np.linalg.norm(bbox[1] - bbox[2]))
    )

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    ).reshape(4, 2)

    assert bbox.shape == (4, 2), f"Unexpected box shape: {bbox.shape}"
    assert dst.shape == (4, 2), f"Unexpected dst shape: {dst.shape}"

    perspective = cv2.getPerspectiveTransform(bbox, dst)
    warped = cv2.warpPerspective(img, perspective, (width, height))

    return warped if type(image) == np.ndarray else Image.fromarray(warped)
