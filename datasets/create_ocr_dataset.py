import json
import os
import pickle
from glob import glob
from typing import Union

import cv2
import numpy as np
import pandas as pd
import pdf2image
import tqdm
from PIL import Image


def crop_masked_rectangle(
    image: Union[np.ndarray, Image.Image, str],
    box: Union[list[tuple[int, int]], list[tuple[float, float]]],
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

    return warped


DATA_PATH = "ocr_dataset"


def DDI():
    directory = "DDI_dataset"

    if not os.path.exists("DDI_dataset"):
        return None

    book_paths = glob(os.path.join(directory, "*"))
    paths = []
    for book in book_paths:
        paths += glob(os.path.join(book, "gen_imgs", "*"))

    result_image = os.path.join(DATA_PATH, "images")
    os.makedirs(result_image + "/DDI/", exist_ok=True)

    images_row, text_row = [], []

    for i, path in enumerate(tqdm.tqdm(paths, desc="Collecting dataset images")):
        if i > 400:
            break
        img_path = path
        image = cv2.imread(img_path)

        boxes_path = os.path.join(
            os.path.dirname(os.path.dirname(img_path)),
            "gen_boxes",
            os.path.splitext(os.path.basename(img_path))[0] + ".pickle",
        )

        with open(boxes_path, "rb") as f:
            data = pickle.load(f)
        for ind, info in enumerate(data):
            box, text = info["box"], info["text"]
            new_img_path = os.path.join(result_image, f"DDI/{i}_{ind}.png")
            box = [
                (box[0][1], box[0][0]),
                (box[1][1], box[1][0]),
                (box[2][1], box[2][0]),
                (box[3][1], box[3][0]),
            ]
            images_row.append(new_img_path)
            text_row.append(text)
            croped = crop_masked_rectangle(image, box)
            cv2.imwrite(new_img_path, croped)

    dataframe = pd.DataFrame({"image_path": images_row, "text": text_row})
    dataframe.to_parquet("ocr_ddi.parquet")


def PDFA():
    def sectorize_pdf(dir):
        files = os.listdir(dir)
        return [f for f in files if f.endswith(".pdf")]

    directory = "PDFA_dataset"
    if not os.path.exists("PDFA_dataset"):
        return None

    pdf_files = sectorize_pdf(directory)
    result_image = os.path.join(DATA_PATH, "images")

    os.makedirs(result_image + "/PDFA/", exist_ok=True)
    images_row, text_row = [], []

    for i, pdf_name in enumerate(tqdm.tqdm(pdf_files, desc="Processing PDFA files")):
        if i > 50:
            break
        json_name = f"{os.path.splitext(pdf_name)[0]}.json"
        pdf_path = os.path.join(directory, pdf_name)
        json_path = os.path.join(directory, json_name)

        with open(json_path, "r", encoding="utf-8") as j:
            data = json.load(j)

        images = pdf2image.convert_from_path(
            pdf_path, poppler_path=r"C:\poppler\Library\bin"
        )
        for ind, (img, page) in enumerate(
            tqdm.tqdm(zip(images, data["pages"]), desc=f"Processing {i} page")
        ):
            for sec_ind, (text, box) in enumerate(
                zip(page["words"]["text"], page["words"]["bbox"])
            ):
                box = [
                    (box[0] * img.width, box[1] * img.height),
                    ((box[0] + box[2]) * img.width, box[1] * img.height),
                    ((box[0] + box[2]) * img.width, (box[1] + box[3]) * img.height),
                    ((box[0]) * img.width, (box[1] + box[3]) * img.height),
                ]
                img_path = os.path.join(result_image, f"PDFA/{i}_{ind}_{sec_ind}.png")

                croped = crop_masked_rectangle(img, box)
                cv2.imwrite(img_path, croped)
                images_row.append(img_path)
                text_row.append(text)

    dataframe = pd.DataFrame({"image_path": images_row, "text": text_row})
    dataframe.to_parquet("ocr_pdfa.parquet")


def KOTHD():
    directory = "KOTHD_dataset"
    if not os.path.exists("KOTHD_dataset"):
        return None

    result_image = os.path.join(DATA_PATH, "images")

    os.makedirs(result_image + "/KOTHD/", exist_ok=True)
    images_row, text_row = [], []

    annotations_path = os.path.join(directory, "ann")
    images_path = os.path.join(directory, "img")

    annotations = os.listdir(annotations_path)

    for i, ann in enumerate(tqdm.tqdm(annotations, desc="Processing KOTHD")):
        try:
            with open(
                os.path.join(annotations_path, ann), encoding="utf-8", mode="r"
            ) as j:
                data = json.load(j)
            image = cv2.imread(images_path + "/" + data["name"])
            text = data["description"]

            new_image_path = os.path.join(result_image, f"KOTHD/{i}.png")
            cv2.imwrite(new_image_path, image)
            images_row.append(new_image_path)
            text_row.append(text)
        except Exception as e:
            continue

    dataframe = pd.DataFrame({"image_path": images_row, "text": text_row})
    dataframe.to_parquet("ocr_kothd.parquet")


def SynthRu():
    directory = "SynthRu_dataset"
    if not os.path.exists("SynthRu_dataset"):
        return None

    result_image = os.path.join(DATA_PATH, "images")

    os.makedirs(result_image + "/SynthRu/", exist_ok=True)
    images_row, text_row = [], []

    with open(os.path.join(directory, "gt.txt"), "r", encoding="utf-8") as txt:
        for i, line in enumerate(tqdm.tqdm(txt, desc="SynthRu processing")):
            res = line.split(",")
            img_path, text = res[0], res[1]

            image = cv2.imread(os.path.join(directory, img_path))
            new_img_path = os.path.join(result_image, f"SynthRu/{i}.png")

            cv2.imwrite(new_img_path, image)
            images_row.append(new_img_path)
            text_row.append(text[:-1])

    dataframe = pd.DataFrame({"image_path": images_row, "text": text_row})
    dataframe.to_parquet("ocr_synthru.parquet")


def WikiKZ():
    import ast
    import random

    directory = "WikiKZ_dataset"
    if not os.path.exists("WikiKZ_dataset"):
        return None

    result_image = os.path.join(DATA_PATH, "images")

    os.makedirs(result_image + "/WikiKZ/", exist_ok=True)
    images_row, text_row = [], []
    full_directory_data = os.path.join(directory, "parsing_pdf/result/data")
    full_directory_images = os.path.join(directory, "parsing_pdf/result/images")

    datas = os.listdir(full_directory_data)

    for i, data in enumerate(tqdm.tqdm(datas, desc="WikiKZ processing")):
        if i > 700:
            break
        image = cv2.imread(os.path.join(full_directory_images, data[:-3] + "png"))
        with open(
            os.path.join(full_directory_data, data), "r", encoding="utf-8"
        ) as txt:
            for j, line in enumerate(txt):
                try:
                    idx = line.find("[")
                    text = line[:idx].strip()
                    box_str = line[idx:].strip()
                    box_list = ast.literal_eval(box_str)
                    w = box_list[2] - box_list[0]
                    h = box_list[3] - box_list[1]
                    box = [
                        (
                            box_list[0] + w * random.uniform(-0.05, 0.05),
                            box_list[1] + h * random.uniform(-0.05, 0.05),
                        ),
                        (
                            box_list[2] + w * random.uniform(-0.05, 0.05),
                            box_list[1] + h * random.uniform(-0.05, 0.05),
                        ),
                        (
                            box_list[2] + w * random.uniform(-0.05, 0.05),
                            box_list[3] + h * random.uniform(-0.05, 0.05),
                        ),
                        (
                            box_list[0] + w * random.uniform(-0.05, 0.05),
                            box_list[3] + h * random.uniform(-0.05, 0.05),
                        ),
                    ]
                    croped = crop_masked_rectangle(image, box)
                    new_img_path = os.path.join(result_image, f"WikiKZ/{i}_{j}.png")
                    cv2.imwrite(new_img_path, croped)
                    images_row.append(new_img_path)
                    text_row.append(text)
                except Exception as e:
                    print(e)
    dataframe = pd.DataFrame({"image_path": images_row, "text": text_row})
    dataframe.to_parquet("ocr_wikikz.parquet")


def GNHK():
    directory = "GNHK_dataset/train"
    if not os.path.exists("GNHK_dataset"):
        return None

    result_image = os.path.join(DATA_PATH, "images")

    os.makedirs(result_image + "/GNHK/", exist_ok=True)
    images_row, text_row = [], []

    for i, file in enumerate(tqdm.tqdm(os.listdir(directory), desc="GNHK processing")):
        if not file.endswith(".json"):
            continue
        image = cv2.imread(os.path.join(directory, file[:-4] + "jpg"))
        with open(os.path.join(directory, file), "r", encoding="utf-8") as j:
            data = json.load(j)
        for j, d in enumerate(data):
            if d["text"] == "%math%":
                continue
            text = d["text"]
            poly = d["polygon"]
            box = [
                (poly["x0"], poly["y0"]),
                (poly["x1"], poly["y1"]),
                (poly["x2"], poly["y2"]),
                (poly["x3"], poly["y3"]),
            ]
            try:
                croped = crop_masked_rectangle(image, box)
                new_img_path = os.path.join(result_image, f"GNHK/{i}_{j}.png")
                cv2.imwrite(new_img_path, croped)
                images_row.append(new_img_path)
                text_row.append(text)
            except Exception as e:
                print(e)
    dataframe = pd.DataFrame({"image_path": images_row, "text": text_row})
    dataframe.to_parquet("ocr_gnhk.parquet")

def HWCYR():
    directory = "HWCYR_dataset/train"
    if not os.path.exists(directory):
        return None
    dataset = pd.read_csv("HWCYR_dataset/train.tsv",sep='\t')

    images_row, text_row = [], []
    result_image = os.path.join(DATA_PATH, "images")
    for i, (image_dir, text) in tqdm.tqdm(dataset.iterrows(), desc='Processing HWCYR'):
        image = cv2.imread(os.path.join(directory, image_dir))
        new_img_path = os.path.join(result_image, f'HWCYR/{i}.png')
        cv2.imwrite(new_img_path, image)
        images_row.append(new_img_path)
        text_row.append(text)
    dataframe = pd.DataFrame({"image_path": images_row, "text": text_row, 'source': ['HWCYR' for _ in range(len(images_row))]})
    dataframe.to_parquet("ocr_hwcyr.parquet")
    



def create_ocr(base_datasets: list[str]):
    for dataset_fn in base_datasets:
        dataset_fn()


if __name__ == "__main__":
    base_datasets = [HWCYR]  # Вставить сюда нужные датасеты
    create_ocr(base_datasets=base_datasets)
