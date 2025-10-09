import argparse
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm


def convert_numpy_types(obj):
    """Рекурсивно преобразует numpy типы в стандартные python типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj


def process_box(box_data, image, item_num, box_num, images_dir, pbar_box=None):
    try:
        if len(box_data) == 2 and isinstance(box_data[1], (list, tuple)):
            polygon, (text, confidence) = box_data
        else:
            if pbar_box:
                pbar_box.update(1)
            return None

        polygon_np = np.array(polygon, dtype=np.int32)
        x_min, y_min = polygon_np.min(axis=0)
        x_max, y_max = polygon_np.max(axis=0)

        word_image = image.crop((x_min, y_min, x_max, y_max))

        img_filename = f"invoices-and-receipts_ocr_v1_word_{item_num}_{box_num}.png"
        img_path = os.path.join(images_dir, img_filename)
        word_image.save(img_path)

        if pbar_box:
            pbar_box.update(1)

        return (
            img_filename,
            {"dimensions": [int(y_max - y_min), int(x_max - x_min)], "text": text},
        )

    except Exception as e:
        print(f"Ошибка в элементе {item_num}, боксе {box_num}: {e}")
        if pbar_box:
            pbar_box.update(1)
        return None


def process_item(item, item_num, images_dir, pbar_items=None):
    try:
        image = item['image']
        raw_data = json.loads(item['raw_data'])
        ocr_boxes = ast.literal_eval(raw_data['ocr_boxes'])

        results = []
        with tqdm(
            total=len(ocr_boxes),
            desc=f"Обработка боксов элемента {item_num}",
            leave=False,
        ) as pbar_box:
            for box_num, box_data in enumerate(ocr_boxes):
                result = process_box(
                    box_data, image, item_num, box_num, images_dir, pbar_box
                )
                if result:
                    results.append(result)

        if pbar_items:
            pbar_items.update(1)

        return results

    except Exception as e:
        print(f"Ошибка при обработке элемента {item_num}: {e}")
        if pbar_items:
            pbar_items.update(1)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', default=r'E:\ocr_datasets\eng\invoices-and-receipts_ocr_v1'
    )
    parser.add_argument(
        '--images_dir',
        default=r'E:\ocr_datasets\eng\invoices-and-receipts_ocr_v1\images',
    )
    parser.add_argument(
        '--output_json',
        default=r'E:\ocr_datasets\eng\invoices-and-receipts_ocr_v1\annotations.json',
    )
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)
    annotations = {}

    print("Загрузка датасета...")
    ds = load_dataset(
        "mychen76/invoices-and-receipts_ocr_v1", cache_dir=args.dataset_path
    )
    total_items = len(ds['train'])

    print(f"Обработка {total_items} элементов...")
    with tqdm(total=total_items, desc="Общий прогресс") as pbar_main:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for i, item in enumerate(ds['train']):
                future = executor.submit(
                    process_item, item, i, args.images_dir, pbar_main
                )
                futures.append(future)

            for future in as_completed(futures):
                item_results = future.result()
                if item_results:
                    for img_filename, annotation in item_results:
                        annotations[img_filename] = convert_numpy_types(annotation)

    print("Сохранение результатов...")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(annotations), f, ensure_ascii=False, indent=2)

    print("\nПримеры сохраненных данных:")
    for k, v in list(annotations.items())[:5]:
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
