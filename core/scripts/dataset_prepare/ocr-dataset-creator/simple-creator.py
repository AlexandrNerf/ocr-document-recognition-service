import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm.auto import tqdm


def convert_coordinates_yolo_to_pixels(yolo_coords, img_width, img_height):
    """Преобразование координат YOLO (нормализованные) в пиксельные координаты"""
    x_center, y_center, width, height = yolo_coords
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = int(x_center - width / 2)
    x_max = int(x_center + width / 2)
    y_min = int(y_center - height / 2)
    y_max = int(y_center + height / 2)

    return x_min, y_min, x_max, y_max


def read_label_file(label_path):
    """Чтение файла метки с автоматическим определением кодировки (сначала UTF-16, затем UTF-8)"""
    encodings = ['utf-16', 'utf-8']  # Порядок важен - сначала пробуем UTF-16

    for encoding in encodings:
        try:
            with open(label_path, 'r', encoding=encoding) as f:
                return f.readlines()
        except Exception as _:
            continue

    # Если ни одна кодировка не подошла
    print(f"Не удалось прочитать файл {label_path} в кодировках utf-16 или utf-8")
    return []


def process_image(image_path, label_path, output_images_dir):
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
        annotations = []

        if not os.path.exists(label_path):
            return []

        lines = read_label_file(label_path)  # Используем новую функцию чтения

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            text = parts[0]
            yolo_coords = list(map(float, parts[1:5]))

            x_min, y_min, x_max, y_max = convert_coordinates_yolo_to_pixels(
                yolo_coords, img_width, img_height
            )

            word_image = image.crop((x_min, y_min, x_max, y_max))

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            img_filename = f"DDI_{base_name}_word_{i}.png"
            img_path = os.path.join(output_images_dir, img_filename)
            word_image.save(img_path)

            annotations.append(
                (
                    img_filename,
                    {"dimensions": [y_max - y_min, x_max - x_min], "text": text},
                )
            )

        return annotations

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', default=r'E:\ocr_datasets\ddi_for_tester\result_server'
    )
    parser.add_argument(
        '--output_json',
        default=r'E:\ocr_datasets\ddi_for_tester\result_server\annotations.json',
    )
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset_dir, 'images')
    labels_dir = os.path.join(args.dataset_dir, 'data')
    output_images_dir = os.path.join(args.dataset_dir, 'word_images')

    os.makedirs(output_images_dir, exist_ok=True)
    annotations = {}

    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    total_images = len(image_files)

    print(f"Найдено {total_images} изображений для обработки...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            image_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            futures.append(
                executor.submit(
                    process_image, image_path, label_path, output_images_dir
                )
            )

        for future in tqdm(
            as_completed(futures), total=total_images, desc="Обработка изображений"
        ):
            image_annotations = future.result()
            for img_filename, annotation in image_annotations:
                annotations[img_filename] = annotation

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено {len(annotations)} аннотаций в {args.output_json}")
    print("Примеры сохраненных данных:")
    for k, v in list(annotations.items())[:5]:
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
