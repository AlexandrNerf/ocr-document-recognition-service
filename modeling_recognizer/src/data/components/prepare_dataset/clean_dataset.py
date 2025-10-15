import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm


def clean_labels_and_images(label_path, new_label_path, main_images_path):
    """
    Удаляет элементы из JSON-файла и связанные изображения на основе условий.

    :param label_path: Путь к исходному JSON-файлу с метками.
    :param new_label_path: Путь для сохранения обновленного JSON-файла.
    :param main_images_path: Путь к папке с изображениями.
    """

    CUSTOM_VOCAB = 'АӘБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФҺЦЧШЩЪЫІЬЭЮЯаәбвгғдеёжзийкқлмнңоөпрстуұүфһцчшщъыіьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!$#()?-.,:;@%&*+=[]{}'

    # Загружаем JSON-файл
    with open(label_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    all_files = list(labels_data.keys())
    print('Полное количетсво данных', len(all_files))

    # Список для хранения ключей элементов, которые нужно удалить
    keys_to_delete = []

    # Итерируемся по данным
    for filename, label_info in tqdm(labels_data.items(), desc="Processing files"):
        text = label_info.get('text', '')
        dim = label_info.get('dimensions')
        invalid_chars = {char for char in text if char not in CUSTOM_VOCAB}
        # Проверяем условия для удаления
        if (
            len(text) == 1  # Удаляются элементы с одинночными символами
            or dim[0] > dim[1] * 2  #  Удаляются элементы у которых высота больше ширины в 2 раза
            or not (
                10 < dim[0] < 50 and 10 < dim[1] < 200
            )  # Удаляются элементы у который высота не (10, 50) и шириина не (10, 200)
            or invalid_chars  # Удаляются элементы, которых нет в vocab
        ):
            image_path = os.path.join(main_images_path, filename)
            print(f'Элемент помечен для удаления: {filename}')
            print(f'Путь к изображению: {image_path}')

            # Добавляем ключ в список для удаления
            keys_to_delete.append(filename)

            # Удаляем изображение, если оно существует
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f'Изображение удалено: {image_path}')

    # Удаляем помеченные элементы из JSON
    for key in keys_to_delete:
        del labels_data[key]

    # Сохраняем обновленный JSON-файл
    with open(new_label_path, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=4)

    print(f'Обновленный JSON-файл сохранен: {new_label_path}')


if __name__ == '__main__':
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label-path',
        type=Path,
        default='/home/ovsyannikoviv/datasets/clean/all_in_one/val.json',
        help='Путь к входным меткам JSON',
    )
    parser.add_argument(
        '--new-label-path',
        type=Path,
        default='/home/ovsyannikoviv/datasets/clean/all_in_one/val.json',
        help='Путь для сохранения очищенных меток JSON',
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        default='/home/ovsyannikoviv/datasets/clean/all_in_one',
        help='Путь к каталогу изображений',
    )
    args = parser.parse_args()

    # Вызываем основную функцию
    clean_labels_and_images(
        label_path=args.label_path,
        new_label_path=args.new_label_path,
        main_images_path=args.images_dir,
    )
