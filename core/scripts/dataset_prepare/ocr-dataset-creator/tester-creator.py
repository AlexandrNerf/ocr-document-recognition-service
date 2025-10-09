import os
import random
from concurrent.futures import ThreadPoolExecutor
from shutil import copyfile

from tqdm import tqdm

# Настройки (можно менять)
SETTINGS = {
    'output_dir': r'E:\ocr_datasets\ddi_for_tester\result_server',  # Выходная директория
    'count_per_lang': 100,  # Количество образцов на язык
    'workers': 8,  # Количество потоков
    # Пути к директориям с данными (можно указать несколько для каждого языка)
    'languages': {
        'ru': {
            'images': [
                r'E:\ocr_datasets\ddi_for_tester\ru\1\images',
                r'E:\ocr_datasets\ddi_for_tester\ru\2\images',
            ],
            'data': [
                r'E:\ocr_datasets\ddi_for_tester\ru\1\data',
                r'E:\ocr_datasets\ddi_for_tester\ru\2\data',
            ],
        },
        'kz': {
            'images': [r'E:\ocr_datasets\ddi_for_tester\kz\images'],
            'data': [r'E:\ocr_datasets\ddi_for_tester\kz\data'],
        },
        'eng': {
            'images': [
                r'E:\ocr_datasets\ddi_for_tester\eng\1\images',
                r'E:\ocr_datasets\ddi_for_tester\eng\2\images',
            ],
            'data': [
                r'E:\ocr_datasets\ddi_for_tester\eng\1\data',
                r'E:\ocr_datasets\ddi_for_tester\eng\2\data',
            ],
        },
    },
}


def process_file(args):
    """Обрабатывает один файл: копирует и переименовывает изображение и метку"""
    (
        file_base,
        input_images_dir,
        input_labels_dir,
        output_images_dir,
        output_labels_dir,
        lang_code,
        new_idx,
    ) = args

    # Ищем оригинальное расширение файла изображения
    image_ext = None
    for f in os.listdir(input_images_dir):
        if f.startswith(file_base + '.'):
            image_ext = f.split('.')[-1]
            break

    if not image_ext:
        return False

    # Новое имя файла
    new_name = f"test_{lang_code}_{new_idx}"

    # Копируем изображение
    src_image = os.path.join(input_images_dir, f"{file_base}.{image_ext}")
    dst_image = os.path.join(output_images_dir, f"{new_name}.{image_ext}")
    copyfile(src_image, dst_image)

    # Копируем метку (предполагаем расширение .txt)
    src_label = os.path.join(input_labels_dir, f"{file_base}.txt")
    dst_label = os.path.join(output_labels_dir, f"{new_name}.txt")
    if os.path.exists(src_label):
        copyfile(src_label, dst_label)

    return True


def process_language(lang_data, output_dir, lang_code, max_count, start_idx):
    """Обрабатывает данные для одного языка"""
    input_images_dir = lang_data['images']
    input_labels_dir = lang_data['data']
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'data')

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Получаем список файлов (без расширения)
    image_files = [
        f.split('.')[0]
        for f in os.listdir(input_images_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]

    random.shuffle(image_files)

    if len(image_files) < max_count:
        print(
            f"Warning: Only {len(image_files)} files available for language {lang_code} (requested {max_count})"
        )
        # Берем все доступные файлы, если запрошено больше чем есть
        selected_files = image_files.copy()
    else:
        selected_files = image_files[:max_count]

    # Подготавливаем аргументы для каждого файла
    tasks = [
        (
            file_base,
            input_images_dir,
            input_labels_dir,
            output_images_dir,
            output_labels_dir,
            lang_code,
            i + start_idx,
        )
        for i, file_base in enumerate(selected_files)
    ]

    # Обрабатываем файлы с отображением прогресса
    success_count = 0
    with ThreadPoolExecutor(max_workers=SETTINGS['workers']) as executor:
        results = list(
            tqdm(
                executor.map(process_file, tasks),
                total=len(tasks),
                desc=f'Processing {lang_code}',
                leave=False,
            )
        )
        success_count = sum(results)

    return success_count


def prepare_temp_dir(lang, dirs, output_dir):
    """Создает временную директорию с объединенными файлами для языка"""
    temp_dir = os.path.join(output_dir, f'temp_{lang}')
    temp_images_dir = os.path.join(temp_dir, 'images')
    temp_data_dir = os.path.join(temp_dir, 'data')

    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_data_dir, exist_ok=True)

    # Объединяем все изображения из всех директорий
    for img_dir in dirs['images']:
        if not os.path.exists(img_dir):
            continue

        # Создаем префикс на основе имени родительской папки
        dir_prefix = os.path.basename(os.path.dirname(img_dir))

        for f in os.listdir(img_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                file_base, ext = os.path.splitext(f)
                # Добавляем префикс к имени файла
                new_name = f"{dir_prefix}_{file_base}{ext}"
                src = os.path.join(img_dir, f)
                dst = os.path.join(temp_images_dir, new_name)
                if not os.path.exists(dst):
                    copyfile(src, dst)

                # Также копируем соответствующий файл меток
                label_src = os.path.join(
                    img_dir.replace('images', 'data'), f"{file_base}.txt"
                )
                if os.path.exists(label_src):
                    label_dst = os.path.join(
                        temp_data_dir, f"{dir_prefix}_{file_base}.txt"
                    )
                    if not os.path.exists(label_dst):
                        copyfile(label_src, label_dst)

    return {'images': temp_images_dir, 'data': temp_data_dir}


def main():
    # Создаем выходную директорию
    os.makedirs(SETTINGS['output_dir'], exist_ok=True)

    # Подготавливаем временные директории для каждого языка
    lang_data = {}
    with tqdm(SETTINGS['languages'].items(), desc='Preparing temp dirs') as pbar:
        for lang, dirs in pbar:
            pbar.set_postfix({'lang': lang})
            lang_data[lang] = prepare_temp_dir(lang, dirs, SETTINGS['output_dir'])

    # Обрабатываем каждый язык
    total_files = 0
    start_idx = 1

    with tqdm(SETTINGS['languages'].items(), desc='Processing languages') as pbar:
        for lang, data in pbar:
            pbar.set_postfix({'lang': lang})
            processed = process_language(
                lang_data=lang_data[lang],
                output_dir=SETTINGS['output_dir'],
                lang_code=lang,
                max_count=SETTINGS['count_per_lang'],
                start_idx=start_idx,
            )
            total_files += processed
            start_idx += SETTINGS['count_per_lang']

    # Удаляем временные директории
    for lang in SETTINGS['languages'].keys():
        temp_dir = os.path.join(SETTINGS['output_dir'], f'temp_{lang}')
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)

    print(
        f"\nDataset created successfully with {total_files} files in {SETTINGS['output_dir']}"
    )


if __name__ == '__main__':
    main()
