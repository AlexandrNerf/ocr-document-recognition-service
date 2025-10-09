import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def load_annotations(annotation_path: str) -> dict[str, dict]:
    with tqdm(desc=f"Loading annotations {Path(annotation_path).name}", leave=False):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def collect_dataset_files(
    data_dirs: list[str], annotation_paths: list[str], lang: str
) -> list[tuple[str, dict]]:
    dataset = []
    for data_dir, ann_path in tqdm(
        zip(data_dirs, annotation_paths), desc=f"Collecting {lang} files", leave=False
    ):
        annotations = load_annotations(ann_path)
        for img_name, annotation in tqdm(
            annotations.items(), desc=f"Processing {Path(data_dir).name}", leave=False
        ):
            img_path = os.path.join(data_dir, img_name)
            if os.path.exists(img_path):
                dataset.append((img_path, annotation, lang))
    return dataset


def balance_datasets(
    datasets: dict[str, list[tuple[str, dict]]], samples_per_lang: int, val_split: float
) -> tuple[list[tuple[str, dict, str]], list[tuple[str, dict, str]]]:
    train_data = []
    val_data = []

    for lang, data in tqdm(datasets.items(), desc="Balancing datasets"):
        if len(data) > samples_per_lang:
            sampled_data = random.sample(data, samples_per_lang)
        else:
            sampled_data = data

        # Split into train and val
        split_idx = int(len(sampled_data) * (1 - val_split))
        train_data.extend(sampled_data[:split_idx])
        val_data.extend(sampled_data[split_idx:])

    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data


def copy_file(src_dst: tuple[str, str], pbar: tqdm = None) -> bool:
    src, dst = src_dst
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            with open(src, 'rb') as f_src, open(dst, 'wb') as f_dst:
                f_dst.write(f_src.read())
        if pbar:
            pbar.update(1)
        return True
    except Exception:
        if pbar:
            pbar.update(1)
        return False


def get_folder_size(folder: str) -> int:
    total_size = 0
    for dirpath, _, filenames in tqdm(os.walk(folder), desc="Calculating folder size"):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def create_balanced_dataset(
    output_dir: str,
    datasets_ru: list[str],
    datasets_kz: list[str],
    datasets_eng: list[str],
    annotations_ru: list[str],
    annotations_kz: list[str],
    annotations_eng: list[str],
    samples_per_lang: int,
    val_split: float,
    workers: int = 4,
) -> None:
    data_dirs = {
        'ru': (datasets_ru, annotations_ru),
        'kz': (datasets_kz, annotations_kz),
        'eng': (datasets_eng, annotations_eng),
    }

    all_data = {}
    for lang, (dirs, anns) in tqdm(data_dirs.items(), desc="Processing languages"):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(collect_dataset_files, dirs, anns, lang)
                for _ in range(len(dirs))
            ]
            lang_data = []
            for future in tqdm(
                as_completed(futures), desc=f"Collecting {lang} data", leave=False
            ):
                lang_data.extend(future.result())
        all_data[lang] = lang_data

    train_data, val_data = balance_datasets(all_data, samples_per_lang, val_split)

    # Create output directories
    train_data_dir = os.path.join(output_dir, 'train_images')
    val_data_dir = os.path.join(output_dir, 'val_images')
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)

    def process_data(
        data: list[tuple[str, dict, str]], output_dir: str, prefix: str
    ) -> tuple[dict, dict]:
        copy_tasks = []
        annotations = {}
        lang_counts = {'ru': 0, 'kz': 0, 'eng': 0}

        for i, (src_path, annotation, lang) in tqdm(
            enumerate(data), desc=f"Preparing {prefix} copy tasks"
        ):
            ext = Path(src_path).suffix
            dst_path = os.path.join(output_dir, f"{lang}_{prefix}_{i}{ext}")
            copy_tasks.append((src_path, dst_path))
            annotations[f"{lang}_{prefix}_{i}{ext}"] = annotation
            lang_counts[lang] += 1

        with tqdm(
            total=len(copy_tasks), desc=f"Copying {prefix} files", unit="file"
        ) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(copy_file, task, pbar) for task in copy_tasks
                ]
                for _ in tqdm(
                    as_completed(futures), desc="Processing files", leave=False
                ):
                    pass

        return annotations, lang_counts

    # Process train data
    train_annotations, train_counts = process_data(train_data, train_data_dir, 'train')
    train_ann_path = os.path.join(output_dir, 'train.json')
    with tqdm(desc="Saving train annotations"):
        with open(train_ann_path, 'w', encoding='utf-8') as f:
            json.dump(train_annotations, f, ensure_ascii=False, indent=2)

    # Process validation data
    val_annotations, val_counts = process_data(val_data, val_data_dir, 'val')
    val_ann_path = os.path.join(output_dir, 'val.json')
    with tqdm(desc="Saving validation annotations"):
        with open(val_ann_path, 'w', encoding='utf-8') as f:
            json.dump(val_annotations, f, ensure_ascii=False, indent=2)

    train_size = get_folder_size(train_data_dir)
    val_size = get_folder_size(val_data_dir)

    print("\nDataset statistics:")
    print(f"Samples per language: {samples_per_lang}")
    print(f"Validation split: {val_split * 100:.1f}%")

    print("\nTrain set:")
    print(f"  Russian (ru) words: {train_counts['ru']}")
    print(f"  Kazakh (kz) words: {train_counts['kz']}")
    print(f"  English (eng) words: {train_counts['eng']}")
    print(f"  Total words: {sum(train_counts.values())}")
    print(f"  Dataset size: {train_size / (1024 * 1024):.2f} MB")

    print("\nValidation set:")
    print(f"  Russian (ru) words: {val_counts['ru']}")
    print(f"  Kazakh (kz) words: {val_counts['kz']}")
    print(f"  English (eng) words: {val_counts['eng']}")
    print(f"  Total words: {sum(val_counts.values())}")
    print(f"  Dataset size: {val_size / (1024 * 1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Create balanced OCR dataset with train/val split'
    )
    parser.add_argument(
        '--output_dir', required=True, help='Output directory for balanced dataset'
    )
    parser.add_argument(
        '--samples_per_lang',
        type=int,
        required=True,
        help='Number of samples to take per language',
    )
    parser.add_argument(
        '--val_split',
        type=float,
        required=True,
        help='Fraction of data to use for validation (e.g. 0.2 for 20%)',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers for parallel processing',
    )
    args = parser.parse_args()

    if not (0 < args.val_split < 1):
        raise ValueError("val_split must be between 0 and 1")

    datasets_ru = [r'E:\ocr_datasets\ru\ddi\result\word_images']
    datasets_kz = [
        r'E:\ocr_datasets\kz\prepared_big_dataset_kz\big_dataset_kz\train_images'
    ]
    datasets_eng = [
        r'E:\ocr_datasets\eng\invoices-and-receipts_ocr_v1\images',
        r'E:\ocr_datasets\eng\DDI_new\word_images',
    ]

    annotations_ru = [r'E:\ocr_datasets\ru\ddi\result\annotations.json']
    annotations_kz = [
        r'E:\ocr_datasets\kz\prepared_big_dataset_kz\big_dataset_kz\train.json'
    ]
    annotations_eng = [
        r'E:\ocr_datasets\eng\invoices-and-receipts_ocr_v1\annotations.json',
        r'E:\ocr_datasets\eng\DDI_new\annotations.json',
    ]

    assert len(datasets_kz) != 0, "Fill datasets_kz!"
    assert len(datasets_eng) != 0, "Fill datasets_eng!"

    create_balanced_dataset(
        output_dir=args.output_dir,
        datasets_ru=datasets_ru,
        datasets_kz=datasets_kz,
        datasets_eng=datasets_eng,
        annotations_ru=annotations_ru,
        annotations_kz=annotations_kz,
        annotations_eng=annotations_eng,
        samples_per_lang=args.samples_per_lang,
        val_split=args.val_split,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
