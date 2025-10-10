import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import fitz
import gensim
import pandas as pd
from tqdm import tqdm


def custom_preprocess(text: str, min_len: int = 2, max_len: int = 15) -> str:
    """Предобработка текста с токенизацией и фильтрацией по длине токенов."""
    tokens = []
    for token in gensim.utils.tokenize(text, lower=False):
        if min_len <= len(token) <= max_len:
            tokens.append(token)
    return ' '.join(tokens)


def clean_text(text: str) -> str:
    """Очистка текста от лишних пробелов и переносов строк."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_max_text_length(
    font_size: int = 12, page_width: int = 595, margins: int = 40
) -> int:
    """Рассчитывает примерное количество символов, которое поместится на странице PDF."""
    avg_char_width = font_size * 0.6
    chars_per_line = int((page_width - 2 * margins) / avg_char_width)
    return chars_per_line * 50  # <-- ? #TODO


def create_pdf(
    chunk: list[str],
    chunk_num: int,
    output_dir: str,
    font_family: str,
    font_size: int,
    page_width: int,
    page_height: int,
    margins: int,
) -> None:
    """Создает PDF файл из переданного чанка текста."""
    doc = fitz.open()
    page = doc.new_page(width=page_width, height=page_height)
    rect = page.rect + (margins, margins, -margins, -margins)

    processed_sentences = []
    for sentence in chunk:
        cleaned = clean_text(sentence)
        processed = custom_preprocess(cleaned)
        processed_sentences.append(processed)

    full_text = ' '.join(processed_sentences)

    html_content = f"""
    <div style='font-family: {font_family}; font-size: {font_size}px; text-align: justify;'>
        {full_text}
    </div>
    """

    page.insert_htmlbox(rect, html_content)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pdf_{chunk_num}.pdf")
    doc.save(output_path)
    doc.close()


def process_sentences(
    sentences: list[str],
    output_dir: str,
    max_workers: int = 4,
    font_family: str = "Arial",
    font_size: int = 12,
    page_width: int = 595,
    page_height: int = 842,
    margins: int = 40,
    max_chars: Optional[int] = None,
) -> None:

    if max_chars is None:
        max_chars = calculate_max_text_length(font_size, page_width, margins)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        cleaned = clean_text(sentence)
        processed = custom_preprocess(cleaned)
        if current_length + len(processed) > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [processed]
            current_length = len(processed)
        else:
            current_chunk.append(processed)
            current_length += len(processed)

    if current_chunk:
        chunks.append(current_chunk)

    success = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(
                create_pdf,
                chunk=chunk,
                chunk_num=i,
                output_dir=output_dir,
                font_family=font_family,
                font_size=font_size,
                page_width=page_width,
                page_height=page_height,
                margins=margins,
            )
            futures.append(future)

        for _ in tqdm(futures, total=len(chunks), desc="Создание PDF"):
            success += 1


def main():
    parser = argparse.ArgumentParser(description="Создание PDF из текстовых данных")

    parser.add_argument("input_csv", type=str, help="Путь к входному CSV файлу")
    parser.add_argument(
        "output_dir", type=str, help="Директория для сохранения PDF файлов"
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="sentence",
        help="Название колонки с текстом в CSV (по умолчанию: 'sentence')",
    )
    parser.add_argument(
        "--min_token_len",
        type=int,
        default=2,
        help="Минимальная длина токена (по умолчанию: 2)",
    )
    parser.add_argument(
        "--max_token_len",
        type=int,
        default=15,
        help="Максимальная длина токена (по умолчанию: 15)",
    )

    parser.add_argument(
        "--font_family",
        type=str,
        default="Arial",
        help="Шрифт для PDF (по умолчанию: Arial)",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=12,
        help="Размер шрифта в пунктах (по умолчанию: 12)",
    )
    parser.add_argument(
        "--page_width",
        type=int,
        default=595,
        help="Ширина страницы в пунктах (по умолчанию: 595 - A4)",
    )
    parser.add_argument(
        "--page_height",
        type=int,
        default=842,
        help="Высота страницы в пунктах (по умолчанию: 842 - A4)",
    )
    parser.add_argument(
        "--margins",
        type=int,
        default=40,
        help="Отступы от краев страницы (по умолчанию: 40)",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help="Максимальное количество символов на PDF (авто по умолчанию)",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Максимальное количество потоков (по умолчанию: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничение количества обрабатываемых строк",
    )

    args = parser.parse_args()

    wiki_df = pd.read_csv(args.input_csv)
    sentences = wiki_df[args.text_column].tolist()

    if args.limit is not None:
        sentences = sentences[: args.limit]

    global custom_preprocess
    original_custom_preprocess = custom_preprocess

    def wrapped_preprocess(text):
        return original_custom_preprocess(text, args.min_token_len, args.max_token_len)

    custom_preprocess = wrapped_preprocess

    process_sentences(
        sentences=sentences,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        font_family=args.font_family,
        font_size=args.font_size,
        page_width=args.page_width,
        page_height=args.page_height,
        margins=args.margins,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
