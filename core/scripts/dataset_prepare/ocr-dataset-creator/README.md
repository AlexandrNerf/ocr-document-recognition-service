# Скрипты для подготовки датасетов для обучения модели OCR

## Необходимый формат меток
Пример:
```json
{
  "file-9-page-80_294.png": {
    "dimensions": [28, 80],
    "text": "танылды"
  }
}
```

## invoices-and-receipts_ocr_v1_creator.py
Скачивает с HuggingFace и приводит к нужному виду датасет ENG/Receipts OCR.

```python
python invoices-and-receipts_ocr_v1_creator.py \
  --dataset_path ./datasets/invoices \
  --images_dir ./datasets/invoices/word_images \
  --output_json ./datasets/invoices/annotations.json \
  --workers 8
```


## simple-creator.py

`python simple-creator.py --dataset_dir path/to/dataset --output_json annotations.json`

Приводит произвольный датасет в формате 

```
dataset/
├── images/       # Исходные изображения
│   ├── img1.png
│   └── img2.jpg
└── labels/       # Текстовые файлы с аннотациями
    ├── img1.txt
    └── img2.txt
```

в необходимый формат. 
