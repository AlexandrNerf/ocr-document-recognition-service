# Проект распознавания мультиязычных текстов OCR

## Установка окружения

```
conda create -n ocr-project python=3.12.8
conda activate ocr-project

pip install poetry=2.2.1
poetry install
```

## Запуск ядра

Перед началом запуска надо подтянуть веса

```
dvc get https://huggingface.co/NerfmanOriginal/ocr-diploma-models crnn_vgg16_lstm256_baseline__2025_10_08.pth -o /weights
```

Теперь можно работать с нашим ядром

Для этого переходим в папку `/core`

Там инструкции и подробный пайплайн
