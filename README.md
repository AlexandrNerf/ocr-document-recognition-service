# Проект распознавания мультиязычных текстов OCR

## Установка окружения

```
conda create -n ocr-project python=3.12.8
conda activate ocr-project

pip install poetry=2.2.1
poetry install
```

## Запуск

Перед началом запуска надо подтянуть веса

```
dvc pull -r models
```

Теперь можно работать с нашим ядром

Для этого переходим в папку `/core`

Там инструкции и подробный пайплайн
