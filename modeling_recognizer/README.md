<div align="center">

# OCR modeling


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

Репозиторий для обучения моделей OCR. 
Реализация обучения CRNN и parseq из библиотеки docTR

</div>

## Структура

Вид основной директории и её содержание:


```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra конфиги
│   ├── callbacks                <- Callbacks
│   ├── data                     <- Датасеты и даталоадеры
│   ├── debug                    <- Отладка
│   ├── experiment               <- Конфиги для экспериментов
│   ├── extras                   <- Дополнительные фичи
│   ├── hparams_search           <- Поиск гиперпарамов (optuna)
│   ├── hydra                    <- Hydra доп. конфиги
│   ├── local                    <- Конфиги для локального доступа
│   ├── logger                   <- Логирование
│   ├── model                    <- Модели
│   ├── paths                    <- Конфиг с путями
│   ├── trainer                  <- Параметры обучения
│   │
│   ├── eval.yaml             <- Конфиг валидации и теста
│   └── train.yaml            <- Конфиг для обучения
│
├── logs                   <- Логи (появятся при запуске экспериментов)
│
├── notebooks              <- Тетрадки с полезными функциями
│
├── src                    <- Ядро моделинга
│   ├── data                     <- Данные
│   ├── models                   <- Модели
│   ├── lit_modules              <- Новые элементы для обучения (расписания lr, колбеки)
│   ├── utils                    <- Доп. утилиты (логирование, вывод через rich)
│   │
│   ├── eval.py                  <- Валидация
│   └── train.py                 <- Обучение
│
├── tests                  <- Тесты
│
└── README.md
```

<br>

## 🚀  Подготовка данных

Среда полностью совместима с головной средой проекта

Перед началом обучения необходимо подтянуть веса

Заходим в папку `datasets` в корневой директории, читаем readme для установки.

После загрузки должен подтянуться датасет в папку `datasets`.
Он состоит из файлов .parquet и архива с фото. 

**Важно**: при любом раскладе скачаются все фотографии (их около 900k), их число в разных версиях датасета не будет сокращаться.

Бейзлайном является датасет `ocr_dataset_v3_hnm.parquet` - версия со всеми наборами данных, но отобрано только 50 тысяч.

## 🚀  Запуск

Базовый скрипт обучения

```bash
python src/train.py
```

Для запуска экспериментов добавим:

```bash
python src/train.py experiment=<EXP_NAME>.yaml
```

Имя конфига выбираем в соответствии с `configs/experiment` файлами. На данный момент бейзлайн - `baseline_parseq_hnm_rerun.yaml`.

Для валидации:

```bash
python src/eval.py
```

<br>


## ⚡ Возможности

<details>
<summary><b>Изменение конфига в консоли</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Заметка**: Также можно добавлять параметры через `+`.

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>Обучение на GPU, CPU и даже DDP</b></summary>

```bash
python train.py trainer=cpu

python train.py trainer=gpu

python train.py +trainer.tpu_cores=8

python train.py trainer=mps
```

> **Важно**: Замечены проблемы с DistributedDataParallel запуском, для запуска может потребоваться фикс.

</details>

<details>
<summary><b>Встроенный mixed precision</b></summary>

```bash
python train.py trainer=gpu +trainer.precision=16
```

</details>

<details>
<summary><b>Поддержка всех популярных логгеров</b></summary>

В конфиге пишем

```yaml
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
python train.py logger=wandb
```

> **Заметка**: Немного информации о трекинге от авторов Lightning [here](#experiment-tracking).

> **Заметка**: Для wandb - [setup account](https://www.wandb.com/).

> **Заметка**: [Здесь](https://wandb.ai/hobglob/template-dashboard/) пример логирования через wandb


</details>

<details>
<summary><b>Callback по желанию</b></summary>

```bash
python train.py callbacks=default
```

> **Заметка**: Подробнее о настройке сохранения, ранней остановки и др [здесь](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Заметка**: Коллбеки находятся по пути [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Фишки Lightning</b></summary>

```yaml
python train.py +trainer.gradient_clip_val=0.5

python train.py +trainer.val_check_interval=0.25

python train.py +trainer.accumulate_grad_batches=10

python train.py +trainer.max_time="00:12:00:00"
```

> **Заметка**: Немного о полезных фишках: [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Простая отладка</b></summary>

```bash
python train.py debug=default

python train.py debug=fdr

python train.py debug=profiler

python train.py debug=overfit

python train.py +trainer.detect_anomaly=true

# Ограничить данные в процентном соотношении (может быть полезно для отладки)
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Заметка**: В [configs/debug/](configs/debug/) лежат настройки отладки

</details>

<details>
<summary><b>Продолжение обучения</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Заметка**: Принимается путь или url.

> **Заметка**: Логирование начинается заново

</details>

<details>
<summary><b>Валидация чекпоинта</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

</details>

<details>
<summary><b>Сетка гиперпараметров</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra обрабатывает конфиги "лениво", поэтому при запуске новой работы лучше до этого конфиги не трогать

</details>

<details>
<summary><b>Сетка гиперпараметров с Optuna</b></summary>

```bash
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **Заметка**: [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) запускается через [свой конфиг](configs/hparams_search/mnist_optuna.yaml).

> **Важно**: При ошибке одной работы последующие тоже завершаются

> **Заметка**: Пока что не используем Optuna

</details>

<details>
<summary><b>Выполнение всех экспериментов</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Заметка**:  [Здесь](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run) немного о фишках Hydra. Источник команды: [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Пре-коммит</b></summary>

```bash
pre-commit run -a
```

> **Заметка**: Использовать при работе с непосредственно репозиторием для линтеров и форматинга. Подробнее почитать 
про форматинг кода можно [здесь](#best-practices).

Обновление `.pre-commit-config.yaml`:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Тесты</b></summary>

```bash
pytest

pytest tests/test_train.py

pytest -k "not slow"
```

</details>

<details>
<summary><b>Тэги для экспериментов</b></summary>

Для обозначения запусков:

```bash
python train.py tags=["mnist","experiment_X"]
```

> **Заметка**: Для форматирования: `python train.py tags=\["mnist","experiment_X"\]`.

Если нет тегов:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

Теги обязательны для мультирана

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

</details>

<br>

