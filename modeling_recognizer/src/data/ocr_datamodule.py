import multiprocessing as mp
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomGrayscale,
    RandomPerspective,
    RandomPhotometricDistort,
)
from hydra.utils import instantiate
from doctr import transforms as T

from omegaconf import DictConfig, OmegaConf
# Аугментации
from src.data.components.ocr_dataset import OCRDataset


class OCRDataModule(LightningDataModule):
    """`LightningDataModule` для OCR датасета.

    Docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_path: str,
        input_shape: list[int],
        augs: str = None, 
        dataset: str = 'ocr_dataset_v1.parquet',
        vocab: str = 'АӘБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФҺЦЧШЩЪЫІЬЭЮЯаәбвгғдеёжзийкқлмнңоөпрстуұүфһцчшщъыіьэюя0123456789!$#()?-.,:;@%&*+=[]{}',  # Полный алфавит
        timesteps: int = 32,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_shape = tuple()
        self.batch_size_per_device = self.hparams.batch_size

        if not isinstance(self.hparams.num_workers, int):
            self.num_workers = min(8, mp.cpu_count())
        else:
            self.num_workers = self.hparams.num_workers

        augs_path = f'configs/data/augmentations/{self.hparams.augs}.yaml'
        self.augmentations = instantiate(OmegaConf.load(augs_path))
        print(self.augmentations)
        self.train_transform = Compose([
            T.Resize((input_shape[1], input_shape[2]), preserve_aspect_ratio=True),
            # Augmentations
            *self.augmentations
        ])
        print(self.train_transform)
        self.val_transform = Compose([
            T.Resize((input_shape[1], input_shape[2]), preserve_aspect_ratio=True),
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Опционально: проверка наличия данных"""
        self.dataset_frame = pd.read_parquet(self.hparams.dataset)


    def setup(self, stage: Optional[str] = None) -> None:
        """Загрузка данных и создание датасетов"""
        # Адаптация batch size для multi-GPU
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) не делится на число устройств ({self.trainer.world_size})"
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # Создаем датасеты только если они еще не созданы
        if not self.train_dataset and not self.val_dataset:
            data_len = len(self.dataset_frame)
            train_frac = 0.8
            train_idx = np.random.choice(data_len, size=int(data_len * train_frac), replace=False)
            
            # создаём маску
            train = np.zeros(data_len, dtype=bool)
            train[train_idx] = True

            self.train_dataset = OCRDataset(
                data_path=self.hparams.data_path,
                images=self.dataset_frame[train]['image_path'],
                texts=self.dataset_frame[train]['text'],
                timesteps=self.hparams.timesteps,
                vocab=self.hparams.vocab,
                transform=self.train_transform,
            )

            self.val_dataset = OCRDataset(
                data_path=self.hparams.data_path,
                images=self.dataset_frame[~train]['image_path'],
                texts=self.dataset_frame[~train]['text'],
                timesteps=self.hparams.timesteps,
                vocab=self.hparams.vocab,
                transform=self.val_transform,
            )
            self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.hparams.persistent_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.hparams.persistent_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.hparams.persistent_workers
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Очистка ресурсов при необходимости"""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
