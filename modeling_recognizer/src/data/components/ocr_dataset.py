import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from doctr.datasets import RecognitionDataset

class OCRDataset(Dataset):
    def __init__(self, data_path, images, texts, timesteps, vocab, transform):
        self.images_path = os.path.join(data_path, f'/ocr_dataset/images/')
        self.samples = [
            (os.path.join(data_path, image), text) for image, text in zip(images, texts)
        ]
        self.transform = transform
        self.ts = timesteps

        self.char_list = {ix + 1: char for ix, char in enumerate(vocab)}
        self.char_list[0] = '`'
        self.inv_char_list = {v: k for k, v in self.char_list.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.images_path, filename)
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = np.array(img) 
            img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        except Exception as e:
            print(f'Error loading {filename}: {e}')
            return None, None

        image = self.transform(img)
        return image, label, img_path

    def str2vec(self, string, pad=True):
        string = ''.join([s for s in string if s in self.inv_char_list])
        val = list(map(self.inv_char_list.get, string))
        if pad and len(val) < self.ts:
            val += [0] * (self.ts - len(val))
        return val

    def collate_fn(self, batch):
        batch = [item for item in batch if item[0] is not None]
        if not batch:
            return None, None

        images, labels, img_paths = zip(*batch)
        images = torch.stack(images)
        return (
            images,
            labels,
            img_paths
        )
