import numpy as np
import torch
from .BaseDataset import BaseDataset
from torch.utils.data import Dataset


class DummyTabular(BaseDataset, Dataset):
    _id = 'dummy_tabular'
    _name = 'dummy_tabular'
    _data: np.ndarray

    def load(self):
        return np.random.rand(1000, 10)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class DummyImages(BaseDataset, Dataset):
    _id = 'dummy_imgs'
    _name = 'dummy_images'
    _data: torch.Tensor
    n: int
    c: int
    h: int
    w: int

    def __init__(self, n: int = 1000, c: int = 3, h: int = 32, w: int = 32):
        super().__init__(n=n, c=c, h=h, w=w)

    def load(self):
        imgs = torch.rand(self.n, self.c, self.h, self.w, dtype=torch.float32)
        targets = torch.randint(0, 10, (self.n,))
        return [tuple([img, targets]) for img, target in zip(imgs, targets)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


