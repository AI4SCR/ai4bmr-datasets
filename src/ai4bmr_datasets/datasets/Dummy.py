import torch
from torch.utils.data import Dataset

from .BaseDataset import BaseDataset


class DummyTabular(BaseDataset):

    def __init__(self, n_samples: int = 1000, n_features: int = 10):
        self.n_samples = n_samples
        self.n_features = n_features
        super().__init__(lazy_setup=False)

    def load(self):
        return [i for i in torch.rand(self.n_samples, self.n_features, dtype=torch.float32)]


class DummyImages(BaseDataset, Dataset):

    def __init__(self, n: int = 1000, n_channels: int = 3, height: int = 32, width: int = 32):
        super().__init__()

        self.n: int = n
        self.n_channels: int = n_channels
        self.height: int = height
        self.width: int = width

        self.imgs = torch.rand(self.n, self.n_channels, self.height, self.width, dtype=torch.float32)
        self.targets = torch.randint(0, 10, (self.n,))

    def load(self):
        return [{'image': i, 'target': t} for i, t in zip(self.imgs, self.targets)]


class DummyImagesPlain(DummyImages):

    def __init__(self, n: int = 1000, n_channels: int = 3, height: int = 32, width: int = 32):
        super().__init__(n=n, n_channels=n_channels, height=height, width=width)

    def load(self):
        return [(i, t) for i, t in zip(self.imgs, self.targets)]


