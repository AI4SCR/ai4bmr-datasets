import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imsave
from ..datamodels.Image import Image

class DummyImages:

    def __init__(
        self,
        save_dir: Path = Path("~/data/datasets/dummy-images").expanduser(),
        num_samples: int = 10,
        num_channels: int = 3,
        height: int = 32,
        width: int = 32,
        num_classes: int = 2,
    ):
        self.image = self.metadata = None

        self.save_dir = Path(save_dir).expanduser().resolve()
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(exist_ok=True, parents=True)
        self.panel_path = self.images_dir / "panel.parquet"
        self.metadata_dir = self.save_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True, parents=True)

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.num_classes = num_classes

        rng = np.random.default_rng(seed=42)

        targets = rng.integers(0, num_classes, self.num_samples)
        for i, label in enumerate(targets):

            image = rng.random((self.num_channels, self.height, self.width)).astype(np.float32)
            imsave(self.images_dir / f"{i}.tiff", image)

            metadata = pd.DataFrame({"target": label}, index=[i])
            metadata.index.name = "sample_id"
            metadata.to_parquet(self.metadata_dir / f"{i}.parquet")

        panel = pd.DataFrame({"target": range(num_channels)})
        panel.to_parquet(self.panel_path)

    def setup(self):
        self.images = [
            Image(data_path=p, metadata_path=self.panel_path)
            for p in self.images_dir.glob("*.tiff")
        ]
        self.metadata = pd.read_parquet(self.metadata_dir)

from torch.utils.data import Dataset
class DummyImagesTorch(Dataset):
    def __init__(self, num_samples=1000, num_classes=2, num_channels=3, height=224, width=224):
        from ai4bmr_datasets.datasets.DummyImages import DummyImages
        self.dataset = DummyImages(num_samples=num_samples, num_classes=num_classes, num_channels=num_channels,
                                   height=height, width=width)

    def setup(self):
        self.dataset.setup()

    def __len__(self):
        return self.dataset.num_samples

    def __getitem__(self, idx):
        image = self.dataset.images[idx].data
        target = self.dataset.metadata.loc[idx, 'target']

        image = torch.from_numpy(image).float()
        target = torch.tensor(target).long()
        return {'image': image.data, 'target': target}