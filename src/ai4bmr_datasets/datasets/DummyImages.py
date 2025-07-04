import numpy as np
import pandas as pd


class DummyTabular:

    def __init__(
        self, num_samples: int = 1000, num_features: int = 10, num_classes: int = 2
    ):
        self.num_samples = num_samples
        self.num_features = num_features

        rng = np.random.default_rng(seed=42)

        self.data = pd.DataFrame(rng.random((self.num_samples, self.num_features)))
        self.data.index = self.data.index.astype(str)
        self.data.index.name = "sample_id"

        metadata = pd.DataFrame(
            rng.integers(0, num_classes, num_samples),
            columns=["label_id"],
            dtype="category",
        )

        self.metadata = metadata.convert_dtypes()
        self.metadata["label"] = [
            ["type_1", "type_2"][i] for i in self.metadata["label_id"]
        ]
        self.metadata["label"] = self.metadata["label"].astype("category")
        self.metadata = self.metadata.convert_dtypes()
        self.metadata.index = self.metadata.index.astype(str)
        self.metadata.index.name = "sample_id"

        self.sample_ids = self.metadata.index.to_list()

    def load(self):
        return dict(data=self.data, metadata=self.metadata)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return {
            "sample_id": sample_id,
            "data": self.data.loc[sample_id].to_numpy(),
            "metadata": self.metadata.loc[sample_id].to_dict(),
        }


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
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(exist_ok=True, parents=True)
        self.panel_path = self.images_dir / "panel.parquet"
        self.metadata_dir = self.save_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True, parents=True)

        self.num_sample = num_samples
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.num_classes = num_classes

        rng = np.random.default_rng(seed=42)

        targets = rng.integers(0, num_classes, self.num_sample)
        for i, label in enumerate(targets):
            image = rng.random((self.num_channels, self.height, self.width)).astype(
                np.float16
            )
            imsave(self.images_dir / f"{i}.tiff", image)
            metadata = pd.DataFrame({"label_id": label}, index=[i])
            metadata.index.name = "sample_id"
            metadata.to_parquet(self.metadata_dir / f"{i}.parquet")

        panel = pd.DataFrame({"target": range(num_channels)})
        panel.to_parquet(self.panel_path)

    def setup(self):
        images = [
            Image(data_path=p, metadata_path=self.panel_path)
            for p in self.images_dir.glob("*.tiff")
        ]
        metadata = pd.read_parquet(self.metadata_dir)
        return dict(images=images, metadata=metadata)
