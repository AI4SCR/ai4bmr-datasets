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
        self.data.columns = self.data.columns.astype(str)
        self.data.index = self.data.index.astype(str)
        self.data.index.name = "sample_id"

        self.metadata = pd.DataFrame(
            rng.integers(0, num_classes, num_samples),
            columns=["label_id"],
            dtype="category",
        )
        self.metadata["label"] = [
            ["type_1", "type_2"][i] for i in self.metadata["label_id"]
        ]
        self.metadata["label"] = self.metadata["label"].astype("category")
        self.metadata = self.metadata.convert_dtypes()
        self.metadata.index = self.metadata.index.astype(str)
        self.metadata.index.name = "sample_id"

        self.sample_ids = self.metadata.index.to_list()

    def setup(self):
        pass

    def __len__(self):
        len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return {
            "sample_id": sample_id,
            "data": self.data.loc[sample_id].to_numpy(),
            "metadata": self.metadata.loc[sample_id].to_dict(),
        }
