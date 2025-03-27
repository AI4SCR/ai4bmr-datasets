from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch.utils.data import Dataset


class Split(str, Enum):
    COLUMN_NAME = "split"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def generate_splits(
    metadata: pd.DataFrame,
    target_column_name: str = "target",
    test_size: float | None = None,
    val_size: float | None = None,
    stratify: bool = False,
    random_state: int = 42,
    verbose: int = 1,
):
    assert test_size or val_size, "Either `test_size` or `val_size` must be provided"

    # TODO: support Grouped Split for multi-samples per patient cases
    # TODO: should we allow for re-splitting, i.e. check if `split_column_name` already exists in metadata?
    # TODO: should we overwrite the metadata?

    metadata = metadata.copy()
    assert metadata.index.has_duplicates == False
    num_samples = len(metadata)

    # split into train, test
    if test_size:
        splitter = StratifiedShuffleSplit if stratify else ShuffleSplit
        splitter = splitter(n_splits=1, test_size=test_size, random_state=random_state)

        y = metadata[target_column_name] if stratify else None
        train_indices, test_indices = next(splitter.split(np.zeros(num_samples), y=y))

        train_indices = metadata.index[train_indices]
        test_indices = metadata.index[test_indices]
    else:
        train_indices = metadata.index
        test_indices = []

    # split train into train, val
    if val_size:
        train_metadata, test_metadata = (
            metadata.loc[train_indices],
            metadata.loc[test_indices],
        )
        num_train_samples = len(train_metadata)

        val_splitter = StratifiedShuffleSplit if stratify else ShuffleSplit
        val_splitter = val_splitter(n_splits=1, test_size=val_size, random_state=random_state)

        y = train_metadata[target_column_name] if stratify else None
        train_indices, val_indices = next(val_splitter.split(np.zeros(num_train_samples), y=y))

        train_indices = train_metadata.index[train_indices]
        val_indices = train_metadata.index[val_indices]
    else:
        val_indices = []

    # sanity checks
    assert set(train_indices).union(val_indices).union(test_indices) == set(metadata.index.values)
    assert set(train_indices).intersection(test_indices) == set()
    assert set(val_indices).intersection(test_indices) == set()

    logger.info(
        f"Split dataset in:\n"
        f"  num_test: {len(test_indices)}\n"
        f"  num_train: {len(train_indices)}\n"
        f"  num_val: {len(val_indices)}"
    )

    metadata.loc[test_indices, Split.TEST] = "test"
    metadata.loc[train_indices, Split.TRAIN] = "train"
    metadata.loc[val_indices, Split.VAL] = "val"
    assert metadata.split.isna().any() == False

    return metadata


def has_splits(metadata: pd.DataFrame):
    return Split.COLUMN_NAME in metadata.columns


def generate_subset(dataset: Dataset, metadata: pd.DataFrame | None, num_samples: int | None = None):
    from torch.utils.data import Subset

    subset_idx = np.random.choice(len(metadata), num_samples, replace=False)
    return Subset(dataset, subset_idx), metadata.iloc[subset_idx]
