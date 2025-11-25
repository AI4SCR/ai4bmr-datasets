import os

from dotenv import load_dotenv
load_dotenv()

import tempfile
from pathlib import Path

import pytest


def prepare_dataset(base_dir: Path, dataset_name: str):
    match dataset_name:
        case "Jackson2020":
            from ai4bmr_datasets import Jackson2020 as Dataset
        case "Danenberg2022":
            from ai4bmr_datasets import Danenberg2022 as Dataset
        case "Cords2024":
            from ai4bmr_datasets import Cords2024 as Dataset
        case "Keren2018":
            from ai4bmr_datasets import Keren2018 as Dataset
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    image_version = metadata_version = feature_version = 'published'
    mask_version = 'published_cell' if dataset_name == "Danenberg2022" else 'published'
    ds = Dataset(base_dir=base_dir,
                 image_version=image_version,
                 mask_version=mask_version,
                 metadata_version=metadata_version, load_metadata=True,
                 feature_version=feature_version, load_intensity=True)
    ds.prepare_data()
    ds.setup()

    assert isinstance(ds.images, dict)
    assert isinstance(ds.masks, dict)
    import pandas as pd
    assert isinstance(ds.metadata, pd.DataFrame)
    assert isinstance(ds.intensity, pd.DataFrame)


@pytest.mark.skip(reason="Only run when downloading changes. This will trigger the whole end-to-end runs.")
@pytest.mark.parametrize("dataset_name", ["Keren2018", "Jackson2020", "Danenberg2022", "Cords2024"])
def test_prepare_data(dataset_name):
    tmpdir = None
    if tmpdir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            prepare_dataset(base_dir=base_dir, dataset_name=dataset_name)
    else:
        prepare_dataset(base_dir=tmpdir, dataset_name=dataset_name)

@pytest.mark.skip(reason="Only run when downloading changes. This will trigger the whole end-to-end runs.")
@pytest.mark.parametrize("dataset_name", ["Keren2018", "Jackson2020", "Danenberg2022", "Cords2024"])
def test_checksums(dataset_name):
    dataset_name = 'Keren2018'
    match dataset_name:
        case "Jackson2020":
            from ai4bmr_datasets import Jackson2020 as Dataset
        case "Danenberg2022":
            from ai4bmr_datasets import Danenberg2022 as Dataset
        case "Cords2024":
            from ai4bmr_datasets import Cords2024 as Dataset
        case "Keren2018":
            from ai4bmr_datasets import Keren2018 as Dataset
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    ds = self = Dataset(base_dir=None, image_version='', mask_version='')
    ds.download()