def test_annotated():
    from ai4bmr_datasets import Keren2018 as Dataset
    from ai4bmr_datasets import Jackson2020 as Dataset
    from ai4bmr_datasets import Danenberg2022 as Dataset
    from pathlib import Path

    ds = self = Dataset(base_dir=Path('/users/amarti51/prometex/data/datasets/') / Dataset.name)
    ds.create_annotated(mask_version='published_cell')
