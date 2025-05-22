

def test_Jackson2023():
    from pathlib import Path
    from ai4bmr_datasets.datasets.Jackson2023 import Jackson2023
    image_version='published'
    mask_version='published'
    metadata_version='published'
    ds = self = Jackson2023(base_dir=Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Jackson2023'))
    ds.download()
    ds.prepare_data()
    ds.setup(image_version=image_version, mask_version=mask_version, metadata_version=metadata_version)

test_Jackson2023()