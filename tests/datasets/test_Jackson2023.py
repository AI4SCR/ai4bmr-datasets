def test_Jackson2023():
    from pathlib import Path
    from ai4bmr_datasets.datasets.Jackson2023 import Jackson2023
    image_version = mask_version = metadata_version = feature_version = 'published'
    ds = self = Jackson2023(base_dir=Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Jackson2023'))
    ds.setup(image_version=image_version, mask_version=mask_version,
             metadata_version=metadata_version, feature_version=feature_version,
             load_metadata=True, load_intensity=True)
