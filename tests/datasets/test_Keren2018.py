def test_Keren2018():
    from ai4bmr_datasets.datasets.Keren2018 import Keren2018
    from pathlib import Path

    base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Keren2018')
    image_version = mask_version = metadata_version = 'published'
    feature_version = 'published-unfiltered'

    ds = self = Keren2018(base_dir=base_dir)
    ds.download()
    ds.setup(image_version=image_version, mask_version=mask_version,
             metadata_version=metadata_version, feature_version=feature_version,
             load_metadata=True, load_intensity=True)
