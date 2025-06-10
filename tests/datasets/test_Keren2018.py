def test_Keren2018():
    from ai4bmr_datasets import Keren2018
    from pathlib import Path

    base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Keren2018')
    image_version = mask_version = metadata_version = feature_version = 'published'

    ds = self = Keren2018(base_dir=base_dir)
    ds.setup(image_version=image_version, mask_version=mask_version,
             metadata_version=metadata_version, load_metadata=True,
             feature_version=feature_version, load_intensity=True, load_spatial=True)
