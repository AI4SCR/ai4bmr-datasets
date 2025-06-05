
def test_Cords2024():
    from ai4bmr_datasets import Cords2024 as Dataset
    from pathlib import Path
    image_version = mask_version = metadata_version = feature_version = 'published'
    datasets_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/')
    ds = self = Dataset(base_dir=datasets_dir / Dataset.name)
    ds.setup(image_version=image_version, mask_version=mask_version,
             metadata_version=metadata_version, load_metadata=True,
             feature_version=feature_version, load_intensity=True, load_spatial=True)

