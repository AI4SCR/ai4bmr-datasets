def test_Danenberg2022():
    from pathlib import Path
    from ai4bmr_datasets import Danenberg2022 as Dataset
    datasets_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/')

    image_version = metadata_version = feature_version = 'published'
    ds = self = Dataset(base_dir=datasets_dir / Dataset.name)
    ds.prepare_data()
    ds.setup(image_version=image_version, mask_version='published_cell',
             metadata_version=metadata_version, load_metadata=True,
             feature_version=feature_version, load_intensity=True)

    assert isinstance(ds.images, dict)
    assert isinstance(ds.masks, dict)
    import pandas as pd
    assert isinstance(ds.metadata, pd.DataFrame)
    assert isinstance(ds.intensity, pd.DataFrame)