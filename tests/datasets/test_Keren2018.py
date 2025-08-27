def test_Keren2018():
    from ai4bmr_datasets import Keren2018
    image_version = mask_version = metadata_version = feature_version = 'published'
    ds = self = Keren2018(image_version=image_version, mask_version=mask_version,
                            metadata_version=metadata_version, load_metadata=True,
                            feature_version=feature_version, load_intensity=True, load_spatial=True)
    ds.prepare_data()
    ds.setup()

    assert isinstance(ds.images, dict)
    assert isinstance(ds.masks, dict)
    import pandas as pd
    assert isinstance(ds.metadata, pd.DataFrame)
    assert isinstance(ds.intensity, pd.DataFrame)