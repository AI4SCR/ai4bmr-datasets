
def test_Cords2024_process_rdata():
    from ai4bmr_datasets import Cords2024 as Dataset

    ds = Dataset()
    ds.process_rdata(force=True)

    assert (ds.raw_dir / 'single_cell_experiment_objects/SingleCellExperiment Objects/intensity.parquet').exists()
    assert (ds.raw_dir / 'single_cell_experiment_objects/SingleCellExperiment Objects/spatial.parquet').exists()
    assert (ds.raw_dir / 'single_cell_experiment_objects/SingleCellExperiment Objects/cell_types.parquet').exists()


def test_Cords2024():
    from ai4bmr_datasets import Cords2024 as Dataset

    image_version = mask_version = metadata_version = feature_version = 'published'
    ds = self = Dataset()
    ds.setup(image_version=image_version, mask_version=mask_version,
             metadata_version=metadata_version, load_metadata=True,
             feature_version=feature_version, load_intensity=True, load_spatial=True)

    assert isinstance(ds.images, dict)
    assert isinstance(ds.masks, dict)
    import pandas as pd
    assert isinstance(ds.metadata, pd.DataFrame)
    assert isinstance(ds.intensity, pd.DataFrame)

