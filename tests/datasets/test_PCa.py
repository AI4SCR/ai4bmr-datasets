def test_PCa():
    from ai4bmr_datasets.datasets import PCa
    from pathlib import Path
    import pandas as pd

    base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa")
    ds = dataset = self = PCa(base_dir=base_dir)

    # self = ds
    # ds.label_transfer()
    # ds.create_annotations()

    df = pd.read_parquet(ds.annotations_dir)

    # ds.create_metadata()
    # ds.create_clinical_metadata()
    # ds.create_tma_annotations()
    # dataset.process()
    import pandas as pd

    sampels = pd.read_parquet(ds.sample_metadata_path, engine="fastparquet")
    data = dataset.load(
        image_version="filtered", mask_version="filtered", features_as_published=False
    )
    # data['samples'].to_csv('/users/amarti51/pca-annotations.csv')
    data = dataset.load(image_version="filtered", features_as_published=False)