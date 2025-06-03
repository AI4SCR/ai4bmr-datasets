def test_PCa():
    from ai4bmr_datasets.datasets import PCa
    from pathlib import Path
    import pandas as pd

    base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa")
    ds = dataset = self = PCa(base_dir=base_dir)
    df = pd.read_parquet(ds.annotations_dir)
    sampels = pd.read_parquet(ds.sample_metadata_path, engine="fastparquet")
    data = dataset.setup(image_version="filtered", mask_version="filtered")