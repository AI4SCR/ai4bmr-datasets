import pandas as pd


def test_PCa():
    from ai4bmr_datasets.datasets import PCa
    from pathlib import Path
    import pandas as pd

    base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa")
    ds = dataset = PCa(base_dir=base_dir)
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


def test_BLCa():
    from pathlib import Path
    from ai4bmr_datasets.datasets import BLCa

    base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/BLCa")
    dataset = BLCa(base_dir=base_dir)
    dataset.setup()
    data = dataset.load()


def test_TNBC():
    from ai4bmr_datasets.datasets.TNBC import TNBC
    from pathlib import Path

    base_dir = (
        Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/TNBC")
        .expanduser()
        .resolve()
    )
    ds = TNBC(base_dir=base_dir)
    # ds.process()
    data = ds.load()
    ds.panel


def test_Cords2024():
    from ai4bmr_datasets.datasets.Cords2024 import Cords2024
    from pathlib import Path

    base_dir = (
        Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/NSCLCv2")
        .expanduser()
        .resolve()
    )
    ds = Cords2024(base_dir=base_dir)
    ds.process()

    _ = ds.load()
    ds.panel

def test_Cords2023():
    from ai4bmr_datasets.datasets.Cords2023 import Cords2023
    from pathlib import Path

    base_dir = (
        Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2023")
        .expanduser()
        .resolve()
    )
    ds = Cords2023(base_dir=base_dir)
    ds.download()
