from pathlib import Path


def test_PCa():
    from ai4bmr_datasets.datasets import PCa

    base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa")
    dataset = PCa(base_dir=base_dir)
    dataset.setup()
    data = dataset.load()


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

    base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/TNBC").expanduser().resolve()
    ds = TNBC(base_dir=base_dir)
    # ds.process()
    data = ds.load()
    ds.panel


def test_NSCLC():
    from ai4bmr_datasets.datasets.NSCLC import NSCLC
    from pathlib import Path

    base_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/NSCLCv2").expanduser().resolve()
    ds = NSCLC(base_dir=base_dir)
    ds.process()

    _ = ds.load()
    ds.panel
