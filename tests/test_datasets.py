from pathlib import Path


def test_PCa():
    from ai4bmr_datasets.datasets import PCa

    base_dir = Path("/Users/adrianomartinelli/data/datasets/PCa")
    dataset = PCa(base_dir=base_dir)
    dataset.setup()
    data = dataset.load()


def test_BLCa():
    from pathlib import Path
    from ai4bmr_datasets.datasets import BLCa

    base_dir = Path("/Users/adrianomartinelli/data/datasets/BLCa")
    dataset = BLCa(base_dir=base_dir)
    dataset.setup()
    data = dataset.load()


def test_TNBC():
    from ai4bmr_datasets.datasets.TNBC import TNBCv2
    from pathlib import Path

    base_dir = Path("~/data/datasets/TNBC").expanduser().resolve()
    ds = TNBCv2(base_dir=base_dir)
    # ds.process()
    data = ds.load()
