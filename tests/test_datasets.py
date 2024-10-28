from pathlib import Path


def test_PCa():
    from ai4bmr_datasets.datasets import PCa
    base_dir = Path('/Users/adrianomartinelli/data/datasets/PCa')
    dataset = PCa(base_dir=base_dir)
    dataset.setup()
    data = dataset.load()


def test_BLCa():
    from pathlib import Path
    from ai4bmr_datasets.datasets import BLCa
    base_dir = Path('/Users/adrianomartinelli/data/datasets/BLCa')
    dataset = BLCa(base_dir=base_dir)
    dataset.setup()
    data = dataset.load()


def test_TNBC():
    from ai4bmr_datasets.datasets import TNBC
    base_dir = Path('/Users/adrianomartinelli/data/datasets/TNBC')
    dataset = TNBC(base_dir=base_dir)
    dataset.setup()
    data = dataset.load()
