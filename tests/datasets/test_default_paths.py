from ai4bmr_core.datasets.Dataset import BaseDataset
from pathlib import Path


def test_default_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = "None"

        def load(self):
            return "Hello World"

    d = D()
    assert d.base_dir == Path.home() / ".cache" / "ai4bmr" / "datasets" / "World"
    assert d.raw_dir == d.base_dir / "01_raw"
    assert d.processed_dir == d.base_dir / "02_processed"
    assert d.force_download is False
