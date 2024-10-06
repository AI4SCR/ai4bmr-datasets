from pathlib import Path

from ai4bmr_core.datasets.Dataset import BaseDataset


def test_base_dir_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""

        def load(self):
            return "Hello World"

    d = D(base_dir=Path("~/Downloads/ai4bmr-core/").expanduser())
    assert d.base_dir == Path("~/Downloads/ai4bmr-core/").expanduser()
    assert d.raw_dir == d.base_dir / "01_raw"
    assert d.processed_dir == d.base_dir / "02_processed"


def test_raw_processed_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def load(self):
            return "Hello World"

    d = D(
        raw_dir=Path("~/Downloads/random_path/raw").expanduser(),
        processed_dir=Path("~/processed").expanduser(),
    )
    assert d.base_dir == Path.home() / ".cache" / "ai4bmr" / "datasets" / "World"
    assert d.raw_dir == Path("~/Downloads/random_path/raw").expanduser()
    assert d.processed_dir == Path("~/processed").expanduser()


def test_dir_raw_processed_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""

        def load(self):
            return "Hello World"

    d = D(
        base_dir=Path("~/Downloads/dir/").expanduser(),
        raw_dir=Path("~/Downloads/random_path/raw").expanduser(),
        processed_dir=Path("~/processed").expanduser(),
    )
    assert d.base_dir == Path("~/Downloads/dir/").expanduser()
    assert d.raw_dir == Path("~/Downloads/random_path/raw").expanduser()
    assert d.processed_dir == Path("~/processed").expanduser()
