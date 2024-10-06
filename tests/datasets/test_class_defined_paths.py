from pathlib import Path

from ai4bmr_core.datasets.Dataset import BaseDataset


def test_class_define_data_dir():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""
        _base_dir = Path("~/Downloads/ai4bmr-core/").expanduser()

        def load(self):
            return "Hello World"

    d = D()
    assert d.base_dir == Path("~/Downloads/ai4bmr-core/").expanduser()
    assert d.raw_dir == d.base_dir / "01_raw"
    assert d.processed_dir == d.base_dir / "02_processed"


def test_class_raw_and_processed():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""
        _raw_dir: Path = Path("~/Downloads/ai4bmr-core/raw").expanduser()
        _processed_dir: Path = Path("~/Downloads/ai4bmr-core/process").expanduser()

        def load(self):
            return "Hello World"

    d = D()
    assert d.base_dir == Path.home() / ".cache" / "ai4bmr" / "datasets" / "World"
    assert d.raw_dir == Path("~/Downloads/ai4bmr-core/raw").expanduser()
    assert d.processed_dir == Path("~/Downloads/ai4bmr-core/process").expanduser()


def test_class_data_raw_and_processed():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""
        _base_dir = Path("~/Downloads/ai4bmr-core/").expanduser()
        _raw_dir: Path = Path("~/Downloads/ai4bmr-core/raw").expanduser()
        _processed_dir: Path = Path("~/Downloads/ai4bmr-core/process").expanduser()

        def load(self):
            return "Hello World"

    d = D()
    assert d.base_dir == Path("~/Downloads/ai4bmr-core/").expanduser()
    assert d.raw_dir == Path("~/Downloads/ai4bmr-core/raw").expanduser()
    assert d.processed_dir == Path("~/Downloads/ai4bmr-core/process").expanduser()
