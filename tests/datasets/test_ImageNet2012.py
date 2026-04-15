from pathlib import Path
import pytest

from ai4bmr_datasets import ImageNet2012


def test_imagenet2012_import_export():
    from ai4bmr_datasets import ImageNet2012 as ExportedDataset

    assert ExportedDataset is ImageNet2012


def test_imagenet2012_resolve_default_base_dir(monkeypatch):
    monkeypatch.delenv("AI4BMR_DATASETS_DIR", raising=False)

    ds = ImageNet2012()

    expected = Path.home() / ".cache" / "ai4bmr_datasets" / ImageNet2012.name
    assert ds.base_dir.resolve() == expected.resolve()


def test_imagenet2012_resolve_base_dir_from_env(monkeypatch):
    root = Path("/tmp/ai4bmr_datasets")
    monkeypatch.setenv("AI4BMR_DATASETS_DIR", str(root))

    ds = ImageNet2012()

    assert ds.base_dir == root / ImageNet2012.name


@pytest.mark.parametrize(
    ("split", "expected_keys"),
    [
        ("train", ("devkit", "train")),
        ("val", ("devkit", "val")),
    ],
)
def test_imagenet2012_required_archives(split, expected_keys, tmp_path):
    ds = ImageNet2012(base_dir=tmp_path, split=split)

    assert ds.required_archive_keys == expected_keys


def test_imagenet2012_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split must be one of"):
        ImageNet2012(base_dir=tmp_path, split="test")


def test_imagenet2012_download_uses_default_urls(monkeypatch, tmp_path):
    ds = ImageNet2012(base_dir=tmp_path, split="val")
    calls = []

    def fake_download(source, target_path):
        calls.append((source, target_path.name))
        target_path.write_bytes(b"archive")

    monkeypatch.setattr(ds, "_download_url", fake_download)
    monkeypatch.setattr(ds, "_validate_md5", lambda path, expected_md5, source: None)

    ds.download()

    assert calls == [
        (
            ImageNet2012.DEFAULT_SOURCES["devkit"],
            ImageNet2012.ARCHIVES["devkit"][0],
        ),
        (
            ImageNet2012.DEFAULT_SOURCES["val"],
            ImageNet2012.ARCHIVES["val"][0],
        ),
    ]


def test_imagenet2012_download_overrides_default_sources(monkeypatch, tmp_path):
    ds = ImageNet2012(base_dir=tmp_path, split="val")
    calls = []

    def fake_download(source, target_path):
        calls.append((source, target_path.name))
        target_path.write_bytes(b"archive")

    monkeypatch.setattr(ds, "_download_url", fake_download)
    monkeypatch.setattr(ds, "_validate_md5", lambda path, expected_md5, source: None)

    ds.download(sources={"val": "https://mirror.example.org/ILSVRC2012_img_val.tar"})

    assert calls[-1] == (
        "https://mirror.example.org/ILSVRC2012_img_val.tar",
        ImageNet2012.ARCHIVES["val"][0],
    )


def test_imagenet2012_download_copies_local_sources(monkeypatch, tmp_path):
    devkit = tmp_path / "local_devkit.tar.gz"
    val = tmp_path / "local_val.tar"
    devkit.write_bytes(b"devkit")
    val.write_bytes(b"val")

    ds = ImageNet2012(base_dir=tmp_path / "imagenet", split="val")
    monkeypatch.setattr(ds, "_validate_md5", lambda path, expected_md5, source: None)

    ds.download(sources={"devkit": devkit, "val": val})

    assert (ds.base_dir / ImageNet2012.ARCHIVES["devkit"][0]).read_bytes() == b"devkit"
    assert (ds.base_dir / ImageNet2012.ARCHIVES["val"][0]).read_bytes() == b"val"


def test_imagenet2012_download_rejects_unknown_source_key(tmp_path):
    ds = ImageNet2012(base_dir=tmp_path, split="val")

    with pytest.raises(ValueError, match="Unknown ImageNet2012 source keys"):
        ds.download(sources={"test": "https://example.org/test.tar"})


def test_imagenet2012_download_raises_clear_error_for_html_response(monkeypatch, tmp_path):
    ds = ImageNet2012(base_dir=tmp_path, split="val")

    class FakeResponse:
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def raise_for_status(self):
            return None

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="ImageNet downloads may require login"):
        ds._download_url(
            source=ImageNet2012.DEFAULT_SOURCES["val"],
            target_path=tmp_path / ImageNet2012.ARCHIVES["val"][0],
        )


def test_imagenet2012_getitem_returns_ai4bmr_dict(tmp_path):
    image_path = tmp_path / "val" / "n01440764" / "ILSVRC2012_val_00000293.JPEG"

    class FakeImageNet:
        samples = [(str(image_path), 0)]
        classes = [("tench", "Tinca tinca")]
        wnids = ["n01440764"]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return "image", 0

    ds = ImageNet2012(base_dir=tmp_path, split="val")
    ds.dataset = FakeImageNet()

    sample = ds[0]

    assert sample == {
        "sample_id": "val/n01440764/ILSVRC2012_val_00000293.JPEG",
        "image": "image",
        "target": 0,
        "class_index": 0,
        "class_name": ("tench", "Tinca tinca"),
        "wnid": "n01440764",
        "path": image_path,
        "split": "val",
    }


def test_imagenet2012_requires_setup_before_access(tmp_path):
    ds = ImageNet2012(base_dir=tmp_path, split="val")

    with pytest.raises(RuntimeError, match="Call setup"):
        len(ds)


def test_imagenet2012_setup_instantiates_torchvision_imagenet(monkeypatch, tmp_path):
    calls = []

    class FakeImageNet:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    import sys

    fake_torchvision = type(sys)("torchvision")
    fake_datasets = type(sys)("torchvision.datasets")
    fake_datasets.ImageNet = FakeImageNet
    fake_torchvision.datasets = fake_datasets
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", fake_datasets)

    ds = ImageNet2012(base_dir=tmp_path, split="val", transform="transform", target_transform="target_transform")

    assert ds.setup() is ds
    assert isinstance(ds.dataset, FakeImageNet)
    assert calls == [
        {
            "root": tmp_path,
            "split": "val",
            "transform": "transform",
            "target_transform": "target_transform",
        }
    ]
