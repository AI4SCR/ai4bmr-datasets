from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from loguru import logger


class ImageNet2012:
    """Thin ai4bmr-datasets wrapper around ``torchvision.datasets.ImageNet``."""

    name = "ImageNet2012"
    id = "ILSVRC2012"
    doi = "10.1007/s11263-015-0816-y"

    DEFAULT_SOURCES = {
        "devkit": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
        "train": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
        "val": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    }

    ARCHIVES = {
        "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
        "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
        "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    }

    REQUIRED_ARCHIVES = {
        "train": ("devkit", "train"),
        "val": ("devkit", "val"),
    }

    def __init__(
        self,
        base_dir: Path | str | None = None,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable | None = None,
    ):
        if split not in self.REQUIRED_ARCHIVES:
            raise ValueError("ImageNet2012 split must be one of: train, val")

        self.base_dir = self.resolve_base_dir(base_dir=base_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.dataset = None

    def resolve_base_dir(self, base_dir: Path | str | None) -> Path:
        if base_dir is None:
            base_dir = os.environ.get("AI4BMR_DATASETS_DIR", None)
            if base_dir is None:
                return Path.home().resolve() / ".cache" / "ai4bmr_datasets" / self.name
            return Path(base_dir) / self.name
        return Path(base_dir)

    @property
    def required_archive_keys(self) -> tuple[str, ...]:
        return self.REQUIRED_ARCHIVES[self.split]

    def download(self, sources: dict[str, str | Path] | None = None, force: bool = False):
        """Download or copy the archives needed by the configured split.

        ``sources`` may override any default source with an HTTPS URL or local path.
        ImageNet's official image archives can require login; failures from the
        default URLs include guidance for supplying explicit sources.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)

        merged_sources = dict(self.DEFAULT_SOURCES)
        if sources:
            unknown_keys = set(sources) - set(self.ARCHIVES)
            if unknown_keys:
                raise ValueError(f"Unknown ImageNet2012 source keys: {sorted(unknown_keys)}")
            merged_sources.update({key: str(value) for key, value in sources.items()})

        for key in self.required_archive_keys:
            file_name, expected_md5 = self.ARCHIVES[key]
            target_path = self.base_dir / file_name
            source = merged_sources[key]

            if target_path.exists() and not force:
                logger.info(f"Skipping {target_path.name}, already exists.")
            else:
                if self._is_url(source):
                    self._download_url(source=source, target_path=target_path)
                else:
                    self._copy_local_source(source=Path(source).expanduser(), target_path=target_path)

            self._validate_md5(path=target_path, expected_md5=expected_md5, source=source)

    def setup(self):
        try:
            from torchvision.datasets import ImageNet
        except ImportError as error:
            raise ImportError(
                "ImageNet2012 requires torchvision. Install it with `pip install ai4bmr-datasets[torch]` "
                "or install torchvision separately."
            ) from error

        kwargs = {
            "root": self.base_dir,
            "split": self.split,
            "transform": self.transform,
            "target_transform": self.target_transform,
        }
        if self.loader is not None:
            kwargs["loader"] = self.loader

        self.dataset = ImageNet(**kwargs)
        return self

    def __len__(self) -> int:
        self._require_setup()
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        self._require_setup()
        image, target = self.dataset[idx]
        path = Path(self.dataset.samples[idx][0])
        class_name = self.dataset.classes[target]
        wnid = self.dataset.wnids[target]

        try:
            sample_id = path.relative_to(self.base_dir).as_posix()
        except ValueError:
            sample_id = path.stem

        return {
            "sample_id": sample_id,
            "image": image,
            "target": target,
            "class_index": target,
            "class_name": class_name,
            "wnid": wnid,
            "path": path,
            "split": self.split,
        }

    def _require_setup(self):
        if self.dataset is None:
            raise RuntimeError("Call setup() before accessing ImageNet2012 samples.")

    @staticmethod
    def _is_url(source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in {"http", "https"}

    def _download_url(self, source: str, target_path: Path):
        import requests
        from tqdm import tqdm

        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        tmp_path.unlink(missing_ok=True)

        logger.info(f"Downloading {source} -> {target_path}")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            with requests.get(source, headers=headers, stream=True, allow_redirects=True) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError("received an HTML response instead of an ImageNet archive")

                total_size = int(response.headers.get("Content-Length", 0))
                with tmp_path.open("wb") as file_handle, tqdm(
                    desc=target_path.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file_handle.write(chunk)
                            progress.update(len(chunk))

            tmp_path.replace(target_path)
        except Exception as error:
            tmp_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {target_path.name} from {source}. ImageNet downloads may require login. "
                "Pass explicit `sources` with accessible URLs or local archive paths, or place the official "
                f"archive at {target_path}."
            ) from error

    @staticmethod
    def _copy_local_source(source: Path, target_path: Path):
        if not source.exists():
            raise FileNotFoundError(f"ImageNet2012 source archive does not exist: {source}")

        if source.resolve() == target_path.resolve():
            logger.info(f"Using existing archive {target_path}")
            return

        logger.info(f"Copying {source} -> {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_path)

    @classmethod
    def _validate_md5(cls, path: Path, expected_md5: str, source: str):
        if not path.exists():
            raise FileNotFoundError(f"Expected ImageNet2012 archive at {path}")

        actual_md5 = cls._compute_md5(path)
        if actual_md5 != expected_md5:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {path.name} (expected {expected_md5}, got {actual_md5}). "
                "ImageNet downloads may require login; pass explicit `sources` with accessible URLs or local "
                "archive paths if the default URL returned an authentication page or incomplete archive."
            )

    @staticmethod
    def _compute_md5(path: Path) -> str:
        hasher = hashlib.md5()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
