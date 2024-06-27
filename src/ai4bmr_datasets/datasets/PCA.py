import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from pydantic import ConfigDict
from pydantic import Field, BaseModel

from .BaseDataset import BaseDataset
from ..data_models.Dataset import Dataset


class Sample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_name: str

    img_path: str | Path
    masks_path: str | Path
    panel_path: str | Path

    in_memory: bool = False
    _img: np.ndarray = None
    _masks: np.ndarray = None
    # _mask: np.ndarray = None
    _panel: pd.DataFrame = None

    @property
    def img(self) -> np.ndarray:
        _img = self._img
        if _img is None:
            _img = tiff.imread(self.img_path)
            self._img = _img if self.in_memory else None
        return _img

    # @computed_field
    # @property
    # def mask(self) -> np.ndarray:
    #     _mask = self._mask
    #     if _mask is None:
    #         _mask = np.expand_dims(tiff.imread(self.mask_path), 0)
    #         self._mask = _mask if self.in_memory else None
    #     return _mask

    @property
    def masks(self) -> np.ndarray:
        _masks = self._masks
        if _masks is None:
            _masks = np.expand_dims(tiff.imread(self.masks_path), 0)
            self._masks = _masks if self.in_memory else None
        return _masks

    @property
    def panel(self) -> pd.DataFrame:
        _panel = self._panel
        if _panel is None:
            _panel = pd.read_csv(self.panel_path)
            self._panel = _panel if self.in_memory else None
        return _panel

    @property
    def height(self) -> int:
        return self.img.shape[-2]

    @property
    def width(self) -> int:
        return self.img.shape[-1]


class PCA(BaseDataset):
    model_config = Dataset.model_config | ConfigDict(extra='allow')
    _id = 'pca'
    _name = 'pca'
    _data = dict[str, any]

    def __init__(self, base_dir: Path, img_version: str = '', mask_version: str = '', **kwargs):
        raw_dir = base_dir / "raw"
        processed_dir = base_dir

        super().__init__(base_dir=base_dir, raw_dir=raw_dir, processed_dir=processed_dir, **kwargs)

        self.configs_dir: Path = self.processed_dir / "configs"

        self.imgs_base_dir: Path = self.processed_dir / "imgs"
        self.img_version: str = img_version
        self.imgs_dir: Path = self.imgs_base_dir / self.img_version

        self.panel_path: Path = self.imgs_dir / 'panel.csv'
        self.panel: pd.DataFrame = None

        self.masks_base_dir: Path = self.processed_dir / "masks"
        self.mask_version: str = mask_version
        self.masks_dir: Path = self.masks_base_dir / self.mask_version

        self.mcd_metadata_dir = self.processed_dir / "mcd_metadata"
        self.mcd_metadata: pd.DataFrame = None

        self.sample_names: list[Path] = None

    def load(self):
        self.panel = pd.read_csv(self.panel_path)

        if self.mcd_metadata_dir.exists():
            self.mcd_metadata = self.mcd_metadata_dir.glob('**/*.txt')
            self.mcd_metadata = pd.concat([pd.read_csv(f) for f in self.mcd_metadata_dir.glob('**/*.txt')])
            self.mcd_metadata = self.mcd_metadata.assign(
                sample_name=self.mcd_metadata['file_name_napari'].str.replace('.tiff', '')
            )

        img_filenames = set([f.stem for f in self.imgs_dir.glob('*.tiff')])
        mask_filenames = set([f.stem for f in self.masks_dir.glob('*.tiff')])
        self.sample_names = sorted(img_filenames & mask_filenames)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx) -> Sample:
        sample_name = self.sample_names[idx]
        return Sample(
            sample_name=sample_name,
            img_path=self.imgs_dir / f"{sample_name}.tiff",
            masks_path=self.masks_dir / f"{sample_name}.tiff",
            panel_path=self.panel_path
        )

    # def __iter__(self):
    #     for idx in range(len(self)):
    #         yield self[idx]
