import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from pydantic import BaseModel, ConfigDict, Field, computed_field

from .BaseDataset import BaseDataset, BaseDatasetConfig


class SegmentedImage(BaseModel):
    model_config = BaseDatasetConfig.model_config | ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_name: str

    img_path: str | Path
    masks_path: str | Path
    panel_path: str | Path
    objects_label_path: str | Path

    in_memory: bool = False
    _img: np.ndarray = None
    _masks: np.ndarray = None
    _panel: pd.DataFrame = None
    _objects_label: pd.DataFrame = None

    @property
    def img(self) -> np.ndarray:
        _img = self._img
        if _img is None:
            _img = tiff.imread(self.img_path)
            self._img = _img if self.in_memory else None
        return _img

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
    def object_labels(self) -> pd.DataFrame:
        _objects_label = self._objects_label
        if _objects_label is None:
            _objects_label = pd.read_csv(self.objects_label_path)
            self._objects_label = _objects_label if self.in_memory else None
        return _objects_label

    @computed_field
    @property
    def height(self) -> int:
        return self.img.shape[-2]

    @computed_field
    @property
    def width(self) -> int:
        return self.img.shape[-1]

    def get_channels(self, channel_names: list[str]) -> np.ndarray:
        idcs = [self.channel_names.index(cn) for cn in channel_names]
        img = self.img[idcs, ...]
        return img


class SegmentedImages(BaseDataset):

    def __init__(self,
                 processed_dir: str | Path,
                 img_version: str = '',
                 mask_version: str = '',
                 channel_names: list[str] = None,
                 in_memory: bool = False
                 ):

        super().__init__(processed_dir=processed_dir)

        self._data: list[SegmentedImage]

        self.in_memory = in_memory
        self.configs_dir: Path = self.processed_dir / "configs"

        self.imgs_base_dir: Path = self.processed_dir / "imgs"
        self.img_version: str = img_version
        self.imgs_dir: Path = self.imgs_base_dir / self.img_version

        self.masks_base_dir: Path = self.processed_dir / "masks"
        self.mask_version: str = mask_version
        self.masks_dir: Path = self.masks_base_dir / self.mask_version

        self.panel_path: Path = self.imgs_dir / 'panel.csv'
        self.panel: pd.DataFrame = None

        self.object_labels_path = self.masks_dir / 'object_labels.parquet'
        self.object_labels: pd.DataFrame = None

        self.channel_names = channel_names or []
        self.channel_indices: list[int] = None

        self.mcd_metadata_dir = self.processed_dir / "mcd_metadata"
        self.mcd_metadata: pd.DataFrame = None

        self.sample_names: list[Path] = None

    def load(self):
        # TODO: should and how should we support subsetself.channel_indices = [self.panel.name.tolist().index(cn) for
        #  cn in self.channel_names]ting the channels?
        self.panel = pd.read_csv(self.panel_path)
        self.object_labels = pd.read_parquet(self.object_labels_path)
        self.channel_indices = [self.panel.name.tolist().index(cn) for cn in self.channel_names]

        if self.mcd_metadata_dir.exists():
            self.mcd_metadata = self.mcd_metadata_dir.glob('**/*.txt')
            self.mcd_metadata = pd.concat([pd.read_csv(f) for f in self.mcd_metadata_dir.glob('**/*.txt')])
            self.mcd_metadata = self.mcd_metadata.assign(
                sample_name=self.mcd_metadata['file_name_napari'].str.replace('.tiff', '')
            )

        img_filenames = set([f.stem for f in self.imgs_dir.glob('*.tiff')])
        mask_filenames = set([f.stem for f in self.masks_dir.glob('*.tiff')])
        self.sample_names = sorted(img_filenames & mask_filenames)

        images = []
        for sample_name in self.sample_names:
            images.append(
                SegmentedImage(
                    sample_name=sample_name,
                    img_path=self.imgs_dir / f"{sample_name}.tiff",
                    masks_path=self.masks_dir / f"{sample_name}.tiff",
                    panel_path=self.panel_path,
                    objects_label_path=self.object_labels_path,
                    in_memory=self.in_memory
                ))
        return images

    def __getitem__(self, idx) -> SegmentedImage:
        return super().__getitem__(idx)

    def __iter__(self) -> SegmentedImage:
        for item in super().__iter__():
            yield item
