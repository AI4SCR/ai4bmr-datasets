from pathlib import Path

import pandas as pd
from pydantic import ConfigDict

from .BaseDataset import BaseDataset
from ..data_models.Dataset import Dataset
from ..data_models.Image import Image


class IMC(BaseDataset):
    model_config = Dataset.model_config | ConfigDict(extra='allow')
    _id = 'imc'
    _name = 'imc'
    _data = None  # we do not keep data in memory

    def __init__(self, base_dir: Path,
                 img_version: str = '', mask_version: str = '', channel_names: list[str] = None,
                 **kwargs):

        raw_dir = base_dir / "raw"
        processed_dir = base_dir

        super().__init__(base_dir=base_dir, raw_dir=raw_dir, processed_dir=processed_dir, **kwargs)

        self.configs_dir: Path = self.processed_dir / "configs"

        self.imgs_base_dir: Path = self.processed_dir / "imgs"
        self.img_version: str = img_version
        self.imgs_dir: Path = self.imgs_base_dir / self.img_version

        self.panel_path: Path = self.imgs_dir / 'panel.csv'
        self.panel: pd.DataFrame = None
        self.channel_names = channel_names or []
        self.channel_indices: list[int] = None

        self.masks_base_dir: Path = self.processed_dir / "masks"
        self.mask_version: str = mask_version
        self.masks_dir: Path = self.masks_base_dir / self.mask_version

        self.mcd_metadata_dir = self.processed_dir / "mcd_metadata"
        self.mcd_metadata: pd.DataFrame = None

        self.sample_names: list[Path] = None

    def load(self):
        # TODO: should and how should we support subsetself.channel_indices = [self.panel.name.tolist().index(cn) for
        #  cn in self.channel_names]ting the channels?
        self.panel = pd.read_csv(self.panel_path)
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

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx) -> Image:
        sample_name = self.sample_names[idx]
        return Image(
            sample_name=sample_name,
            img_path=self.imgs_dir / f"{sample_name}.tiff",
            masks_path=self.masks_dir / f"{sample_name}.tiff",
            panel_path=self.panel_path
        )

    def __iter__(self) -> Image:
        for idx in range(len(self)):
            yield self[idx]


