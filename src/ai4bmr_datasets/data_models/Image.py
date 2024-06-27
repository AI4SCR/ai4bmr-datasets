import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from pydantic import BaseModel, ConfigDict, Field, computed_field


class Image(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_name: str

    img_path: str | Path
    masks_path: str | Path
    panel_path: str | Path

    in_memory: bool = False
    _img: np.ndarray = None
    _masks: np.ndarray = None
    _panel: pd.DataFrame = None

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
