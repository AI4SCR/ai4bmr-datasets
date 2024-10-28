import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from pydantic import BaseModel, Field


class Image(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | int = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None

    data_path: str | Path
    metadata_path: str | Path | None

    in_memory: bool = False
    _data: np.ndarray = None
    _metadata: pd.DataFrame = None

    @property
    def data(self) -> np.ndarray:
        _data = self._data
        if _data is None:
            _data = tiff.imread(self.data_path)
            self._data = _data if self.in_memory else None
        return _data

    @property
    def metadata(self) -> pd.DataFrame:
        _metadata = self._metadata
        if _metadata is None and self.metadata_path:
            _metadata = pd.read_parquet(self.metadata_path)
            self._metadata = _metadata if self.in_memory else None
        return _metadata


class Mask(Image):
    pass


class SegmentedImage(BaseModel):
    id: str | int = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None

    image: Image
    mask: Mask


class Crop(SegmentedImage):
    id: str | int = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None

    object_id: int | None = None
    bbox: tuple[int, int, int, int]
