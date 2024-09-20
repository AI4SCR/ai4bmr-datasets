import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from pydantic import BaseModel, ConfigDict, Field, computed_field


class Image(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | int = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None

    data_path: str | Path
    metadata_path: str | Path

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
        if _metadata is None:
            _metadata = pd.read_parquet(self.metadata_path)
            self._metadata = _metadata if self.in_memory else None
        return _metadata


class Mask(Image):
    pass