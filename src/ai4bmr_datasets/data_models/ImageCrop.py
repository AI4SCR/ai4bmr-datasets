from .Image import Image

import uuid
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field


class ImageCrop(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    img: Image
    label: int
    bbox: tuple[int, int, int, int]

    in_memory: bool = False
    _crop: np.ndarray = None
    _masks: np.ndarray = None

    @property
    def crop(self) -> np.ndarray:
        min_row, min_col, max_row, max_col = self.bbox
        _crop = self._crop
        if self._crop is None:
            _crop = self.img.img[..., min_row:max_row, min_col:max_col]
            self._crop = _crop if self.in_memory else None
        return _crop

    @property
    def masks(self) -> np.ndarray:
        min_row, min_col, max_row, max_col = self.bbox
        _masks = self._masks
        if self._masks is None:
            _masks = self.img.masks[..., min_row:max_row, min_col:max_col]
            self._crop = _masks if self.in_memory else None
        return _masks

    @property
    def mask(self) -> np.ndarray:
        _mask = self.masks.copy()
        return _mask == self.label

    @computed_field
    @property
    def height(self) -> int:
        # NOTE: we need to sync bbox with Resizes and Crops
        min_row, min_col, max_row, max_col = self.bbox
        return max_row - min_row

    @computed_field
    @property
    def width(self) -> int:
        # NOTE: we need to sync bbox with Resizes and Crops
        min_row, min_col, max_row, max_col = self.bbox
        return max_col - min_col

