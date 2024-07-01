import uuid

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field
from skimage.measure import regionprops

from .BaseDataset import BaseDataset
from .SegmentedImages import SegmentedImage, SegmentedImages


class SegmentedImageCrop(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    source: SegmentedImage
    label: int
    bbox: tuple[int, int, int, int]

    in_memory: bool = False
    _img: np.ndarray = None
    _masks: np.ndarray = None

    @property
    def img(self) -> np.ndarray:
        min_row, min_col, max_row, max_col = self.bbox
        _img = self._img
        if self._img is None:
            _img = self.source.img[..., min_row:max_row, min_col:max_col]
            self._img = _img if self.in_memory else None
        return _img

    @property
    def masks(self) -> np.ndarray:
        min_row, min_col, max_row, max_col = self.bbox
        _masks = self._masks
        if self._masks is None:
            _masks = self.source.masks[..., min_row:max_row, min_col:max_col]
            self._masks = _masks if self.in_memory else None
        return _masks

    @property
    def mask(self) -> np.ndarray[bool]:
        return self.masks == self.label

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


class SegmentedImagesCrops(BaseDataset):

    def __init__(self,
                 dataset: SegmentedImages,
                 padding: int = 0,
                 use_centroid: bool = False,
                 in_memory: bool = False,
                 # NOTE: we move this to transformations
                 # remove_signal_outside_mask: bool = True
                 # **kwargs
                 ):
        kwargs = {}
        raw_dir = kwargs.get('processed_dir', None) or dataset.processed_dir
        super().__init__(raw_dir=raw_dir, **kwargs)

        self.dataset = dataset
        self.padding = padding
        self.use_centroid = use_centroid
        self.in_memory = in_memory
        # self.remove_signal_outside_mask = remove_signal_outside_mask

    def load(self):
        crops = []
        for img in self.dataset:
            masks = img.masks.copy().squeeze()
            # NOTE: for now we only support 2D masks
            assert masks.ndim == 2, f'Expected 2D masks, got {masks.ndim}'
            # source = source[self.channel_indices, ...] if self.channel_indices else source
            props = regionprops(masks)

            H, W = masks.shape
            for region in props:
                if self.use_centroid:
                    min_row, min_col = max_row, max_col = region.centroid
                else:
                    min_row, min_col, max_row, max_col = region.bbox
                min_row, min_col, max_row, max_col = int(min_row), int(min_col), int(max_row), int(max_col)

                min_row, min_col = max(0, min_row - self.padding), max(0, min_col - self.padding)
                max_row, max_col = min(H, max_row + self.padding), min(W, max_col + self.padding)

                bbox = min_row, min_col, max_row, max_col

                crop = SegmentedImageCrop(source=img, label=region.label, bbox=bbox, in_memory=self.in_memory)
                crops.append(crop)
        return crops

    def get_sample_crops(self, sample_name) -> list:
        return list(filter(lambda x: x.source.sample_name == sample_name, self._data))
