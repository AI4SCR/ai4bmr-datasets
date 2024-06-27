from skimage.measure import regionprops

from .IMC import IMC
from ..data_models.ImageCrop import ImageCrop


class CropCollectionView:

    def __init__(self,
                 dataset: IMC,
                 padding: int = 0,
                 use_centroid: bool = False,
                 in_memory: bool = False,
                 # NOTE: we move this to transformations
                 # remove_signal_outside_mask: bool = True
                 ):

        self.dataset = dataset
        self.padding = padding
        self.use_centroid = use_centroid
        self.in_memory = in_memory
        # self.remove_signal_outside_mask = remove_signal_outside_mask

        self.crops = []

    def setup(self) -> 'CropCollectionView':
        for img in self.dataset:
            masks = img.masks.copy().squeeze()
            # NOTE: for now we only support 2D masks
            assert masks.ndim == 2, f'Expected 2D masks, got {masks.ndim}'
            # img = img[self.channel_indices, ...] if self.channel_indices else img
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

                crop = ImageCrop(img=img, label=region.label, bbox=bbox, in_memory=self.in_memory)
                self.crops.append(crop)
        return self

    def __getitem__(self, idx):
        return self.crops[idx]

    def get_sample_crops(self, sample_name) -> list:
        return list(filter(lambda x: x.img.sample_name == sample_name, self.crops))

    def __len__(self):
        return len(self.crops)

    def __iter__(self) -> ImageCrop:
        for idx in range(len(self)):
            yield self[idx]