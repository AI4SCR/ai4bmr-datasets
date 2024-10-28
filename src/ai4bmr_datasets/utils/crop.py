import numpy as np
import pandas as pd
from skimage.measure import regionprops


def get_crops_from_image(self,
                         image: np.ndarray, masks: np.ndarray,
                         sample_name: str,
                         objects_label: pd.DataFrame):
    assert objects_label.index.names == ['sample_name', 'object_id']
    assert ['label_name', 'label_id'] in objects_label.columns

    masks = masks.squeeze()
    # NOTE: for now we only support 2D masks
    assert masks.ndim == 2, f'Expected 2D masks, got {masks.ndim}'
    assert image.shape[1:] == masks.shape

    crops = []
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

        object_id = region.label
        labels = objects_label = objects_label.loc[(sample_name, object_id), :]
        crop = Crop(image=image[..., min_row:max_row, min_col:max_col],
                    masks=masks[..., min_row:max_row, min_col:max_col],
                    object_id=object_id,
                    mask=masks[..., min_row:max_row, min_col:max_col] == object_id,
                    label_name=labels.label_name,
                    label_id=labels.label_id,
                    )
        crops.append(crop)
