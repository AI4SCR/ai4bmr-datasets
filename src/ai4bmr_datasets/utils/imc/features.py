import numpy as np
from skimage.measure import regionprops_table
import pandas as pd


# %%
def spatial_features(
    mask: np.ndarray,
    properties: tuple[str] = ("label", "centroid", "eccentricity", "area", "extent"),
):
    rp = regionprops_table(mask, properties=properties)
    feat = pd.DataFrame.from_dict(rp).set_index("label").rename(columns={"centroid-0": "y", "centroid-1": "x"})
    feat.index.name = "object_id"
    return feat


def intensity_features(
    img: np.ndarray,
    mask: np.ndarray,
    panel: pd.DataFrame | None = None,
):
    img = np.moveaxis(img, 0, -1)
    rp = regionprops_table(mask, img, properties=("label", "intensity_mean"))
    feat = pd.DataFrame.from_dict(rp).set_index("label")

    if panel is not None:
        mapping = {f"intensity_mean-{num}": panel.index[num] for num in range(len(panel))}
        feat = feat.rename(mapping, axis=1)

    feat.index.name = "object_id"
    return feat
