from ai4bmr_datasets import Keren2018
from pathlib import Path
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt

save_dir = Path.cwd() / 'cell-masks-differences'
save_dir.mkdir(exist_ok=True, parents=True)

base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Keren2018')
image_version = mask_version = metadata_version = feature_version = 'published'

ds_published = Keren2018(base_dir=base_dir)
ds_published.setup(image_version='published', mask_version='published')
ds_annotated = Keren2018(base_dir=base_dir)
ds_annotated.setup(image_version='published', mask_version='annotated')


for sample_id in ds_published.sample_ids:

    if sample_id not in ds_published.masks:
        continue

    mask_raw_path = ds_published.raw_dir / f'tnbc_processed_data/p{sample_id}_labeledcellData.tiff'
    mask_raw = imread(mask_raw_path)
    mask_published = ds_published.masks[sample_id].data
    mask_annotated = ds_annotated.masks[sample_id].data

    objs_published = set(mask_published.flat)
    objs_annotated = set(mask_annotated.flat)

    intx = objs_published.intersection(objs_annotated)
    diff = objs_published - objs_annotated
    assert objs_annotated <= objs_published

    fig, axs = plt.subplots(2, 2, dpi=400)
    axs = axs.flat

    axs[0].imshow(mask_raw > 0, interpolation=None, cmap="grey")
    axs[0].set_title("Mask Raw", fontsize=8)

    axs[1].imshow(mask_published > 0, interpolation=None, cmap="grey")
    axs[1].set_title("Mask published", fontsize=8)

    axs[2].imshow(mask_annotated > 0, interpolation=None, cmap="grey")
    axs[2].set_title("Mask annotated", fontsize=8)

    mask_diff = mask_published.copy()
    filter_ = np.isin(mask_published, list(objs_annotated))
    filter_.sum()
    mask_diff[filter_] = 0
    axs[3].imshow(mask_diff > 0, interpolation=None)
    axs[3].set_title("Difference published - annotated", fontsize=8)

    for ax in axs:
        ax.set_axis_off()

    fig.tight_layout()
    # fig.show()
    fig.savefig(save_dir / f"{sample_id}.png")
    plt.close(fig)