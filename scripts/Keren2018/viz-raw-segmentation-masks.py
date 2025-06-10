from ai4bmr_datasets import Keren2018
from pathlib import Path
from tifffile import imread
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Keren2018')
image_version = mask_version = metadata_version = feature_version = 'published'

ds = Keren2018(base_dir=base_dir)
sample_id = '10'
mask_path = ds.raw_dir / f'TNBC_shareCellData/p{sample_id}_labeledcellData.tiff'
mask = imread(mask_path)
segm_interior_path = ds.raw_dir / f'tnbc/TA459_multipleCores2_Run-4_Point{sample_id}/segmentation_interior.png'
segm_interior = Image.open(segm_interior_path)
segm_interior = np.array(segm_interior)

segm_path = ds.raw_dir / f'tnbc/TA459_multipleCores2_Run-4_Point{sample_id}/segmentation.png'
segm = Image.open(segm_path)
segm = np.array(segm)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(mask)
axs[0].set_title('Mask')
axs[1].imshow(segm_interior)
axs[1].set_title('Segmentation Interior')
axs[2].imshow(segm)
axs[2].set_title('Segmentation')
fig.show()