from ai4bmr_datasets.datasets.Cords2024 import Cords2024
from pathlib import Path
import pandas as pd

image_version='published'
mask_version='published'
metadata_version='published'
ds = self = Cords2024(base_dir=Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024'))
ds.setup(image_version='published', mask_version='published', metadata_version=metadata_version, load_metadata=True)

ds.metadata
len(ds.sample_ids)
ds.images['175_B_110'].data.shape

# ds.download()
# ds.process_acquisitions()
# ds.create_images()
# samples = pd.read_parquet(ds.sample_metadata_path)

import tifffile
p = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024/01_raw/cell_masks_zenodo/Cell masks/178_B_mask/mask/20210112_LC_NSCLC_TMA_176_B_s0_a100_ac_ilastik_s2_Probabilitiescells_mask.tiff'
img = tifffile.imread(p)

p2 = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024/01_raw/acquisitions/178_B_100.tiff'
img2 = tifffile.imread(p2)

p3 = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024/01_raw/acquisitions/176_B_100.tiff'
img3 = tifffile.imread(p3)


from pathlib import Path
import shutil

mcd_dir = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024/01_raw/mcd_dir")

for mcd_file in list(mcd_dir.parent.glob("*.mcd")):
    target_dir = mcd_dir / mcd_file.name  # removes .mcd extension
    target_dir.mkdir(exist_ok=True)
    shutil.move(str(mcd_file), target_dir / mcd_file.name)

ds.panel

def is_sequential(sequence):
    for i in range(1, len(sequence) - 1):
        if not sequence[i] == sequence[i - 1] + 1:
            return False
    return True

is_sequential(list(uniq_mask_objs))
