from ai4bmr_datasets.datasets.TNBC import TNBC
from pathlib import Path
base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/TNBC')
ds = TNBC(base_dir=base_dir)
ds.prepare_data()
ds.setup(image_version='published', mask_version='published', metadata_version='published', load_metadata=True)
