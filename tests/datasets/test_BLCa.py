from ai4bmr_datasets.datasets.BLCa import BLCa
from pathlib import Path

base_dir = Path("/users/amarti51/prometex/data/datasets/BLCa")
ds = BLCa(base_dir=base_dir)
ds.prepare_data() # only called if you want to process the raw data
ds.setup(image_version='filtered', mask_version='cleaned', load_intensity=False, load_metadata=True)

ds.metadata
ds.clinical
