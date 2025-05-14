import pandas as pd

from ai4bmr_datasets.datasets.BLCa import BLCa
from pathlib import Path

base_dir = Path("/users/amarti51/prometex/data/datasets/BLCa")
ds = self= BLCa(base_dir=base_dir)
ds.prepare_data() # only called if you want to process the raw data
ds.setup(image_version='filtered', mask_version='cleaned', load_intensity=False, load_metadata=True)

ds.metadata
ds.clinical

ds.setup(image_version='filtered', mask_version='cleaned', feature_version='for_clustering', load_intensity=True, load_metadata=False)
ds.compute_features(image_version='filtered', mask_version='cleaned', force=True)

# d1 = pd.read_parquet('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/BLCa/02_processed/features/intensity/for_clustering/240307_001.parquet')
# d2 = pd.read_parquet('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/BLCa/02_processed/features/intensity/filtered-cleaned/240307_001.parquet')
# d1.columns
# d2.columns
