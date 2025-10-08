import utils.patch_qc as patch_qc

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#
metadata_path = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/metadata.parquet')
coords_path = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/embeddings/uni_v2/coords.parquet')

#%%
metadata = pd.read_parquet(metadata_path)
coords = pd.read_parquet(coords_path)
print(f"Initial coords shape: {coords.shape}")

idx_cols = coords.index.names
coords = coords.reset_index()
coords = coords.merge(metadata, on='block_id', how='left')
coords.set_index(idx_cols, inplace=True)
assert coords.index.is_unique, "Coordinates index is not unique."


#%%
patch_features = patch_qc.extract_patch_features(coords, base_dir=None, name_col=None)

coords = coords.join(patch_features, how='left')
## no NAs in patch_feature columns
assert not coords[patch_features.columns].isna().any().any(), "There are NaNs in the patch features after joining."
assert coords.index.is_unique, "Coordinates index is not unique after joining features."
print(f"Final patch features shape: {coords.shape}")





#########################
#%%
patch_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/trident/patches/20x_512px_0px_overlap/patches')
feat_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/trident/patch_features/features_conch_v15')

list_patches = sorted(patch_dir.glob('*.h5'))
### remove _patches.h5 suffix
list_patches = [p.stem.replace('_patches', '') for p in list_patches]
print(f"Number of patches found: {len(list_patches)}")
# %%
list_feat = sorted(feat_dir.glob('*.h5'))
list_feat = [p.stem for p in list_feat]
print(f"Number of feature files found: {len(list_feat)}")
# %%
diff = set(list_patches).difference(set(list_feat))
print(f"Number of patches without corresponding feature files: {len(diff)}")
print(diff)

metadata_missing = metadata[metadata['block_id'].isin(diff)]
missing = metadata_missing['sample_id'].tolist()
print(f"Number of unique block_ids missing: {len(missing)}")
# %%
path = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/missing_files_conch.txt"
with open(path, 'w') as f:
    for item in missing:
        f.write("%s\n" % item)
# %%
