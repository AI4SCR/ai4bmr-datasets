#%%
import sys
sys.path.append('/users/mensmeng/workspace/ai4bmr-datasets/src/ai4bmr_datasets/datasets')
from BEAT import *


import argparse

parser = argparse.ArgumentParser(description='Compute BEAT embeddings')
parser.add_argument('--sample_id', type=str, default=None)


args = parser.parse_args()
sample_id = args.sample_id

######## PARAMETERS ##########
#%%
target_mag = 20
patch_size = 512
overlap = 0
patch_extractor = 'conch_v15'

##### WORKFLOW ######

dm = BEAT(id_col = 'block_id')



#%%
seg_dir = dm.dataset_dir / 'segmentation'
#dm.segment(sample_id=sample_id, seg_dir=seg_dir, seg_model='hest')



patch_dir = dm.dataset_dir / 'patches' / f"{target_mag}x_{patch_size}px_{overlap}px_overlap"
feature_dir = dm.dataset_dir / 'patch_features' / f"features_{patch_extractor}"


#%%
dm.create_coords(sample_id=sample_id, seg_dir=seg_dir, patch_dir=patch_dir,
                    target_mag=target_mag, patch_size=patch_size, overlap=overlap)

#%%
dm.create_patch_embeddings(sample_id=sample_id, patch_dir=patch_dir, feature_dir=feature_dir,
                                patch_extractor=patch_extractor, target_mag=target_mag, patch_size=patch_size, overlap=overlap)


# %%
