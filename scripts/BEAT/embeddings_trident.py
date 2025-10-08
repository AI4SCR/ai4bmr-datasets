# %%
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

#%%
import sys
sys.path.append("/users/mensmeng/workspace/ai4bmr-datasets/src/ai4bmr_datasets")

# %%
metadata_path = Path("/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/metadata.parquet")
metadata = pd.read_parquet(metadata_path, engine='fastparquet')

### call job script and pass parameters

sample_ids = metadata.index
sample_id = sample_ids[0]



#%%
slide_feature_extractor = 'titan'
patch_extractor = 'conch_v15'
slide_feature_dir = Path(f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/trident/slide_features/features_{slide_feature_extractor}")
patch_feature_dir = Path(f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/trident/patch_features/features_{patch_extractor}")

#%%
target_mag = 20
patch_size = 512
overlap = 0

#%%
for sample_id in tqdm(sample_ids):
    print(f"Processing sample_id: {sample_id}")
    wsi_name = metadata.loc[sample_id, 'block_id']
    wsi_path = metadata.loc[sample_id, 'sample_path']

    from utils.wsi.feats import extract_slide_features
    extract_slide_features(wsi_path=wsi_path, wsi_name=wsi_name, feature_dir=patch_feature_dir, slide_feature_dir=slide_feature_dir,
                        feature_extractor=patch_extractor, slide_extractor=slide_feature_extractor, target_mag=target_mag, patch_size=patch_size, overlap=overlap)
    print(f'Extracted features for {sample_id} ({wsi_name})')

# #%%
# if not (Path(patch_feature_dir) / f"{wsi_name}.h5").exists():
#     print(f"Patch feature file for {wsi_name} does not exist in {patch_feature_dir}. Please run extract_patch_features() first.")

# if (Path(slide_feature_dir) / f"{wsi_name}.h5").exists():
#     print(f"Slide features already extracted to: {slide_feature_dir}")
    
# else:
#     Path(slide_feature_dir).mkdir(parents=True, exist_ok=True)
#     print(f"Slide features will be saved to: {slide_feature_dir}")


# #%%

# slide_encoder = slide_encoder_factory(slide_feature_extractor)
# patch_encoder = slide_to_patch_encoder_name[slide_encoder.enc_name]
# assert patch_encoder == patch_extractor, f"Patch encoder {patch_extractor} does not match the expected patch encoder {patch_encoder} for slide encoder {slide_encoder.enc_name}. Please extract patch features using {patch_encoder} first."

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# slide_encoder.to(device)

# import os
# patch_features_path = os.path.join(patch_feature_dir, f"{wsi_name}.h5")
# print(f"Reading patch features from: {patch_features_path}")



# ##check if the WSI file exists
# assert os.path.exists(wsi_path), f"WSI file {wsi_path} does not exist."

# wsi = OpenSlideWSI(slide_path=wsi_path, lazy_init=False)
# wsi.name = wsi_name


# #%%


# %%
### read embeddings
slide_feature_extractor = 'titan'
slide_feature_dir = Path(f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/trident/slide_features/features_{slide_feature_extractor}")
file_paths = list(slide_feature_dir.glob("*.h5"))
import h5py

dict_features = {}
dict_attributes = {}
for file_path in file_paths:
    print(f"Processing {file_path}...")
    with h5py.File(file_path, 'r') as f:
        f.visit(print)  # List all keys in the HDF5 file
        sample_id = file_path.stem

        attributes = f['coords'].attrs
        # assert attributes['name'] == sample_id, f"Section ID mismatch: {attributes['name']} != {sample_id}"
        dict_attributes[sample_id] = {k: v for k, v in attributes.items()}

        dict_features[sample_id] = f['features'][:]

df_features = pd.DataFrame(dict_features).T
df_attributes = pd.DataFrame(dict_attributes).T


#################
#%%
path = Path("/users/mensmeng/workspace/beat_meso/df.parquet")
df = pd.read_parquet(path, engine='fastparquet')

## overwrite ffpe id :
#%%
df['block_id'] = df['SampleID']
df.loc[df['SampleID'] == '1GAY_06', 'block_id'] = '1G4Y_06'
df.loc[df['SampleID'] == '1IWW_05', 'block_id'] = '1IWV_05'

df_attributes = df_attributes.merge(df, left_on='name', right_on='block_id', how='left')
#%%
### scale, pca, umap
import scanpy as sc
import anndata as ad
adata = ad.AnnData(X=(df_features.values))
adata.obs = df_attributes

#%%
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)    
sc.tl.umap(adata)


# %%
col ='histology'
fig = sc.pl.umap(adata, color=col, return_fig=True, title=col)
col ='block_id'
adata.obs['block_id'] = adata.obs['block_id'].astype('category')
fig = sc.pl.umap(adata, color=col, return_fig=True, title=col, palette='tab20')
# %%
