#%% 
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
#from ai4bmr_datasets.utils.data_prep.data import coords_to_df, features_to_df
import sys
sys.path.append('/users/mensmeng/workspace/ai4bmr-datasets/src/ai4bmr_datasets/datasets')
#%%
fm_model = 'uni_v2'
downsample = False
results_dir = Path("/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/embeddings")
coords_file = results_dir / fm_model / "coords.parquet"
feats_file = results_dir / fm_model / "feats.parquet"

# %%
feats = pd.read_parquet(feats_file, engine='fastparquet')
coords = pd.read_parquet(coords_file, engine='fastparquet')

# %%
metadata_path = Path("/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/metadata.parquet")
metadata = pd.read_parquet(metadata_path, engine='fastparquet')
# %%
sys.path.append('/users/mensmeng/workspace/ai4bmr-datasets/src/ai4bmr_datasets/')



# %%
import anndata as ad
import scanpy as sc
temp_adata_path = results_dir / fm_model / "anndata_temp.pkl"
with open(temp_adata_path, 'rb') as f:
    adata = pd.read_pickle(f)
# adata = ad.read_h5ad(temp_adata_path, )
# %%
#################
path = Path("/users/mensmeng/workspace/beat_meso/df.parquet")
df = pd.read_parquet(path, engine='fastparquet')

#%%
ffpe_ids = set(df['SampleID'].unique())
print(f"Number of unique ffpe_ids in df: {len(ffpe_ids)}")
pat_ids = set(adata.obs['block_id'].unique())
print(f"Number of unique pat_ids in adata: {len(pat_ids)}")

print(f"Number of unique pat_ids not in ffpe_ids: {len(pat_ids - ffpe_ids)}")
print(f"pat_ids not in ffpe_ids: {pat_ids - ffpe_ids}")
print(f"Number of unique ffpe_ids not in pat_ids: {len(ffpe_ids - pat_ids)}")
print(f"ffpe_ids not in pat_ids: {ffpe_ids - pat_ids}")

## overwrite ffpe id :
#%%
df['block_id'] = df['SampleID']
df.loc[df['SampleID'] == '1GAY_06', 'block_id'] = '1G4Y_06'
df.loc[df['SampleID'] == '1IWW_05', 'block_id'] = '1IWV_05'
# %%
obs = adata.obs.copy()
obs = obs.merge(df, on='block_id', how='left')
# %%
adata.obs = obs.set_index(adata.obs.index)

# %%
cols = ['histology', 'stage', 'best_response', 'el_trt_arm']
for col in cols:
    if col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype('category')        
        fig = sc.pl.umap(adata, color=[col], frameon=False, legend_fontsize=8, return_fig=True)
        fig_path = results_dir / fm_model / "plots" / f'UMAP_by_{col}.png'
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        #fig.savefig(fig_path, bbox_inches='tight', dpi=300)

# %%
adata.obs['pat_id'] = adata.obs['pat_id'].astype('category')
fig = sc.pl.umap(adata, color=['pat_id'], frameon=False, legend_fontsize=8, return_fig=True, palette='tab20')
# %%
fig = sc.pl.umap(adata, color=['leiden_1.0'], frameon=False, legend_loc='on data', return_fig=True, palette='tab20')

# %%
