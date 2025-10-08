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
import scanpy as sc
import seaborn as sns
import anndata as ad

#%%
fm_model = 'conch_v15'
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
if downsample:
    from utils.data_prep.data import downsample_per_id   

    feats = downsample_per_id(feats, id_col='block_id', frac=0.2, random_state=42)
    # %%
    coords = coords.loc[feats.index]
assert feats.index.equals(coords.index)


#%%
# ### read anndata
# import anndata as ad
# anndata_path = results_dir / fm_model / "anndata.h5ad"
# adata = ad.read_h5ad(anndata_path)

from utils.data_prep.anndata_interface import create_anndata, add_metadata
adata = create_anndata(X=feats, obs=coords)
if 'block_id' not in adata.obs.columns:
    adata.obs['block_id'] = adata.obs['sample_id']
    adata.obs.drop(columns=['sample_id'], inplace=True)
adata = add_metadata(adata, metadata=metadata, on='block_id', how='left', suffix='_meta')
# %%
## scale, pca, umap
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)

#%%
sc.tl.umap(adata)
## save a temporary copy
#%%
# temp_adata_path = results_dir / fm_model / "anndata_temp.h5ad"
# adata.obs.index = adata.obs.index.astype(str)
# adata.obsm['spatial'].index = adata.obsm['spatial'].index.astype(str)
#adata.write_h5ad(temp_adata_path)


# %%

adata.obs['pat_id'] = adata.obs['pat_id'].astype('category')
fig = sc.pl.umap(adata, color=['pat_id'], frameon=False, legend_fontsize=8, return_fig=True, palette='tab20')
fig_path = results_dir / fm_model / "plots" / 'UMAP_by_pat_id.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, bbox_inches='tight', dpi=300)

# %%
#resolution = 1.0
for resolution in [0.5, 1.0]:
    cluster_name = f'leiden_{resolution}'
    if cluster_name in adata.obs.columns:
        print(f"{cluster_name} already exists in adata.obs")
    else:
        sc.tl.leiden(adata, resolution=resolution, flavor='igraph', key_added=cluster_name, n_iterations=-1)

    fig = sc.pl.umap(adata, color=[cluster_name],frameon=False, legend_loc='on data', legend_fontsize=8, return_fig=True)
    fig_path = results_dir / fm_model / "plots" / f'UMAP_by_{cluster_name}.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)

    ############### isolated patch visualization ###############
    from utils.plotting.patch_vis import get_palette 
    annotation_col = 'pat_id'
    annotation_color_dict = get_palette(adata, col=annotation_col)
    print(annotation_color_dict)
    cluster_col = cluster_name
    color_dict = get_palette(adata, col=cluster_col)
    n_patches = 15
    #%%
    df_coords = adata.obs.copy()
    clusters = df_coords[cluster_col].unique()
    sorted_clusters = sorted(clusters.astype(int))
    level = 1
    patch_size = df_coords.iloc[0]['patch_size']

    from utils.plotting.patch_vis import plot_multiple_patches, extract_multiple_patches, get_border_colors
    figures = []
    for cluster in tqdm(sorted_clusters):
        ## transfer back to string
        cluster = str(cluster)
        #print(f"Cluster {cluster}")
        df_subset = df_coords[df_coords[cluster_col] == cluster].sample(n_patches)
        patches = extract_multiple_patches(df_subset, level=level, patch_size=patch_size)
        print(f"Cluster {cluster} - {len(patches)}  patches")
        #print(df_subset[annotation_col].cat.categories)

        
        cols_section = get_border_colors(df_subset, annotation_col, color_dict=annotation_color_dict)
        names_section = df_subset[annotation_col].values

        fig, _ = plot_multiple_patches(patches, border_colors=cols_section,subtitles=names_section, title=f'Section ID - Cluster {cluster}')
        figures.append(fig)
        plt.close(fig)  # Close the figure to free memory

    nrows = len(figures)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 2*nrows))

    #Draw existing figures into new subplots
    for ax, f in zip(axes, figures):
        ## draw at high resolution  
        canvas = FigureCanvasAgg(f)
        canvas.draw()
        img = np.array(canvas.buffer_rgba())  # Convert figure to image
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    ## Save the figure
    fig_path = results_dir / fm_model / "plots" / f'Patches_{cluster_col}_by_{annotation_col}.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)

# %%
import pickle 
temp_adata_path = results_dir / fm_model / "anndata_test.pkl"
with open(temp_adata_path, 'wb') as f:
    pickle.dump(adata, f)

with open(temp_adata_path, 'rb') as f:
    adata = pickle.load(f)


# %%
### make plot where i count the number of ids in sample col where each cluster is present
df_patches = adata.obs.copy()
cluster_col = 'leiden_1.0'
sample_col = 'block_id'
color_dict = get_palette(adata, col=cluster_col)
cluster_sample_counts = df_patches.groupby(cluster_col)[sample_col].nunique().reset_index()
cluster_sample_counts.columns = [cluster_col, 'num_samples']
# %%
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(cluster_sample_counts[cluster_col], cluster_sample_counts['num_samples'], color=[color_dict[c] for c in cluster_sample_counts[cluster_col]])
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of Unique Samples')
ax.set_title('Number of Unique Samples per Cluster')
ax.set_xticks(cluster_sample_counts[cluster_col])
ax.set_xticklabels(cluster_sample_counts[cluster_col], rotation=45) 

### same as percentage of total samples
total_samples = df_patches[sample_col].nunique()
for bar in bars:
    height = bar.get_height()
    percentage = (height / total_samples) * 100
    ax.annotate(f'{percentage:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=8)
    
fig_path = results_dir / fm_model / "plots" / f'Num_Samples_per_{cluster_col}.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, bbox_inches='tight', dpi=300)
# %%
## average proportion of patch composition per sample
df_patches = adata.obs.copy()
cluster_col = 'leiden_1.0'
sample_col = 'block_id'
# Calculate the proportion of each cluster in each sample
proportions = df_patches.groupby([sample_col, cluster_col]).size().unstack(fill_value=0)
proportions = proportions.div(proportions.sum(axis=1), axis=0)


#%%
avg_proportions = proportions.mean().reset_index()
avg_proportions.columns = [cluster_col, 'avg_proportion']
### plot as pie chart
fig, ax = plt.subplots(figsize=(12, 12))
ax.pie(avg_proportions['avg_proportion'], labels=avg_proportions[cluster_col], autopct='%1.1f%%', startangle=140, colors=[color_dict[c] for c in avg_proportions[cluster_col]])
ax.set_title('Average Proportion of Each Cluster Across All Samples')
# %%
fig_path = results_dir / fm_model / "plots" / f'Average_Proportion_{cluster_col}.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, bbox_inches='tight', dpi=300)




# %%
df_patches = adata.obs.copy()
sample_col = 'block_id'
sections = df_patches[sample_col].unique()


#%%
from utils.plotting.patch_vis import visualize_clusters, get_palette
patch_size = df_patches.iloc[0]['patch_size']
cluster_col = 'leiden_1.0'
level0_magnification = df_patches.iloc[0]['level0_magnification']
target_magnification = df_patches.iloc[0]['target_magnification']
color_dict = get_palette(adata, col=cluster_col)

save_dir = results_dir / fm_model / "plots" / f"visualiation_test_{cluster_col}"
save_dir.mkdir(parents=True, exist_ok=True)
#%%
section = "1H5K_0M"
#%%
for section in sections:
    print(f'Processing {section}')

    df_section = df_patches[df_patches[sample_col] == section]
    df_section = df_section.filter(['x', 'y', cluster_col, 'sample_path', sample_col], axis=1)

    wsi_path = df_section['sample_path'].values[0]


    visualize_clusters(df_coords=df_section, 
                    wsi_path = wsi_path,
                    name_col=sample_col,
                    cluster_col=cluster_col,
                mag_scr=level0_magnification, mag_target=target_magnification, patch_size=patch_size,
                plot_prefix=section, color_dict=color_dict, result_dir=save_dir, size_thumbnail=10000)
    
    print(f'Finished {section}')

