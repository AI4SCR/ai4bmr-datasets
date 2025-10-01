### goal: create anndata from attributes of derm dataset object
#%%
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad



#%%
def create_anndata(X, obs=None,
                   var=None,
                   uns=None,
                   obsm=None, **kwargs):
    
    ## check input
    assert X.shape[0] == obs.shape[0], "Number of patches in X and obs must match"
    if var is not None:
        assert X.shape[1] == var.shape[0], "Number of features in X and var must match"
    else:
        var = pd.DataFrame(index=X.columns if isinstance(X, pd.DataFrame) else [f"feat_{i}" for i in range(X.shape[1])])
    

    ## if X and obs have index, align them
    if isinstance(X, pd.DataFrame) and isinstance(obs, pd.DataFrame):
        X, obs = X.align(obs, axis=0, join='inner')
        print(f"Aligned X and obs, new shape: {X.shape}")

    X = X.values if isinstance(X, pd.DataFrame) else X
    index_cols = obs.index.names if isinstance(obs.index, pd.MultiIndex) else ['index']
    obs = obs.reset_index(drop=False)

    

    adata = ad.AnnData(X=X, obs=obs, var=var, uns=uns, obsm=obsm, **kwargs)
    if {'x', 'y'}.issubset(obs.columns):
        spatial = obs[['x', 'y']+index_cols]
        spatial.index = spatial.index.astype(str)
        adata.obsm['spatial'] = spatial


    return adata

#%%
def add_metadata(adata, metadata: None, on=None, how='left', suffix='_meta'):

    obs = adata.obs.copy()
    if metadata is None or metadata.empty:
        ## look if key metadat or clinical is in uns
        if 'metadata' in adata.uns:
            metadata = adata.uns['metadata']
        elif 'clinical' in adata.uns:
            metadata = adata.uns['clinical']
        else:
            print("No metadata provided and no metadata found in adata.uns")
            return adata
    metadata.reset_index(drop=False, inplace=True)
    obs = obs.merge(metadata, how=how, left_on=on, right_on=on, suffixes=('', suffix))
   
    adata.obs = obs
    return adata


### do umap on anndata
### do clustering on anndata
### extract colormaps from anndata
### plot spatial anndata



    

if __name__ == "__main__":
    #from dataset.DermCHUV import DermCHUV
    self = DermCHUV()
    self.data_dir = Path("/users/mensmeng/workspace/dermatology/Visium_HD_Dermato/Visium_HD_Dermato/he_analysis/test_cohort/data/uni_v2_20x_256px_0ov")
    self.setup()

    #%%
    X = self.feats
    obs = self.coords
    uns = {'clinical': self.clinical}

    #%%
    