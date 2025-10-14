import json
import subprocess
import textwrap
from pathlib import Path


# from dotenv import load_dotenv
# load_dotenv('/users/mensmeng/workspace/dermatology/derm-pipeline/.env')


import geopandas as gpd
import numpy as np
import openslide
import pandas as pd
import torch


#%%
class BEAT:
    INVALID_SAMPLE_IDS = []
        # '1GUG.0M.2',  # barcode
        # '1GVQ.06.2',  # barcode
        # '1GUG.0M.0',  # mpp_x != mpp_y
        # '1GVQ.06.1',  # mpp_x != mpp_y
    
    PATH_TO_TRIDENT = Path("/users/mensmeng/workspace/dermatology/TRIDENT_pipeline/new_install/trident/trident")

    def __init__(self, base_dir: Path | None = None, id_col: str = 'sample_id'):

        self.base_dir = base_dir or Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned')
        self.raw_dir = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/HnE/BEAT-MESO_rescan_40x_preXenium')
        self.processed_dir = self.base_dir / '02_processed'
        self.dataset_dir = self.processed_dir / 'datasets' / 'beat_hne' / 'trident'
        print(f"Dataset directory: {self.dataset_dir}")

        self.id_col = id_col

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        # GLOBAL PARAMS
        # self.target_mpp = 0.8624999831542972
        self.target_mpp = 0.2211

        self.feature_path = None
        self.coords_path = None
        self.metadata_path = None
        self.anndata_path = None

        import sys
        sys.path.append("/users/mensmeng/workspace/ai4bmr-datasets/src/ai4bmr_datasets")


    def prepare_metadata(self):
        def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
            df = df[df['tech_id'] == 'HE']
            df = df[df['include'] == True]

            ## remove columns where all values are NA
            df = df.dropna(axis=1, how='all')
            assert df.index.is_unique
            assert df.sample_path.is_unique
            assert df.sample_path.isna().sum() == 0
            return df
        


        clinical = pd.read_csv('/users/mensmeng/workspace/beat_meso/metadata_beat_meso_final_rescanned.csv')
        clinical = filter_metadata(clinical)
        clinical = clinical.set_index('sample_id')

        save_path = self.dataset_dir / 'metadata.parquet'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        clinical.to_parquet(save_path, engine='fastparquet')
        print(f"Saved metadata to {save_path}")
        self.metadata_path = save_path

    def get_metadata(self):
        return pd.read_parquet(self.dataset_dir / 'metadata.parquet', engine='fastparquet')

    def get_sample_ids(self):
        clinical = self.get_metadata()
        
        sample_ids = set(clinical.index.tolist()) - self.INVALID_SAMPLE_IDS
        return sample_ids

    def get_wsi_path(self, sample_id: str):
        clinical = self.get_metadata()
        assert sample_id in clinical.index, f"{sample_id} not in metadata index."
        wsi_path = clinical[clinical.index == sample_id].sample_path.item()
        assert Path(wsi_path).exists()
        return wsi_path
    
    def get_wsi_name(self, sample_id: str, id_col: str = 'sample_id'):
        clinical = self.get_metadata()
        assert sample_id in clinical.index, f"{sample_id} not in metadata index."
        if id_col == 'sample_id':
            wsi_name = sample_id
        else:
            assert id_col in clinical.columns, f"{id_col} not in metadata columns."
            wsi_name = clinical[clinical.index == sample_id][id_col].item()
        return wsi_name

    def get_mpp_and_resolution(self, slide: openslide.OpenSlide):
        # MPP
        # MPP (may be missing for some vendors)
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 'nan'))
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 'nan'))
        assert not np.isnan(mpp_x) or not np.isnan(mpp_y), 'MPP is missing'
        assert np.isclose(mpp_x, mpp_y), f'mpp_x={mpp_x} != mpp_y={mpp_y}'

        # Magnification (nominal objective power, if available)
        magnification = slide.properties.get("openslide.objective-power", "NA")

        return mpp_x, magnification

    def check_wsi(self, sample_id: str):

        if sample_id in self.INVALID_SAMPLE_IDS:
            return
        
        wsi_path = self.get_wsi_path(sample_id=sample_id)

        try:
            slide = openslide.OpenSlide(wsi_path)
            mpp_x, mag = self.get_mpp_and_resolution(slide)
            print(f'{sample_id} has mpp={mpp_x:.4f} at {mag}x')
        except Exception as e:
            print(f'Error opening {wsi_path} for {sample_id}: {e}')
            self.INVALID_SAMPLE_IDS.add(sample_id)
            return
        

    def prepare_wsi_trident(self, sample_id: str,
                            seg_model: str = 'hest',
                            target_mag: int = 20,
                            patch_size: int = 256,
                            overlap: int = 0,
                            patch_extractor: str = 'uni_v2',
                            slide_extractor: str | None = None):

        wsi_path = self.get_wsi_path(sample_id=sample_id)
        wsi_name = self.get_wsi_name(sample_id=sample_id, id_col=self.id_col)
        ### call job script and pass parameters

        ## check wsi
        self.check_wsi(sample_id=sample_id)


        ###### SEGMENTATION ###############
        from utils.wsi.segment import segment_wsi
        seg_dir = self.dataset_dir / 'segmentation' 
        seg_dir.mkdir(parents=True, exist_ok=True)
        segment_wsi(wsi_path=wsi_path, wsi_name=wsi_name, seg_model=seg_model, seg_dir=seg_dir, force_rerun=True)

        print(f'Segmentation saved to {seg_dir / "contours_geojson" / f"{wsi_name}.geojson"}')

        ######### PATCHING #################
        from utils.wsi.coords import patch_slide

        # overlap = 0
        # patch_size = 256
        # target_mag = 20
        # patch_extractor = 'uni_v2'

        patch_dir = self.dataset_dir / 'patches' / f"{target_mag}x_{patch_size}px_{overlap}px_overlap"
        patch_dir.mkdir(parents=True, exist_ok=True)

        ## run patching
        patch_slide(wsi_name, wsi_path, seg_dir, patch_dir, target_mag=target_mag, patch_size=patch_size, overlap=overlap, min_tissue_proportion=0)

        print(f'Patches saved to {patch_dir / "patches" / f"{wsi_name}_patches.h5"}')

        ###### FEATURE EXTRACTION ###############
        feature_dir = self.dataset_dir / 'patch_features' / f"features_{patch_extractor}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        ### run feature extraction 
        from utils.wsi.feats import extract_patch_features
        extract_patch_features(wsi_path=wsi_path, wsi_name=wsi_name, patch_dir=patch_dir, feature_dir=feature_dir,
                               feature_extractor=patch_extractor, target_mag=target_mag, patch_size=patch_size, overlap=overlap)
        
        print(f'Patch features saved to {feature_dir / f"{wsi_name}.h5"}')


        ## run slide extraction if slide extractor is not None
        ######### SLIDE FEATURE EXTRACTION #################
        if slide_extractor is not None:
            slide_feature_dir = self.dataset_dir / 'slide_features' / f"slide_features_{slide_extractor}"
            slide_feature_dir.mkdir(parents=True, exist_ok=True)

            from utils.wsi.feats import extract_slide_features
            extract_slide_features(wsi_path=wsi_path, wsi_name=wsi_name, feature_dir=feature_dir, slide_feature_dir=slide_feature_dir,
                                   feature_extractor=patch_extractor, slide_extractor=slide_extractor, target_mag=target_mag, patch_size=patch_size, overlap=overlap)

        print(f'Finished processing {sample_id} ({wsi_name})')



        ### separate segmentation
    def segment(self, sample_id: str, seg_dir, seg_model: str = 'hest'):
        wsi_path = self.get_wsi_path(sample_id=sample_id)
        wsi_name = self.get_wsi_name(sample_id=sample_id, id_col=self.id_col)
        ### call job script and pass parameters

        ## check wsi
        self.check_wsi(sample_id=sample_id)


        ###### SEGMENTATION ###############
        from utils.wsi.segment import segment_wsi
        seg_dir.mkdir(parents=True, exist_ok=True)
        segment_wsi(wsi_path=wsi_path, wsi_name=wsi_name, seg_model=seg_model, seg_dir=seg_dir, force_rerun=True)

        print(f'Segmentation saved to {seg_dir / "contours_geojson" / f"{wsi_name}.geojson"}')


    def filter_contours(self):
        # TODO: potentially post-process contours or improve filtering during segmentation.
        # NOTE: we only use contours that are larger than the median area of all found contours
        # q = np.quantile(contours.geometry.area, 0.5)
        # filter_ = contours.geometry.area > q
        # import matplotlib.pyplot as plt
        # plt.imshow(visualize_contours(contours=contours[filter_], slide=slide)).figure.show()
        pass
    

    ### legacy code, for custom patching
    def create_coords(self, sample_id: str, seg_dir, patch_dir, target_mag: int = 20, patch_size: int = 256, overlap: int = 0):
        wsi_path = self.get_wsi_path(sample_id=sample_id)
        wsi_name = self.get_wsi_name(sample_id=sample_id, id_col=self.id_col)
        ### call job script and pass parameters

        ## check wsi
        self.check_wsi(sample_id=sample_id)

        patch_dir = self.dataset_dir / 'patches' / f"{target_mag}x_{patch_size}px_{overlap}px_overlap"
        patch_dir.mkdir(parents=True, exist_ok=True)

        ## run patching
        from utils.wsi.coords import patch_slide
        patch_slide(wsi_name, wsi_path, seg_dir, patch_dir, target_mag=target_mag, patch_size=patch_size, overlap=overlap, min_tissue_proportion=0)

        print(f'Patches saved to {patch_dir / "patches" / f"{wsi_name}_patches.h5"}')
        

    ### legacy code, for custom patch embedding
    def create_patch_embeddings(self, sample_id: str, patch_dir, feature_dir, patch_extractor: str = 'uni_v2',
                                 target_mag: int = 20, patch_size: int = 256, overlap: int = 0):

        wsi_path = self.get_wsi_path(sample_id=sample_id)
        wsi_name = self.get_wsi_name(sample_id=sample_id, id_col=self.id_col)
        ### call job script and pass parameters

        ## check wsi
        self.check_wsi(sample_id=sample_id)

        feature_dir.mkdir(parents=True, exist_ok=True)

        ### run feature extraction 
        from utils.wsi.feats import extract_patch_features
        extract_patch_features(wsi_path=wsi_path, wsi_name=wsi_name, patch_dir=patch_dir, feature_dir=feature_dir,
                               feature_extractor=patch_extractor, target_mag=target_mag, patch_size=patch_size, overlap=overlap)
        
        print(f'Patch features saved to {feature_dir / f"{wsi_name}.h5"}')

        
        
        

    ### legacy code, for custom slide embedding
    def create_slide_embedding(self, sample_id: str, patch_feature_dir: str, slide_feature_dir: str, patch_extractor: str = 'uni_v1',
                                 slide_extractor: str | None = None, target_mag: int = 20, patch_size: int = 256, overlap: int = 0):


        wsi_path = self.get_wsi_path(sample_id=sample_id)
        wsi_name = self.get_wsi_name(sample_id=sample_id, id_col=self.id_col)
        ### call job script and pass parameters

        ## check wsi
        self.check_wsi(sample_id=sample_id)


        Path(slide_feature_dir).mkdir(parents=True, exist_ok=True)

        from utils.wsi.feats import extract_slide_features
        extract_slide_features(wsi_path=wsi_path, wsi_name=wsi_name, feature_dir=patch_feature_dir, slide_feature_dir=slide_feature_dir,
                                feature_extractor=patch_extractor, slide_extractor=slide_extractor, target_mag=target_mag, patch_size=patch_size, overlap=overlap)
        print(f'Extracted features for {sample_id} ({wsi_name})')


        
    
        
    
    
    
    ### legacy code, for custom slide embedding
    def create_coords_data(self, sample_id: str, model_name: str, coords_version: str, max_patches: int = 1000):
        logger.info(f'Computing UMAP for {sample_id}')

        fname = f'model={model_name}-{coords_version}.png'
        save_path = self.dataset_dir / sample_id / 'visualizations' / 'umaps' / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            return

        data = self.get_sample_embeddings(sample_id=sample_id,
                                          model_name=model_name,
                                          coords_version=coords_version,
                                          max_patches=max_patches)

        if len(data) <= 15:
            logger.error(f"{sample_id} has only {len(data)} patches. Can't compute UMAP.")
            return

        ax = plot_umap(data=data)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300)

    def create_patch_feature_data(self, model_name: str, coords_version: str, max_patches_per_slide: int = 100):
        logger.info(f'Computing UMAP for patches')

        fname = f'model={model_name}-{coords_version}-max_patches_per_slide={max_patches_per_slide}.png'
        save_path = self.base_dir / '04_outputs' / 'umaps' / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            logger.info(f'UMAP already exists.')
            return

        data = []
        labels = []
        sample_ids = self.get_sample_ids()
        for sample_id in tqdm(sample_ids):
            tmp = self.get_sample_embeddings(sample_id=sample_id,
                                       model_name=model_name,
                                       coords_version=coords_version,
                                       max_patches=max_patches_per_slide)
            data.append(tmp)
            labels.extend([sample_id] * len(tmp))

        data = torch.concat(data)
        labels = np.array(labels)

        ax = plot_umap(data=data, labels=labels, show_legend=False)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300)

    def create_slide_feature_data(self):

        embeddings_dir: Path = Path(
            '/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/wsi_embed_subset/20x_512px_0px_overlap/slide_features_titan')
        save_path = embeddings_dir.parent / 'umap-slides.png'

        data = []
        for path in sorted(embeddings_dir.glob('*.pt')):
            data.append(torch.load(path))
        data = torch.cat(data).cpu()

        ax = plot_umap(data=data)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=300)

    def add_stats_to_metadata(self):

        sample_dirs = sorted([i for i in self.dataset_dir.iterdir() if i.is_dir()])
        stats = []
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name

            if sample_id in self.INVALID_SAMPLE_IDS:
                continue

            slide = OpenSlide(sample_dir / 'wsi.tiff')
            width, height = slide.dimensions
            mpp = get_mpp(slide=slide)
            record = {'sample_id': sample_id, 'width': width, 'height': height, 'mpp': mpp}

            for patch_size in [448]:
                with open(
                        sample_dir / 'coords' / f'patch_size={patch_size}-stride={patch_size}-mpp=0.8625-overlap=0.25.json') as f:
                    coords = json.load(f)
                    record[f'num_coords_patch_size={patch_size}'] = len(coords)

            stats.append(record)

        stats = pd.DataFrame(stats)
        stats = stats.set_index('sample_id')

        metadata_path = self.dataset_dir / 'metadata.parquet'
        metadata = pd.read_parquet(metadata_path)
        metadata = pd.concat([metadata, stats], axis=1)
        metadata.to_parquet(metadata_path)

    ############ DATA PREPARATION ##############

    def prepare_data(self, save_dir, feats_dir, coords_dir, force_rerun=False):
        from utils.data_prep.data import features_to_df, coords_to_df

        df_features = features_to_df(feats_dir, id_col=self.id_col)
        df_coords = coords_to_df(coords_dir, id_col=self.id_col)

        df_coords, df_features = df_coords.align(df_features, join='inner', axis=0)
        print(f"Coordinates shape: {df_coords.shape}, Features shape: {df_features.shape}")

        
        save_dir.mkdir(parents=True, exist_ok=True)
        ## check if file exists
        if (save_dir / 'feats.parquet').exists() and not force_rerun:
            print(f"Features file already exists at {save_dir / 'feats.parquet'}. Skipping.")
        else:
            df_features.to_parquet(save_dir / 'feats.parquet', engine='fastparquet')
            print(f"Saved features to {save_dir / 'feats.parquet'}")
            self.feature_path = save_dir / 'feats.parquet'
        if (save_dir / 'coords.parquet').exists() and not force_rerun:
            print(f"Coordinates file already exists at {save_dir / 'coords.parquet'}. Skipping.")
        else:
            df_coords.to_parquet(save_dir / 'coords.parquet', engine='fastparquet')
            print(f"Saved coordinates to {save_dir / 'coords.parquet'}")
            self.coords_path = save_dir / 'coords.parquet'
    
        return df_coords, df_features

    def update_coords(self, id_col='sample_id'):
        metadata = self.get_metadata()
        if self.coords_path is None:
            raise ValueError("Coordinates path is not set. Please run prepare_data first.")
        coords = pd.read_parquet(self.coords_path, engine='fastparquet')

        print(f"Initial coords shape: {coords.shape}")

        idx_cols = coords.index.names
        coords = coords.reset_index()
        assert id_col in coords.columns, f"{id_col} not in coordinates columns."
        assert id_col in metadata.index.names, f"{id_col} not in metadata index."
        coords = coords.merge(metadata, on=id_col, how='left')
        coords.set_index(idx_cols, inplace=True)
        assert coords.index.is_unique, "Coordinates index is not unique."


        
        import utils.qc.patch_qc as patch_qc
        patch_features = patch_qc.extract_patch_features(coords, base_dir=None, name_col=None)

        coords = coords.join(patch_features, how='left')
        ## no NAs in patch_feature columns
        assert not coords[patch_features.columns].isna().any().any(), "There are NaNs in the patch features after joining."
        assert coords.index.is_unique, "Coordinates index is not unique after joining features."
        print(f"Final patch features shape: {coords.shape}")

        coords_path = self.coords_path.parent / f"{self.coords_path.stem}_processed.parquet"
        coords.to_parquet(coords_path, engine='fastparquet')
        print(f"Saved updated coordinates to {coords_path}")
        self.coords_path = coords_path
        return coords
    
    def construct_anndata(self, anndata_dir: Path | None = None):
        if self.feature_path is None or self.coords_path is None:
            raise ValueError("Feature path or Coordinates path is not set. Please run prepare_data first.")
        
        from utils.data_prep.anndata_interface import create_anndata
        df_features = pd.read_parquet(self.feature_path, engine='fastparquet')
        df_coords = pd.read_parquet(self.coords_path, engine='fastparquet')


        adata = create_anndata(X=df_features, obs=df_coords)
        metadata = self.get_metadata()
        object_cols = metadata.select_dtypes(include=['object']).columns
        for col in object_cols:
            metadata[col] = metadata[col].astype(str)
        adata.uns['metadata'] = metadata
        
        
        if anndata_dir is not None:
            anndata_dir.parent.mkdir(parents=True, exist_ok=True)
            anndata_path = anndata_dir / 'anndata.h5ad'
            adata.write_h5ad(anndata_path)
            print(f"Saved AnnData to {anndata_path}")
            self.anndata_path = anndata_path

        return adata


#%%
if __name__ == '__main__':
    
    #%%
    self = BEAT(id_col='block_id')
    #%%
    #self.prepare_metadata()

    #%%
    metadata = self.get_metadata()
    sample_ids = metadata.index.tolist()

    ### save sample_ids to txt file
    # with open(self.processed_dir / 'sample_ids.txt', 'w') as f:
    #     for sample_id in sample_ids:
    #         f.write(f"{sample_id}\n")
    #%%
    sample_id = sample_ids[100]


    #%%
    seg_dir = self.dataset_dir / 'segmentation'
    self.segment(sample_id=sample_id, seg_dir=seg_dir, seg_model='hest')

    #%%
    target_mag = 20
    patch_size = 512
    overlap = 0
    patch_extractor = 'conch_v15'

    patch_dir = self.dataset_dir / 'patches' / f"{target_mag}x_{patch_size}px_{overlap}px_overlap"
    feature_dir = self.dataset_dir / 'patch_features' / f"features_{patch_extractor}"


    #%%
    # self.create_coords(sample_id=sample_id, seg_dir=seg_dir, patch_dir=patch_dir,
    #                    target_mag=target_mag, patch_size=patch_size, overlap=overlap)

    # #%%
    # self.create_patch_embeddings(sample_id=sample_id, patch_dir=patch_dir, feature_dir=feature_dir,
    #                              patch_extractor=patch_extractor, target_mag=target_mag, patch_size=patch_size, overlap=overlap)

    # %%
    fm_model = 'resnet50'
    save_dir = self.processed_dir / 'embeddings' / 'resnet50'
    coords_dir = feats_dir = Path(f'/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/datasets/beat_hne/trident/patch_features/features_{fm_model}')

    #%%
    df_coords, df_features = self.prepare_data(save_dir=save_dir, feats_dir=feats_dir, coords_dir=coords_dir, force_rerun=True)


    # %%
    coords_path = save_dir / 'coords.parquet'
    self.coords_path = coords_path
    feature_path = save_dir / 'feats.parquet'
    self.feature_path = feature_path


    #%%
    df_coords = self.update_coords(id_col='block_id')


    #%%
    anndata_dir = self.processed_dir / 'embeddings' / 'uni_v2'
    adata = self.construct_anndata(anndata_dir=anndata_dir)


# %%
## open anndata from h5ad
# import anndata as ad
# adata = ad.read_h5ad(self.anndata_path)