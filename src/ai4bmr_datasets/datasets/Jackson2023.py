import json
from pathlib import Path

import pandas as pd
from ai4bmr_core.utils.tidy import tidy_name
from loguru import logger
from tifffile import imread
from tifffile import imwrite

from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset
from ai4bmr_datasets.utils.download import unzip_recursive
import re
from ai4bmr_core.utils.tidy import tidy_name, relabel_duplicates

base_dir = Path("/users/amarti51/prometex/data/datasets/Jackson2023")
ds = self = Jackson2023(base_dir=base_dir)
ds.create_masks()


# ds.download()

class Jackson2023(BaseIMCDataset):

    @property
    def id(self):
        return "Jackson2023"

    @property
    def name(self):
        return "Jackson2023"

    @property
    def doi(self):
        return "10.1038/s41586-019-1876-x"

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self.base_dir = base_dir

        # populated by `self.load()`
        self.sample_ids = None
        self.samples = None
        self.images = None
        self.masks = None
        self.panel = None
        self.intensity = None
        self.spatial = None

    def prepare_data(self):
        self.download()

        self.create_clinical_metadata()

        self.create_panel()
        self.create_images()
        self.create_masks()

    def download(self, force: bool = False):
        import requests
        import shutil

        download_dir = self.raw_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "https://zenodo.org/records/4607374/files/OMEandSingleCellMasks.zip?download=1": download_dir / 'ome_and_single_cell_masks.zip',
            "https://zenodo.org/records/4607374/files/singlecell_locations.zip?download=1": download_dir / 'single_cell_locations.zip',
            "https://zenodo.org/records/4607374/files/singlecell_cluster_labels.zip?download=1": download_dir / 'single_cell_cluster_labels.zip',
            "https://zenodo.org/records/4607374/files/SingleCell_and_Metadata.zip?download=1": download_dir / 'single_cell_and_metadata.zip',
            "https://zenodo.org/records/4607374/files/TumorStroma_masks.zip?download=1": download_dir / 'tumor_stroma_masks.zip',
        }

        # Download files
        for url, target_path in file_map.items():
            if not target_path.exists() or force:
                logger.info(f"Downloading {url} → {target_path}")
                headers = {"User-Agent": "Mozilla/5.0"}
                with requests.get(url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    with open(target_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                logger.info(f"Skipping download of {target_path.name}, already exists.")

        # Extract zip files
        for target_path in file_map.values():
            unzip_recursive(target_path)

        shutil.rmtree(self.raw_dir / '__MACOSX', ignore_errors=True)

        logger.info("✅ Download and extraction completed.")

    def get_sample_id_from_path(self, path):
        path = str(path)
        search = re.search(r"TMA_(\d+)_(\w+)_s0_a(\d+)", path)
        return '_'.join(search.groups())

    def create_images(self):
        panel = pd.read_parquet(self.get_panel_path('published'))
        clinical = pd.read_parquet(self.clinical_metadata_path)

        raw_images_dir = self.raw_dir / 'ome_and_single_cell_masks/OMEnMasks/ome/ome/'

        save_dir = self.images_dir / self.get_version_name(version='published')
        save_dir.mkdir(parents=True, exist_ok=True)

        for (_, row) in clinical.iterrows():
            img_path = raw_images_dir / row.file_name_full_stack

            if img_path.exists():

                save_path = save_dir / f"{row.sample_id}.tiff"
                if save_path.exists():
                    logger.warning(f"Image file {save_path} already exists. Skipping.")
                    continue

                img = imread(img_path)
                img = img[panel.page.values]

                imwrite(save_path, img)
            else:
                logger.warning(f"Image file {img_path} does not exist. Skipping.")

    def create_masks(self):
        clinical = pd.read_parquet(self.clinical_metadata_path)

        raw_masks_dir = self.raw_dir / 'ome_and_single_cell_masks/OMEnMasks/Basel_Zuri_masks/Basel_Zuri_masks'

        save_dir = self.masks_dir / self.get_version_name(version='published')
        save_dir.mkdir(parents=True, exist_ok=True)

        for (_, row) in clinical.iterrows():
            mask_path = raw_masks_dir / row.file_name_full_stack
            mask_path = str(mask_path).replace('.tiff', '_maks.tiff')

            if Path(mask_path).exists():
                save_path = save_dir / f"{row.sample_id}.tiff"

                if save_path.exists():
                    logger.warning(f"Mask file {save_path} already exists. Skipping.")
                    continue

                mask = imread(mask_path)

                imwrite(save_path, mask)
            else:
                logger.warning(f"Mask file {mask_path} does not exist. Skipping.")

    def create_metadata(self):
        from ai4bmr_core.utils.tidy import tidy_name, relabel_duplicates
        clinical = pd.read_parquet(self.clinical_metadata_path)

        bs_meta_clusters = pd.read_csv(
            self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/Basel_metaclusters.csv')
        zh_meta_clusters = pd.read_csv(
            self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/Zurich_matched_metaclusters.csv')

        metacluster_anno = pd.read_csv(
            self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/Metacluster_annotations.csv', sep=';')

        pg_bs = pd.read_csv(self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/PG_basel.csv')
        pg_zh = pd.read_csv(self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/PG_zurich.csv')
        # filter_ = pg_zh.core.str.startswith('ZTMA208_slide_non-breast')
        # pg_zh = pg_zh[~filter_]

        # sample_ids = pg_zh.core.str.replace('ZTMA208_slide_', '').map(tidy_name)
        # pg_zh = pg_zh.rename(columns=dict(CellId='object_id'))
        # pg_zh['sample_id'] = sample_ids
        # pg_zh = pg_zh.set_index(['sample_id', 'object_id'])
        # assert pg_zh.index.is_unique

        zh = pg_zh.merge(zh_meta_clusters, on="id", how="outer")
        bs = pg_bs.merge(bs_meta_clusters, on="id", how="outer")

        # TODO: can we just drop? Where to those cells come from? -> some for example from ZTMA208_slide_non-breast*
        zh = zh[zh.cluster.notna()]
        zh = zh.set_index(['sample_id', 'object_id'])
        assert zh.index.is_unique

        bs = bs[bs.cluster.notna()]
        bs = bs.rename(columns={"PhenoGraphBasel": "PhenoGraph"})
        bs = bs.set_index(['sample_id', 'object_id'])
        # filter_ = bs.index.duplicated(keep=False)
        # bs[filter_].sort_index()

        assert bs.index.is_unique
        assert set(zh.id).intersection(set(bs.id)) == set()

        metadata = pd.concat([zh, bs])
        assert metadata.index.is_unique
        assert metadata.isna().any().any() == False

        metacluster_anno.columns = metacluster_anno.columns.map(tidy_name)
        metadata = metadata.merge(
            metacluster_anno, left_on="cluster", right_on="metacluster", how="left"
        )
        metadata = metadata.rename(columns={
            'CellId': 'object_id',
            'class': 'label0',
            'cell_type': 'label'
        })
        assert (metadata.CellId.astype(str) != metadata.object_id).any() == False

        metadata = metadata.drop(columns=['CellId', 'PhenoGraph', 'id', 'metacluster'])
        metadata.columns = metadata.columns.map(tidy_name)
        metadata = metadata.set_index(['sample_id', 'object_id'])
        assert metadata.index.is_unique

        metacluster_anno.loc[:, 'label'] = metacluster_anno['cell_type'].str.replace('+', '_pos').map(tidy_name)
        metacluster_anno.loc[:, 'label0'] = metacluster_anno['class'].map(tidy_name)

        label_dict = metacluster_anno.set_index('metacluster')['label'].to_dict()
        label0_dict = metacluster_anno.set_index('metacluster')['label0'].to_dict()

        metadata['label'] = metadata['pheno_graph'].map(label_dict)
        metadata['label0'] = metadata['pheno_graph'].map(label0_dict)

        version_name = self.get_version_name(version='published')
        save_dir = self.metadata_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in metadata.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_clinical_metadata(self):
        from ai4bmr_core.utils.tidy import tidy_name

        bs = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/BaselTMA/Basel_PatientMetadata.csv')
        zh = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv')

        bs = bs.rename(columns={'Post_surgeryTx': 'Post_surgeryTx_other'})

        bs.columns = bs.columns.map(tidy_name)
        zh.columns = zh.columns.map(tidy_name)

        zh["cohort"] = "zurich"
        bs["cohort"] = "basel"

        col_names_map = {
            # 'pr': 'pr_status',
            'prstatus': 'pr_status',
            # 'er': 'er_status',
            'erstatus': 'er_status',
            'hr': 'hr_status',
            'hrstatus': 'hr_status',
        }
        zh = zh.rename(columns=col_names_map)
        bs = bs.rename(columns=col_names_map)

        val_map = {
            '+': 'positive',
            '-': 'negative',
            'nan': pd.NA,
            'NaN': pd.NA,
            # '[]': pd.NA
        }
        zh = zh.map(lambda x: val_map.get(x, x))
        bs = bs.map(lambda x: val_map.get(x, x))

        z = set(zh.columns)
        b = set(bs.columns)
        cols = z.intersection(b)
        z - b
        b - z

        metadata_cols = {
            "FileName_FullStack",
            "Height_FullStack",
            "TMABlocklabel",
            "TMALocation",
            "TMAxlocation",
            "Tag",
            "Width_FullStack",
            "Yearofsamplecollection",
            "yLocation",
            "AllSamplesSVSp4.Array",
            "Comment",
        }

        metadata = pd.concat([bs, zh], ignore_index=True)
        assert (metadata.core.value_counts() == 1).all()

        metadata = metadata.convert_dtypes()

        metadata = metadata.astype(str)
        sample_ids = metadata.core.str.strip().map(tidy_name)
        sample_ids = sample_ids.str.replace('basel_tma_sp', '')
        sample_ids = sample_ids.str.replace('ztma208_slide_', '')
        metadata['sample_id'] = sample_ids
        metadata.set_index('sample_id')

        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path, engine='fastparquet')

    def create_features_spatial(self):
        pass

    def create_features_intensity(self):

        bs = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/BaselTMA/SC_dat.csv')
        zh = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/ZurichTMA/SC_dat.csv')

        # TODO from here

        path = self.raw_dir / 'intensity.parquet'
        metadata = pd.read_parquet(path)

        ids = metadata['object_id'].str.split('_')
        sample_id = ids.str[:-1].str.join('_')
        object_id = ids.str[-1]

        metadata['sample_id'] = sample_id
        metadata['object_id'] = object_id

        metadata = metadata.convert_dtypes()
        metadata = metadata.set_index(['sample_id', 'object_id'])

        version_name = self.get_version_name(version='published')
        save_dir = self.intensity_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in metadata.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_features(self):
        self.create_features_spatial()
        self.create_features_intensity()

    def create_panel(self):
        # TODO: double check panel, I remember there was a problem with naming of one of the channels
        panel = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/Basel_Zuri_StainingPanel.csv')

        panel = panel.dropna(axis=0, how='all')
        panel['page'] = panel.FullStack.astype(int) - 1
        panel['target'] = panel.Target.map(tidy_name)

        filter_ = panel.target.isin(['ruthenium_tetroxide', 'argon_dimers'])
        panel = panel[~filter_]

        panel = relabel_duplicates(panel, 'target')
        assert len(panel) == 43

        panel = panel.convert_dtypes()
        panel = panel.reset_index(drop=True)
        panel.index.name = "channel_index"

        panel_path = self.get_panel_path("published")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)
