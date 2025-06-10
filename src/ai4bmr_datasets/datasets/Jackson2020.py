from pathlib import Path

import pandas as pd
from loguru import logger
from tifffile import imread, imwrite

from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset
from ai4bmr_datasets.utils.download import unzip_recursive
from ai4bmr_core.utils.tidy import tidy_name

class Jackson2020(BaseIMCDataset):
    name = "Jackson2020"
    id = "Jackson2020"
    doi = "10.1038/s41586-019-1876-x"

    def __init__(self, base_dir: Path | None = None):
        super().__init__(base_dir)

    def prepare_data(self):
        self.download()

        self.create_clinical_metadata()

        self.create_panel()
        self.create_images()
        self.create_masks()
        self.create_features()

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

    def create_images(self):
        panel = pd.read_parquet(self.get_panel_path('published'))
        clinical = pd.read_parquet(self.clinical_metadata_path)

        raw_images_dir = self.raw_dir / 'ome_and_single_cell_masks/OMEnMasks/ome/ome/'

        save_dir = self.images_dir / self.get_version_name(version='published')
        save_dir.mkdir(parents=True, exist_ok=True)

        for (sample_id, row) in clinical.iterrows():
            img_path = raw_images_dir / row.file_name_full_stack

            if img_path.exists():

                save_path = save_dir / f"{sample_id}.tiff"
                if save_path.exists():
                    logger.info(f"Image file {save_path} already exists. Skipping.")
                    continue
                else:
                    logger.info(f"Creating image {save_path}")

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

        for (sample_id, row) in clinical.iterrows():
            mask_path = raw_masks_dir / row.file_name_full_stack
            mask_path = str(mask_path).replace('.tiff', '_maks.tiff')

            if Path(mask_path).exists():
                save_path = save_dir / f"{sample_id}.tiff"

                if save_path.exists():
                    logger.info(f"Mask file {save_path} already exists. Skipping.")
                    continue
                else:
                    logger.info(f"Creating mask {save_path}")

                mask = imread(mask_path)

                imwrite(save_path, mask)
            else:
                logger.warning(f"Mask file {mask_path} does not exist. Skipping.")

    def create_metadata(self):
        bs_meta_clusters = pd.read_csv(
            self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/Basel_metaclusters.csv')
        zh_meta_clusters = pd.read_csv(
            self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/Zurich_matched_metaclusters.csv')

        metacluster_anno = pd.read_csv(
            self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/Metacluster_annotations.csv', sep=';')

        pg_bs = pd.read_csv(self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/PG_basel.csv')
        pg_zh = pd.read_csv(self.raw_dir / 'single_cell_cluster_labels/Cluster_labels/PG_zurich.csv')

        # zh_sample_ids = set(zh_meta_clusters.id.str.split('_').str[:-1].str.join('_'))
        # bs_sample_ids = set(bs_meta_clusters.id.str.split('_').str[:-1].str.join('_'))

        zh = pg_zh.merge(zh_meta_clusters, on="id", how="outer")
        bs = pg_bs.merge(bs_meta_clusters, on="id", how="outer")

        # note: these are the number of annotated samples
        #   the following samples are not annotated:
        #       BaselTMA_SP43_Liver_X2Y1    4433
        #       BaselTMA_SP43_Liver_X1Y1    2510
        #       BaselTMA_SP42_Liver_X2Y1    2129
        #       BaselTMA_SP41_Liver_X2Y1    1914
        #       BaselTMA_SP41_Liver_X1Y1     184
        #       ZTMA208_slide_control_Cy15x6      2951
        #       ZTMA208_slide_12_Ay14x1           2709
        #       ZTMA208_slide_51_Ay7x1            1790
        #       ZTMA208_slide_64_Ay3x1            1569
        #       ZTMA208_slide_3_Ay10x1            1495
        #       ZTMA208_slide_17_Ay8x1            1347
        #       ZTMA208_slide_control_Cy15x7      1279
        #       ZTMA208_slide_55_Ay12x1           1248
        #       ZTMA208_slide_5_Ay1x1             1156
        #       ZTMA208_slide_61_Ay9x1            1142
        #       ZTMA208_slide_69_Ay5x1            1111
        #       ZTMA208_slide_42_Ay2x1            1081
        #       ZTMA208_slide_26_Ay4x1             939
        #       ZTMA208_slide_control_Cy15x8       772
        #       ZTMA208_slide_non-breast_Dy1x0     547
        #       ZTMA208_slide_39_Ay16x1            206
        #       ZTMA208_slide_non-breast_Dy3x0     205
        #       ZTMA208_slide_non-breast_Dy2x0     107
        #       ZTMA208_slide_1_Ay13x1              28
        #       ZTMA208_slide_20_Ay6x1              24
        #       ZTMA208_slide_43_Ay11x1             11
        zh = zh[zh.cluster.notna()]
        zh = zh.set_index(['id'])
        assert zh.index.is_unique

        bs = bs[bs.cluster.notna()]
        bs = bs.rename(columns={"PhenoGraphBasel": "PhenoGraph"})
        bs = bs.set_index('id')
        assert bs.index.is_unique
        assert set(zh.index).intersection(set(bs.index)) == set()

        metadata = pd.concat([zh, bs])
        assert metadata.index.is_unique
        assert metadata.isna().any().any() == False
        metadata = metadata.reset_index()

        metacluster_anno.columns = metacluster_anno.columns.map(tidy_name)
        metadata = metadata.merge(
            metacluster_anno, left_on="cluster", right_on="metacluster", how="left"
        )

        metadata = metadata.reset_index(drop=True).drop(columns=['PhenoGraph'])
        metadata = self.add_sample_id_and_object_id(metadata)
        assert metadata.index.is_unique
        metadata = metadata.drop(columns=['id', 'core', 'metacluster'])

        metadata = metadata.rename(columns={
            'class': 'label0',
            'cell_type': 'label'
        })
        metadata.loc[:, 'label'] = metadata['label'].str.replace('+', '_pos').map(tidy_name)
        metadata.loc[:, 'label0'] = metadata['label0'].map(tidy_name)
        metadata.columns = metadata.columns.map(tidy_name)

        version_name = self.get_version_name(version='published')
        save_dir = self.metadata_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in metadata.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def add_sample_id_and_object_id(self, data):
        sample_ids = data.core.str.replace('ZTMA208_slide_', '')
        sample_ids = sample_ids.str.replace('BaselTMA_SP', '')

        data['sample_id'] = sample_ids.map(tidy_name)
        data = data.convert_dtypes()
        data = data.set_index(['sample_id'])

        if 'CellId' in data.columns:
            data = data.rename(columns={'CellId': 'object_id'})
            data = data.set_index(['object_id'], append=True)

        return data

    def create_clinical_metadata(self):

        bs = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/BaselTMA/Basel_PatientMetadata.csv')
        zh = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv')

        bs = bs.rename(columns={'Post_surgeryTx': 'Post_surgeryTx_2'})

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

        # z = set(zh.columns)
        # b = set(bs.columns)
        # cols = z.intersection(b)
        # z - b
        # b - z

        metadata_cols = [
            # "FileName_FullStack",  # we keep this to track the images easily
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
        ]
        metadata_cols = [tidy_name(i) for i in metadata_cols]

        metadata = pd.concat([bs, zh], ignore_index=True)
        assert (metadata.core.value_counts() == 1).all()

        metadata = metadata.drop(columns=metadata_cols)

        metadata = self.add_sample_id_and_object_id(metadata)

        metadata = metadata.convert_dtypes()
        for col in metadata.select_dtypes(include=['object', 'string']).columns:
            if col == 'file_name_full_stack':
                continue
            metadata.loc[:, col] = metadata[col].map(lambda x: tidy_name(str(x)) if not pd.isna(x) else x)

        metadata = metadata.convert_dtypes()
        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path, engine='fastparquet')

    def create_features(self):
        logger.info(f'Creating `published` features')

        panel = pd.read_parquet(self.get_panel_path('published'))
        bs = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/BaselTMA/SC_dat.csv')
        zh = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/ZurichTMA/SC_dat.csv')

        spatial_feat = [
            "Area",
            "Eccentricity",
            "EulerNumber",
            "Extent",
            "MajorAxisLength",
            "MinorAxisLength",
            "Number_Neighbors",
            "Orientation",
            "Percent_Touching",
            "Perimeter",
            "Solidity",
        ]
        expr_feat = set(panel.channel_name) - set(spatial_feat)
        assert len(expr_feat) == 35

        feats = expr_feat.union(spatial_feat)
        bs = bs[bs.channel.isin(feats)]
        zh = zh[zh.channel.isin(feats)]

        bs = bs.pivot(index=["CellId", 'core'], columns="channel", values="mc_counts").reset_index()
        bs = self.add_sample_id_and_object_id(bs)
        bs = bs.drop(columns=['core'])
        assert bs.index.is_unique

        zh = zh.pivot(index=["CellId", 'core'], columns="channel", values="mc_counts").reset_index()
        zh = self.add_sample_id_and_object_id(zh)
        zh = zh.drop(columns=['core'])
        assert zh.index.is_unique

        x = pd.concat([bs, zh])
        assert x.isna().sum().sum() == 0

        intensity = x.loc[:, x.columns.isin(expr_feat)]
        mapping = panel.set_index('channel_name')['target'].to_dict()
        intensity = intensity.rename(columns=mapping)
        intensity = intensity[panel.target]
        assert not intensity.columns.isna().any()
        assert intensity.columns.value_counts().unique().item() == 1

        spatial = x.loc[:, x.columns.isin(spatial_feat)]
        spatial.columns = spatial.columns.map(tidy_name)

        version_name = self.get_version_name(version='published')

        # intensity
        save_dir = self.intensity_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)
        for grp_name, grp_dat in intensity.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

        # spatial
        save_dir = self.spatial_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)
        for grp_name, grp_dat in spatial.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_panel(self):
        panel = pd.read_csv(self.raw_dir / 'single_cell_and_metadata/Data_publication/Basel_Zuri_StainingPanel.csv')

        panel = panel.dropna(axis=0, how='all')
        panel['page'] = panel.FullStack.astype(int) - 1
        panel['target'] = panel.Target.map(tidy_name)
        panel = panel.assign(target_original = panel.Target).drop(columns=['Target'])
        panel.columns = panel.columns.map(tidy_name)

        filter_ = panel.target.isin(['ruthenium_tetroxide', 'argon_dimers'])
        panel = panel[~filter_]

        panel = panel.rename(columns={'metal_tag': 'metal_mass'})
        panel.loc[panel.metal_mass == 'Er167', 'target'] = 'e_cadherin_p_cadherin'
        panel.loc[panel.metal_mass == 'La139', 'target'] = 'histone_h3_trimethylate'
        panel.loc[panel.metal_mass == 'Eu153', 'target'] = 'histone_h3_phospho'
        assert len(panel) == 43

        # NOTE: {'phospho Erk12', '1971527Ho165Di bCaten', '483739Yb171Di Sox9', '2971330Dy161Di EpCAM'}
        #   have only been measured in the Zurich cohort
        metal_mass_to_channel_name = {
            "In113": "1261726In113Di Histone",
            "La139": "473968La139Di Histone",
            "Pr141": "651779Pr141Di Cytoker",
            "Nd142": "3281668Nd142Di Fibrone",
            "Nd143": "3111576Nd143Di Cytoker",
            "Nd144": "971099Nd144Di Cytoker",
            "Nd145": "Nd145Di Twist",
            "Nd146": "77877Nd146Di CD68",
            "Sm147": "346876Sm147Di Keratin",
            "Nd148": "174864Nd148Di SMA",
            "Sm149": "1921755Sm149Di Vimenti",
            "Nd150": "322787Nd150Di cMyc",
            "Eu151": "201487Eu151Di cerbB",
            "Sm152": "8001752Sm152Di CD3epsi",
            "Eu153": "phospho Histone",
            "Sm154": "phospho Erk12",
            "Gd155": "3521227Gd155Di Slug",
            "Gd156": "112475Gd156Di Estroge",
            "Gd158": "312878Gd158Di Progest",
            "Tb159": "207736Tb159Di p53",
            "Gd160": "6967Gd160Di CD44",
            "Dy161": "2971330Dy161Di EpCAM",
            "Dy162": "71790Dy162Di CD45",
            "Dy163": "117792Dy163Di GATA3",
            "Dy164": "361077Dy164Di CD20",
            "Ho165": "1971527Ho165Di bCaten",
            "Er166": "92964Er166Di Carboni",
            "Er167": "1031747Er167Di ECadhe",
            "Er168": "1441101Er168Di Ki67",
            "Tm169": "1021522Tm169Di EGFR",
            "Er170": "phospho S6",
            "Yb171": "483739Yb171Di Sox9",
            "Yb172": "378871Yb172Di vWF",
            "Yb173": "phospho mTOR",
            "Yb174": "98922Yb174Di Cytoker",
            "Lu175": "234832Lu175Di panCyto",
            "Yb176": "198883Yb176Di cleaved",
            "Ir191": "10331253Ir191Di Iridium",
            "Ir193": "10331254Ir193Di Iridium",
        }

        panel['channel_name'] = panel.metal_mass.map(metal_mass_to_channel_name)
        filter_ = panel.metal_mass.duplicated()
        alternative_targets_dict = panel[filter_].set_index('metal_mass').target.to_dict()
        del alternative_targets_dict['Gd158']
        panel['target_alternative'] = panel.metal_mass.map(lambda x: alternative_targets_dict.get(x, pd.NA))
        panel = panel[~filter_]
        assert not panel.page.duplicated().any()

        # remove channels that are not in basel cohort
        channels_only_in_zurich = {'phospho Erk12', '1971527Ho165Di bCaten', '483739Yb171Di Sox9', '2971330Dy161Di EpCAM'}
        filter_ = panel.channel_name.isin(channels_only_in_zurich)
        panel = panel[~filter_]
        assert len(panel) == 35

        # panel[['target', 'channel_name', 'metal_mass',]]
        # panel[['target', 'target_alternative', 'metal_mass',]]

        panel = panel.convert_dtypes()
        panel = panel.reset_index(drop=True)
        panel.index.name = "channel_index"

        panel_path = self.get_panel_path("published")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)


