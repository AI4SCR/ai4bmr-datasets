from loguru import logger
from ai4bmr_core.utils.tidy import tidy_name
from tifffile import imread, imwrite
from pathlib import Path
import pandas as pd
from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset


class PCa(BaseIMCDataset):

    def __init__(self, base_dir: Path | None = None):
        super().__init__(base_dir)

        # raw paths
        self.raw_clinical_metadata_path = (
                self.base_dir / "01_raw" / "metadata" / "ROI_matching_blockID.xlsx"
        )
        self.raw_tma_annotations_path = (
                self.base_dir / "01_raw" / "metadata" / "tma-annotations-v2.xlsx"
        )

        self.raw_tma_tidy_path = self.base_dir / "01_raw" / "metadata" / "TMA_tidy.txt"
        self.raw_annotations_path = self.base_dir / "01_raw" / "annotations" / "labels.parquet"

    def prepare_data(self):
        self.process_acquisitions()
        self.create_panel()

        self.create_images()
        self.create_filtered_images()

        self.create_masks()
        self.create_filtered_masks()

        self.create_clinical_metadata()
        self.create_tma_annotations()

        self.label_transfer()
        self.create_metadata()

        self.create_annotated()

    def process_acquisitions(self):
        """
        Failed to read acquisitions:
          - 240210_IIIBL_X3Y15_150_1
        """
        from readimc import MCDFile, TXTFile
        from tifffile import imwrite
        import json

        acquisitions_dir = self.raw_dir / 'acquisitions'
        acquisitions_dir.mkdir(parents=True, exist_ok=True)

        num_failed_reads = 0

        mcd_paths = [i for i in (self.raw_dir / 'raw').rglob("*.mcd") if i.is_file()]
        for mcd_path in mcd_paths:
            logger.info(f"Processing {mcd_path.name}")
            slide_date = mcd_path.stem

            with MCDFile(mcd_path) as f:
                for slide in f.slides:
                    for acq in slide.acquisitions:

                        item = dict(
                            id=acq.id,
                            description=acq.description,
                            mcd_name=mcd_path.name,
                            channel_names=acq.channel_names,
                            channel_labels=acq.channel_labels,
                            num_channels=acq.num_channels,
                            metal_names=acq.channel_metals,
                        )

                        image_name = f"{slide_date}_{acq.description}_{acq.id}".lower()
                        image_name = tidy_name(image_name)
                        metadata_path = acquisitions_dir / f"{image_name}.json"
                        image_path = acquisitions_dir / f"{image_name}.tiff"

                        if image_path.exists():
                            logger.info(f"Skipping image {image_path}. Already exists.")
                            continue

                        try:
                            logger.info(f"Reading acquisition of image {image_name}")
                            img = f.read_acquisition(acq)
                            imwrite(image_path, img)
                            with open(metadata_path, "w") as f_json:
                                json.dump(item, f_json)
                        except OSError:
                            logger.warning(f"Failed to read acquisition from mcd file, reading from txt file.")

                            try:
                                image_name = f"{slide_date}_{acq.description}_{acq.id}".lower()
                                txt_path = mcd_path.parent / f'{image_name}.txt'
                                with TXTFile(txt_path) as txt_file:
                                    img = txt_file.read_acquisition()
                                    imwrite(image_path, img)
                                    with open(metadata_path, "w") as f_json:
                                        json.dump(item, f_json)
                            except OSError:
                                logger.error(f"Failed to read acquisition!")
                                num_failed_reads += 1

                            continue

    def create_annotated_masks(self):
        import numpy as np
        from tifffile import imread
        from ai4bmr_core.utils.saving import save_mask
        import pandas as pd

        def filter_mask(mask: np.ndarray, object_ids: set[int]):
            mapping = lambda x: x if x in object_ids else 0
            return np.vectorize(mapping)(mask)

        mask_dir = self.get_mask_version_dir(mask_version="cleaned")
        annotations = pd.read_parquet(self.raw_annotations_path)

        new_mask_dir = self.get_mask_version_dir(mask_version="filtered")
        new_mask_dir.mkdir(parents=True, exist_ok=True)

        sample_ids = set(annotations.index.get_level_values("sample_name"))
        for i, sample_id in enumerate(sample_ids):
            logger.info(f"[{i + 1}/{len(sample_ids)}] Processing sample {sample_id}")
            object_ids = set(annotations.loc[sample_id].index)
            mask = imread(mask_dir / f"{sample_id}.tiff")
            mask = filter_mask(mask, object_ids)
            save_mask(mask=mask, save_path=new_mask_dir / f"{sample_id}.tiff")

    def create_metadata(self):
        version_name = self.get_version_name(image_version="filtered", mask_version="filtered")
        metadata_dir = self.metadata_dir / version_name
        if metadata_dir.exists():
            return
        else:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            labels = pd.read_parquet(self.raw_annotations_path)

            for grp_name, grp_data in labels.groupby("sample_id"):
                grp_data = grp_data.convert_dtypes()
                grp_data = grp_data.astype("category")
                annotations_path = metadata_dir / f"{grp_name}.parquet"
                grp_data.to_parquet(annotations_path)

    def label_transfer(self):

        data_dir = self.raw_dir
        path_annotations = Path(data_dir / "clustering" / "annotations.parquet")

        # %%
        annotations = pd.read_parquet(path_annotations, engine="pyarrow")
        levels_to_drop = set(annotations.index.names) - {"sample_name", "object_id"}
        index = annotations.index.droplevel(level=list(levels_to_drop))
        annotations.index = index

        # %%
        # roi_240223_012 = annotations.loc[('240223_012',), :]
        # roi_240223_012.group_name.value_counts()
        # annotations = annotations.drop(labels='240223_012', level=0)
        assert "240223_012" not in set(annotations.index.get_level_values("sample_name"))

        # filter out specific `group_name` based on assessment of Francesco
        filter_ = annotations.group_name == "epithelial-Synaptophysin_pos-p53_pos"
        assert filter_.sum() == 56
        annotations = annotations[~filter_]

        # %%
        path_labels = Path(self.raw_dir / 'metadata' / 'label-names.xlsx')
        labels = pd.read_excel(path_labels, sheet_name="label-names", skiprows=0)

        cols = ["name_in_annotations", "main_group", "label"]
        filter_ = labels.columns.str.contains("meta_group")
        meta_group_cols = labels.columns[filter_]
        cols = cols + meta_group_cols.tolist()
        labels = labels[cols]

        assert labels.isna().any().any() == False
        labels = labels.set_index("name_in_annotations")

        assert set(labels.index) - set(annotations.group_name) == {
            "epithelial-Synaptophysin_pos-p53_pos"
        }

        # %% reassign parent group names according to Francesco's labels
        main_group = labels["main_group"].to_dict()
        annotations = annotations.assign(main_group=annotations.group_name.map(main_group))

        # for meta_group_col in meta_group_cols:
        #     meta_group = labels[meta_group_col].to_dict()
        #     annotations[meta_group_col] = annotations.group_name.map(meta_group)

        label = labels["label"].to_dict()
        annotations = annotations.assign(label=annotations.group_name.map(label))

        assert annotations.isna().any().any() == False
        annotations.index.names = ['sample_id', 'object_id']
        label_to_main_group = annotations.set_index('label')['main_group'].to_dict()

        # %% ASSIGN RECLUSTERD CELLS
        memberships = pd.read_parquet(self.raw_dir / 'reclustering/memberships.parquet')
        label_names = pd.read_excel(self.raw_dir / 'reclustering/label-names.xlsx', dtype=str)
        membership_to_label = label_names.set_index('membership')['label'].to_dict()
        memberships['label'] = memberships['membership'].map(membership_to_label)
        memberships['main_group'] = memberships['label'].map(label_to_main_group)
        assert not memberships.label.isna().any()

        filter_ = annotations.index.isin(memberships.index)
        assert filter_.sum() == len(memberships)
        annotations = annotations[~filter_]
        annotations = pd.concat([annotations, memberships])
        assert len(annotations) == 2214046
        assert annotations.index.is_unique

        # annotations.loc[:, 'main_group'] = annotations.main_group.fillna('mixed')

        # %% ASSIGN RECLUSTERD-V2 CELLS
        memberships = pd.read_parquet(self.raw_dir / 'reclustering-v2/memberships.parquet')
        label_names = pd.read_excel(self.raw_dir / 'reclustering-v2/label-names.xlsx', dtype=str)
        membership_to_label = label_names.set_index('membership')['label'].to_dict()
        memberships['label'] = memberships['membership'].map(membership_to_label)
        memberships['main_group'] = memberships['label'].map(label_to_main_group)
        assert not memberships.label.isna().any()

        filter_ = annotations.index.isin(memberships.index)
        assert filter_.sum() == len(memberships)
        annotations = annotations[~filter_]
        annotations = pd.concat([annotations, memberships])
        assert len(annotations) == 2214046
        assert annotations.index.is_unique

        annotations.loc[:, 'main_group'] = annotations.main_group.fillna('mixed')

        # %% create ids
        main_group_id_dict = sorted(annotations.main_group.unique())
        main_group_id_dict = {k: v for v, k in enumerate(main_group_id_dict)}
        annotations["main_group_id"] = annotations.main_group.map(main_group_id_dict)

        # for meta_group_col in meta_group_cols:
        #     meta_group_id_dict = sorted(annotations[meta_group_col].unique())
        #     meta_group_id_dict = {k: v for v, k in enumerate(meta_group_id_dict)}
        #     annotations[f"{meta_group_col}_id"] = annotations[meta_group_col].map(
        #         meta_group_id_dict
        #     )

        label_id_dict = sorted(annotations.label.unique())
        label_id_dict = {k: v for v, k in enumerate(label_id_dict)}
        annotations["label_id"] = annotations.label.map(label_id_dict)

        annotations = annotations[['label', 'main_group', 'label_id', 'main_group_id']]
        assert annotations.isna().any().any() == False

        id_cols = [col for col in annotations.columns if col.endswith("_id")]
        for id_col in id_cols:
            annotations.loc[:, id_col] = annotations[id_col].astype(int)

        # %% STATS
        cols = ["main_group", "label"]
        cell_counts = annotations[cols].groupby(cols).value_counts()
        cell_counts.to_csv(data_dir / "metadata" / "labels-counts.csv")

        # %%
        cols = ["sample_id", "label"]
        cell_counts = annotations[['label']].groupby(cols).value_counts().unstack()
        cell_counts = cell_counts.convert_dtypes()

        cell_freq = cell_counts.div(cell_counts.sum(axis=1), axis=0)
        cell_freq = cell_freq.convert_dtypes()

        top_n = 3
        top5_per_sample = cell_freq.reset_index().melt(id_vars=['sample_id'])
        top5_per_sample = top5_per_sample \
            .sort_values(['sample_id', 'value'], ascending=[True, False]) \
            .groupby('sample_id') \
            .head(top_n)
        top5_per_sample.to_csv(data_dir / f"top-{top_n}-labels-by-sample.csv")

        cell_counts.columns = pd.MultiIndex.from_product([cell_counts.columns, ['counts']], names=['label', 'stat'])
        cell_freq.columns = pd.MultiIndex.from_product([cell_freq.columns, ['freq']], names=['label', 'stat'])

        cell_stats = pd.concat((cell_counts, cell_freq), axis=1)
        cell_stats = cell_stats.sort_index(level=0, axis=1)
        cell_stats.to_csv(data_dir / "metadata" / "label-stats.csv")

        # %%
        annotations.to_parquet(self.raw_annotations_path, engine='fastparquet')

    def create_images(self):
        raw_acquisitions_dir = self.raw_dir / 'acquisitions'
        save_dir = self.get_image_version_dir('raw')
        panel = pd.read_parquet(save_dir / "panel.parquet")

        for img_path in raw_acquisitions_dir.glob("*.tiff"):
            save_path = save_dir / img_path.name

            if save_path.exists():
                logger.info(f"Image {img_path.name} already exists. Skipping.")
                continue

            logger.info(f"Processing {img_path.name}")

            img = imread(img_path)
            img = img[panel.page]
            imwrite(save_path, img)

    def analyse_differences_between_raw_and_filtered_images(self):

        mcd_metadata = pd.read_parquet(
            '/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/01_raw/metadata/02_processed/mcd_metadata.parquet')

        raw = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/01_raw/images/raw').glob('*.tiff')
        raw = set([i.name for i in raw])

        filtered = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/01_raw/images/filtered').glob('*.tiff')
        filtered = set([i.name for i in filtered])

        acq = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/01_raw/acquisitions').glob('*.tiff')
        acq = set([i.name for i in acq])
        acq = set(['_'.join(i.split('_')[:-1]) + '.tiff' for i in acq])

        len(raw)
        len(acq)
        len(filtered)

        annotations = pd.read_parquet(
            '/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/01_raw/clustering/annotations.parquet')
        anno = set([f'{i}.tiff' for i in annotations.index.get_level_values('sample_name').unique()])

        diff1 = raw - filtered
        diff2 = raw - anno
        diff = diff1.union(diff2)

        # list of removed images with justification
        diff.update([
            "240219_004.tiff",  # remove after inspection of v2 clustering
            '231207_006.tiff',  # low cell count and staining issues
            "240223_012.tiff",  # unspecific staining
            '240119_015.tiff',  # undefined-high fraction
            '240119_016.tiff',  # undefined-high fraction
            '231204_010.tiff',  # undefined-high fraction
            '231204_003.tiff',  # undefined-high fraction
            '231206_007.tiff',  # undefined-high fraction
            '231210_002.tiff',  # undefined-high fraction
            '231204_003.tiff',  # undefined-high fraction
            '240119_017.tiff',  # undefined-high fraction
        ])
        # mcd_metadata[mcd_metadata.file_name_napari.isin(diff)].file_name.to_list()

        exclude_rois = mcd_metadata[mcd_metadata.file_name_napari.isin(diff)][['file_name', 'file_name_napari']]
        # original mapping
        # [
        #     {'file_name': '240119_IIBL_X7Y4_34.tiff', 'file_name_napari': '240119_016.tiff'},
        #     {'file_name': '240119_IIBL_X8Y4_35.tiff', 'file_name_napari': '240119_017.tiff'},
        #     {'file_name': '240219_IVB_X5Y2_8.tiff', 'file_name_napari': '240219_004.tiff'},
        #     {'file_name': '240219_IVB_X7Y2_10.tiff', 'file_name_napari': '240219_005.tiff'},
        #     {'file_name': '240223_IVB_X7Y10_82.tiff', 'file_name_napari': '240223_012.tiff'},
        #     {'file_name': '231210_IBL_X2Y20_209.tiff', 'file_name_napari': '231210_002.tiff'},
        #     {'file_name': '231210_IBL_X7Y21_226.tiff', 'file_name_napari': '231210_012.tiff'},
        #     {'file_name': '231210_IBL_X8Y21_227.tiff', 'file_name_napari': '231210_013.tiff'},
        #     {'file_name': '231204_IBL_X5Y2_8.tiff', 'file_name_napari': '231204_003.tiff'},
        #     {'file_name': '231204_IBL_X1Y4_28.tiff', 'file_name_napari': '231204_010.tiff'},
        #     {'file_name': '231204_LP1.tiff', 'file_name_napari': '231204_017.tiff'},
        #     {'file_name': '231204_LP2.tiff', 'file_name_napari': '231204_018.tiff'},
        #     {'file_name': '231204_LP3.tiff', 'file_name_napari': '231204_019.tiff'},
        #     {'file_name': '231204_LP4.tiff', 'file_name_napari': '231204_020.tiff'},
        #     {'file_name': '231206_IBL_X4Y9_91.tiff', 'file_name_napari': '231206_007.tiff'}
        # ]

    def get_excluded_rois(self) -> list[str]:
        exclude_rois = [
            '240119_iibl_x7y4_34_16.tiff',
            '240119_iibl_x8y4_35_17.tiff',
            '240219_ivb_x5y2_8_4.tiff',
            '240219_ivb_x7y2_10_5.tiff',
            '240223_ivb_x7y10_82_12.tiff',
            '231210_ibl_x2y20_209_2.tiff',
            '231210_ibl_x7y21_226_12.tiff',
            '231210_ibl_x8y21_227_13.tiff',
            '231204_ibl_x5y2_8_3.tiff',
            '231204_ibl_x1y4_28_10.tiff',
            '231206_ibl_x4y9_91_7.tiff',
            '231204_lp1_17.tiff',
            '231204_lp2_18.tiff',
            '231204_lp3_19.tiff',
            '231204_lp4_20.tiff'
        ]
        return exclude_rois

    def create_filtered_images(self):

        raw_acquisitions_dir = self.raw_dir / 'acquisitions'
        panel = pd.read_parquet(self.get_panel_path('raw'))

        panel_save_path = self.get_panel_path('filtered')
        panel_save_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_save_path, engine='fastparquet')

        excl_rois = self.get_excluded_rois()

        save_dir = self.get_image_version_dir('filtered')
        save_dir.mkdir(parents=True, exist_ok=True)
        for img_path in raw_acquisitions_dir.glob("*.tiff"):

            if img_path.name in excl_rois:
                logger.info(f'Filtering out {img_path.name}')
                continue

            save_path = save_dir / img_path.name

            if save_path.exists():
                logger.info(f"Image {img_path.name} already exists. Skipping.")
                continue

            logger.info(f"Processing {img_path.name}")

            img = imread(img_path)
            img = img[panel.page]
            imwrite(save_path, img)

    def get_identifiers_table(self):
        import re

        mcd_metadata = pd.read_parquet(self.raw_dir / 'metadata/02_processed/mcd_metadata.parquet')
        mapping = mcd_metadata[['file_name_napari', 'file_name']]
        mapping.loc[:, 'file_name_tidy'] = mapping.file_name.map(lambda x: f'{tidy_name(Path(x).stem)}.tiff')

        # NOTE: this is the acquisition that is corrupted
        filter_ = mapping.file_name_tidy == '240210_iiibl_x3y15_150.tiff'
        mapping = mapping[~filter_]

        sample_ids = set([i.stem for i in (self.raw_dir / 'acquisitions').glob('*.tiff')])
        file_name_to_sample_id = {f"{re.sub(r'_\d+$', '', i)}.tiff": i for i in sample_ids}
        assert len(file_name_to_sample_id) == len(sample_ids)
        assert set(mapping.file_name_tidy) == set(file_name_to_sample_id.keys())

        mapping.loc[:, 'sample_id'] = mapping.file_name_tidy.map(file_name_to_sample_id)
        return mapping

    def get_napari_file_name_to_sample_id_mapping(self):
        mapping = self.get_identifiers_table()
        mapping = mapping.set_index('file_name_napari')['sample_id'].to_dict()
        assert len(mapping) == 549

        return mapping


    def create_masks(self):
        import shutil

        save_dir = self.get_mask_version_dir('deepcell')
        save_dir.mkdir(parents=True, exist_ok=True)

        mapping = self.get_napari_file_name_to_sample_id_mapping()

        mask_paths = (self.raw_dir / 'masks' / 'deepcell').glob('*.tiff')
        for mask_path in mask_paths:
            sample_id = mapping[mask_path.name]
            save_path = save_dir / f"{sample_id}.tiff"
            shutil.copy2(mask_path, save_path)

    def create_filtered_masks(self):
        import shutil

        save_dir = self.get_mask_version_dir('filtered')
        save_dir.mkdir(parents=True, exist_ok=True)

        mapping = self.get_napari_file_name_to_sample_id_mapping()
        excl_rois = self.get_excluded_rois()

        mask_paths = (self.raw_dir / 'masks' / 'deepcell').glob('*.tiff')
        for mask_path in mask_paths:
            sample_id = mapping[mask_path.name]
            save_path = save_dir / f"{sample_id}.tiff"

            if save_path.name in excl_rois:
                continue

            shutil.copy2(mask_path, save_path)



    def create_graphs(self):
        pass

    def create_features_spatial(self):
        pass

    def create_features_intensity(self):
        pass

    def create_features(self):
        self.create_features_spatial()
        self.create_features_intensity()

    def create_panel(self):
        from ai4bmr_core.utils.tidy import tidy_name
        import json
        import pandas as pd
        panel = None

        raw_acquisitions_dir = self.raw_dir / 'acquisitions'
        for img_path in raw_acquisitions_dir.glob("*.json"):
            with open(img_path, "r") as f:
                img_metadata = json.load(f)
                df = pd.DataFrame(img_metadata)
                df = df[["channel_names", "channel_labels", "metal_names"]]

            if panel is None:
                panel = df
            else:
                assert panel.equals(df)

        keep = panel.channel_labels.str.len() != 0
        panel["keep"] = keep

        panel.index.name = 'page'
        panel = panel.reset_index()

        assert keep.sum() == 40
        panel["target"] = panel.channel_labels.map(tidy_name).values
        panel = panel.rename(columns={'channel_names': 'metal_tag',
                                      'channel_labels': 'channel_label',
                                      'metal_names': 'metal_name'})
        panel = panel.convert_dtypes()
        panel.index.name = "channel_index"

        panel.to_parquet(raw_acquisitions_dir / "panel.parquet")

        panel = panel[panel.keep].reset_index(drop=True)
        panel.index.name = "channel_index"

        metal_tag_to_uniprot_id = {'Pr141': 'P63267', 'Nd142': 'P07288', 'Nd143': 'P08670', 'Nd144': 'P02452',
                                   'Nd145': 'P08247', 'Nd146': 'P13647', 'Sm147': 'P46937', 'Nd148': 'P78386',
                                   'Sm149': 'P23141', 'Nd150': 'P18146', 'Eu151': 'P16284', 'Sm152': 'P08575',
                                   'Eu153': 'P16070', 'Sm154': 'Q12884', 'Gd155': 'Q9BZS1', 'Gd156': 'P01730',
                                   'Gd158': 'P12830', 'Tb159': 'P34810', 'Gd160': 'P31997', 'Dy161': 'P11836',
                                   'Dy162': 'P01732', 'Dy163': 'P11215', 'Dy164': 'Q9H3D4', 'Ho165': 'P35222',
                                   'Er166': 'Q86YL7', 'Er167': 'P17813', 'Er168': 'P46013', 'Tm169': 'P04637',
                                   'Er170': 'P04234', 'Yb171': 'P11308', 'Yb172': 'P42574', 'Yb173': 'P51911',
                                   'Yb174': 'P05783', 'Lu175': 'P43121', 'Yb176': 'P10275', 'Ir191': None,
                                   'Ir193': None, 'Pt195': None, 'Pt196': None, 'Pt198': None}
        panel["uniprot_id"] = panel.metal_tag.map(metal_tag_to_uniprot_id)

        panel_path = self.get_panel_path("raw")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)

    @property
    def name(self):
        return "PCa"

    @property
    def id(self):
        return "PCa"

    @property
    def doi(self):
        return "unpublished"

    def create_clinical_metadata(self):
        """
        This computes the metadata for each object in the dataset. The results can still contain objects that have been
        removed after the clustering, i.e. all elements that where used in the clustering are included in the metadata.
        """
        if self.clinical_metadata_path.exists():
            return

        # %% read metadata
        roi_match_blockId = pd.read_excel(self.raw_clinical_metadata_path, dtype=str)
        roi_match_blockId = roi_match_blockId.drop(columns=["patient level code"])
        roi_match_blockId = roi_match_blockId.rename(
            columns={"Description (slidecode_coordinates_uniquecoreID)": "description"}
        )

        slide_code = roi_match_blockId["description"].str.split("_").str[0].str.strip()
        tma_coordinates = (
            roi_match_blockId["description"].str.split("_").str[1].str.strip()
        )
        tma_sample_ids = (
            roi_match_blockId["description"].str.split("_").str[2].str.strip()
        )

        roi_match_blockId = roi_match_blockId.assign(
            tma_sample_id=tma_sample_ids,
            slide_code=slide_code,
            tma_coordinates=tma_coordinates,
        )

        roi_match_blockId.loc[
            roi_match_blockId.tma_sample_id == "150 - split", "tma_sample_id"
        ] = "150"
        assert roi_match_blockId.tma_sample_id.str.contains("split").sum() == 0

        patient_ids = (
            roi_match_blockId["Original block number"].str.split("_").str[0].str.strip()
        )
        roi_match_blockId = roi_match_blockId.assign(PAT_ID=patient_ids)

        # NOTE: these are the LP samples (i.e. test runs)
        assert roi_match_blockId.PAT_ID.isna().sum() == 4
        roi_match_blockId = roi_match_blockId.dropna(subset=["PAT_ID"])

        assert not roi_match_blockId.slide_code.isna().any()
        assert not roi_match_blockId.tma_coordinates.isna().any()
        assert not roi_match_blockId.tma_sample_id.isna().any()

        tma_id = roi_match_blockId.slide_code + "_" + roi_match_blockId.tma_coordinates
        roi_match_blockId["tma_id"] = tma_id

        tma_tidy = pd.read_csv(self.raw_tma_tidy_path, sep="\t", dtype=str)
        tma_tidy = tma_tidy.drop(columns=["DATE_OF_BIRTH", "INITIALS"])

        tma_tidy.loc[:, "PAT_ID"] = tma_tidy.PAT_ID.astype(str).str.strip()

        # NOTE: the `PATIENT_ID` of the first patient is NA for no obvious reason.
        assert tma_tidy.PATIENT_ID.isna().sum() == 1
        assert tma_tidy["AGE_AT_SURGERY"].isna().any() == False

        metadata = pd.merge(roi_match_blockId, tma_tidy, how="left")
        metadata = metadata.sort_values(["PAT_ID"])

        # NOTE: all patients in the roi_match_block are present in the metadata
        assert metadata.PAT_ID.isna().sum() == 0
        # NOTE: 3 ROIs from 2 patients have no meta data
        assert metadata.DONOR_BLOCK_ID.isna().sum() == 3
        assert metadata[metadata.DONOR_BLOCK_ID.isna()].PAT_ID.nunique() == 2

        # NOTE: for 3 ROIs we do not have patient metadata
        #   thus from the 546 images we have, only 543 are used in downstream analysis
        metadata = metadata[metadata.DONOR_BLOCK_ID.notna()]

        # verify
        # TODO: remove after verification by collaborators
        for i in range(metadata.shape[0]):
            id_ = metadata["tma_sample_id"].values[i]
            ids_ = metadata[
                [
                    "UNIQUE_TMA_SAMPLE_ID_1",
                    "UNIQUE_TMA_SAMPLE_ID_2",
                    "UNIQUE_TMA_SAMPLE_ID_3",
                    "UNIQUE_TMA_SAMPLE_ID_4",
                ]
            ].iloc[i]
            ids_ = set(ids_)
            if id_ not in ids_:
                pass
                print(id_, metadata.PAT_ID.iloc[i])

        filter_ = metadata["Original block number"] == metadata["DONOR_BLOCK_ID"]
        # NOTE: these mismatches are only because we have high and low gleason for the same patient
        mismatches = metadata[
            [
                "PAT_ID",
                "Gleason pattern (TMA core)",
                "Original block number",
                "DONOR_BLOCK_ID",
            ]
        ][~filter_]

        # %% tidy
        metadata = metadata.astype(str)

        metadata.columns = [
            tidy_name(c, split_camel_case=False) for c in metadata.columns
        ]

        metadata["date_of_surgery"] = pd.to_datetime(metadata["date_of_surgery"])

        metadata["age_at_surgery"] = metadata["age_at_surgery"].astype(int)

        mapping = {
            "High Gleason pattern": "high",
            "Low Gleason pattern": "low",
            "Only 1 Gleason pattern": "only_1",
            "unkown": "unknown",
        }
        metadata["gleason_pattern_tma_core"] = metadata.gleason_pattern_tma_core.map(
            mapping
        )

        metadata["psa_at_surgery"] = metadata.psa_at_surgery.astype(float)
        assert metadata["psa_at_surgery"].isna().sum() == 0

        metadata["last_fu"] = metadata["last_fu"].astype(int)
        assert metadata.last_fu.isna().sum() == 0

        metadata["cause_of_death"] = metadata.cause_of_death.map(
            {"0": "alive", "1": "non-PCa_death", "2": "PCa_death"}
        )
        assert metadata.cause_of_death.isna().sum() == 0

        metadata["os_status"] = metadata.os_status.map({"0": "alive", "1": "dead"})
        assert metadata.os_status.isna().sum() == 0

        metadata["psa_progr"] = metadata.psa_progr.astype(int)
        assert metadata.psa_progr.isna().sum() == 0

        metadata["psa_progr_time"] = metadata.psa_progr_time.astype(int)
        assert metadata.psa_progr_time.isna().sum() == 0

        metadata["clinical_progr"] = metadata.clinical_progr.astype(int)
        assert metadata.clinical_progr.isna().sum() == 0

        metadata["clinical_progr_time"] = metadata.clinical_progr_time.astype(int)
        assert metadata.clinical_progr_time.isna().sum() == 0

        metadata["disease_progr"] = metadata.disease_progr.astype(int)
        assert metadata.disease_progr.isna().sum() == 0

        metadata["disease_progr_time"] = metadata.disease_progr_time.astype(int)
        assert metadata.disease_progr_time.isna().sum() == 0

        metadata["recurrence_loc"] = metadata.recurrence_loc.map(
            {
                "0": "no_recurrence",
                "1": "local",
                "2": "metastasis",
                "3": "local_and_metastasis",
            }
        )
        assert metadata.recurrence_loc.isna().sum() == 0

        mapping = {
            i: "recurrence" for i in ["local", "local_and_metastasis", "metastasis"]
        }
        metadata["recurrence"] = metadata["recurrence_loc"].astype(str).replace(mapping)

        # NOTE: we do not have the mapping at this point
        # metadata['cgrading_biopsy'] = metadata.cgrading_biopsy.astype(int)

        metadata["cgs_pat_1"] = metadata.cgs_pat_1.astype(int)
        assert metadata.cgs_pat_1.isna().sum() == 0

        metadata["cgs_pat_2"] = metadata.cgs_pat_2.astype(int)
        assert metadata.cgs_pat_2.isna().sum() == 0

        metadata["cgs_score"] = metadata.cgs_score.astype(int)
        assert metadata.cgs_score.isna().sum() == 0

        # metadata['gs_grp'] = metadata.gs_grp.map({'1': '<=6', '2': '3+4=7', '3': '4+3=7', '4': '8', '5': '9/10'})
        # assert metadata.gs_grp.isna().sum() == 0

        metadata["pgs_score"] = metadata.pgs_score.astype(int)
        assert metadata.pgs_score.isna().sum() == 0

        metadata["ln_status"] = metadata.ln_status.astype(int)
        assert metadata.ln_status.isna().sum() == 0

        metadata["surgical_margin_status"] = metadata.surgical_margin_status.astype(
            float
        )
        assert metadata.surgical_margin_status.astype(float).isna().sum() == 87

        metadata["d_amico_risk"] = metadata["d_amico"].map(
            {"Intermediate Risk": "intermediate", "High Risk": "high"}
        )
        metadata = metadata.drop("d_amico", axis=1)

        metadata = metadata.convert_dtypes()

        # %%
        # vals = metadata.cause_of_death.astype(int).values
        # cats = sorted(set(vals))
        # metadata['cause_of_death'] = pd.Categorical(vals, categories=cats, ordered=True)
        # assert metadata.cause_of_death.isna().sum() == 0

        # %%
        sample_names = [Path(i).stem for i in metadata.file_name_napari]
        metadata = metadata.assign(sample_id=sample_names)
        metadata = metadata.set_index("sample_id")

        # %%
        cat_cols = [
            "file_name_napari",
            "original_block_number",
            "gleason_pattern_tma_core",
            "tma_sample_id",
            "slide_code",
            "pat_id",
            "patient_id",
            "donor_block_id",
            "unique_tma_sample_id_1",
            "unique_tma_sample_id_2",
            "unique_tma_sample_id_3",
            "unique_tma_sample_id_4",
            "cause_of_death",
            "os_status",
            "psa_progr",
            "clinical_progr",
            "disease_progr",
            "recurrence",
            "recurrence_loc",
            "cgrading_biopsy",
            "cgs_pat_1",
            "cgs_pat_2",
            "cgs_score",
            "gs_grp",
            "ct_stage",
            "pt_stage",
            "pgs_score",
            "ln_status",
            "surgical_margin_status",
            "adj_adt",
            "adj_radio",
            "d_amico_risk",
        ]
        metadata = metadata.astype({k: "category" for k in cat_cols})

        # %%
        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path, engine="fastparquet")

    def create_tma_annotations(self):
        samples = pd.read_parquet(self.clinical_metadata_path, engine="fastparquet")
        df = pd.read_excel(self.raw_tma_annotations_path, sheet_name="PCa_ROIs_final")
        # set(samples.index) - set(df.sample_name)

        df.columns = df.columns.map(tidy_name)
        df = df.rename(columns={"unnamed_12": "notes"})
        df["notes"] = df.notes.astype(str).str.strip()

        filter_ = df.sample_name == "sample_name"
        df = df[~filter_]

        filter_ = df.sample_name.notna()
        df = df[filter_]

        mapping = {"y": "yes", "n": "no", "/": "no_evaluation", "nan": pd.NA}
        df = df.map(lambda x: mapping.get(x, x))

        # df.gs_pat_1.unique()
        mapping = {
            "tumor not evident": "no tumor",
            "no evident tumor": "no tumor",
            "Tumor??": "not sure if tumor",
            "no": "unclear annotation",
        }
        df["gs_pat_1"] = df.gs_pat_1.map(lambda x: mapping.get(x, x))
        # df["gs_pat_1"].unique()
        # filter_ = df["gs_pat_1"] == 'unclear annotation'
        # df[filter_]

        mapping = {
            "4 (3 is also an option)": 4,
            "4(few)": 4,
        }
        df["gs_pat_2"] = df.gs_pat_2.map(lambda x: mapping.get(x, x))

        mapping = {"yes,": "yes", "nyes": "yes"}
        df["glandular_atrophy_pin"] = df.glandular_atrophy_pin.map(
            lambda x: mapping.get(x, x)
        )

        mapping = {"yed": "yes"}
        df["non_stromogenic_smc_abundant"] = df.non_stromogenic_smc_abundant.map(
            lambda x: mapping.get(x, x)
        )

        # mapping = {"yno": "unclear_annotation"}
        # df["inflammation"] = df.inflammation.map(lambda x: mapping.get(x, x))

        mapping = {"n3": "no"}
        df["cribriform"] = df.cribriform.map(lambda x: mapping.get(x, x))

        tidy_col_vals = [
            "gs_pat_1",
            "gs_pat_2",
            "stromogenic_smc_loss_reactive_stroma_present",
            "non_stromogenic_smc_abundant",
            "inflammation",
            "glandular_atrophy_pin",
            "cribriform",
        ]
        for col in tidy_col_vals:
            df[col] = df[col].astype(str).map(tidy_name)

        # note: ensure all values are in the expected set of values
        assert df[tidy_col_vals].isin([
            # assessment not possible
            'no_tissue',
            'too_much_loss',
            # 'no_evident_tumor', -> "no_tumor"
            # 'tumor_not_evident', -> "no_tumor"
            'no_evaluation',
            'not_enough_tumor',
            'not_enough_material',
            'not_sure_if_tumor',
            'unclear_annotation',
            # valid cores
            'nan',
            'no_tumor',
            'yes',
            'no',
            '3',
            '4',
            '5',
        ]).all().all()

        filter_ = df.gs_pat_1.isin(
            [
                # assessment not possible
                # 'no_tissue',
                # 'too_much_loss',
                # 'no_evident_tumor',
                # 'tumor_not_evident',
                # 'no_evaluation',
                # 'not_enough_tumor',
                # 'not_enough_material',
                # 'not_sure_if_tumor',
                # 'unclear_annotation',
                # 'nan',
                # cores with assessment
                'no_tumor',
                'yes',
                'no',
                '3',
                '4',
                '5',
            ]
        )

        df = df[filter_]

        mapping = {"nan": pd.NA, None: pd.NA}
        df = df.map(lambda x: mapping.get(x, x))

        assert df.gs_pat_1.isna().sum() == 0

        df = df.convert_dtypes()
        df = df.rename(columns={"sample_name": "sample_id"}).set_index("sample_id")

        # note: fix tma sample id for split sample 150
        df.loc[df.tma_sample_id == "150_1", "tma_sample_id"] = "150"

        # NOTE: check that the data in samples and the data in the tma annotations are the same
        shared_sample_ids = list(set(df.index).intersection(set(samples.index)))
        for i in ["tma_coordinates", "tma_sample_id"]:
            s1 = set(samples.loc[shared_sample_ids][i].astype(str).to_dict().items())
            s2 = set(df.loc[shared_sample_ids][i].astype(str).to_dict().items())
            assert s1 == s2

        set(samples).intersection(set(df))
        shared_cols = ["description", "tma_sample_id", "tma_coordinates"]
        df = df.drop(columns=shared_cols)

        # note: for some reason we cannot store boolean as a category, we use yes/no instead
        df["is_tumor"] = "yes"
        df.loc[df.gs_pat_1 == "no_tumor", "is_tumor"] = "no"

        mapping = {"no_tumor": pd.NA, "3": 3, "4": 4, "5": 5}
        df["gs_pat_1"] = df.gs_pat_1.map(lambda x: mapping.get(x, x))

        mapping = {"no_tumor": pd.NA, "3": 3, "4": 4, "5": 5}
        df["gs_pat_2"] = df.gs_pat_2.map(lambda x: mapping.get(x, x))

        df["gleason_score"] = pd.NA
        gleason_score = df.gs_pat_1.astype(str) + "+" + df.gs_pat_2.astype(str)
        filter_ = df.gs_pat_1.notna()
        df.loc[filter_, "gleason_score"] = gleason_score[filter_]

        gleason_score_sum = df.gs_pat_1[filter_].astype(int) + df.gs_pat_2[
            filter_
        ].astype(int)
        df["gleason_score_sum"] = pd.NA
        df.loc[filter_, "gleason_score_sum"] = gleason_score_sum

        df["gleason_grp"] = pd.NA

        filter_ = df.gleason_score_sum <= 6
        df.loc[filter_, "gleason_grp"] = 1

        filter_ = df.gleason_score == "3+4"
        df.loc[filter_, "gleason_grp"] = 2

        filter_ = df.gleason_score == "4+3"
        df.loc[filter_, "gleason_grp"] = 3

        filter_ = df.gleason_score_sum == 8
        df.loc[filter_, "gleason_grp"] = 4

        filter_ = df.gleason_score_sum >= 9
        df.loc[filter_, "gleason_grp"] = 5

        df = df.convert_dtypes()

        cat_cols = [
            "annotation_roi_he",
            "gs_pat_1",
            "gs_pat_2",
            "stromogenic_smc_loss_reactive_stroma_present",
            "non_stromogenic_smc_abundant",
            "inflammation",
            "glandular_atrophy_pin",
            "cribriform",
            "is_tumor",
            "gleason_grp",
            "gleason_score_sum",
            "gleason_score",
        ]

        df = df.astype({k: "category" for k in cat_cols})
        # df.to_csv('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-datasets/tests/tma_annotations.csv')
        samples = pd.concat([samples, df], axis=1)
        samples = samples.convert_dtypes()
        samples.to_parquet(self.clinical_metadata_path, engine="fastparquet")
