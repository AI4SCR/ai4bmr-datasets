from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from ai4bmr_core.utils.tidy import tidy_name
from loguru import logger

from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset


class BLCa(BaseIMCDataset):

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)

        # raw paths
        self.raw_metadata_path = self.base_dir / '01_raw' / 'metadata' / '02_processed' / 'labels.parquet'
        self.mcd_metadata_path = self.base_dir / "metadata" / "02_processed" / "mcd_metadata.parquet"

    def prepare_data(self):
        # self.process_acquisitions()
        self.create_panel()

        self.create_clinical_metadata()
        self.create_metadata()
        self.create_mask_version_for_all_cells_used_in_clustering()
        self.create_mask_version_for_annotated_cells()

        self.compute_features(image_version="filtered", mask_version="clustered")
        self.compute_features(image_version="filtered", mask_version="annotated")

    def process_acquisitions(self):
        import json
        import re
        from skimage.io import imwrite
        from readimc import MCDFile

        acquisitions_dir = ''
        acquisitions_dir.mkdir(parents=True, exist_ok=True)

        num_failed_reads = 0

        mcd_paths = [i for i in self.raw_dir.rglob("*.mcd") if i.is_file()]
        for mcd_path in mcd_paths:
            logger.info(f"Processing {mcd_path.name}")
            mcd_id = re.search(r".+_TMA_(\d+_\w+).mcd", mcd_path.name).group(1)
            mcd_id = mcd_id.replace('B_2', 'B')

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

                        image_name = (
                            f"{mcd_id}_{acq.id}"  # should be equivalent to 'Tma_ac'
                        )
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
                            logger.error(f"Failed to read acquisition! ")
                            num_failed_reads += 1
                            continue

    def create_panel_form_acquisitions(self):
        from ai4bmr_core.utils.tidy import tidy_name
        import json
        import pandas as pd
        panel = None

        for img_metadata_path in self.raw_acquisitions_dir.glob("*.json"):
            with open(img_metadata_path, "r") as f:
                img_metadata = json.load(f)
                df = pd.DataFrame(img_metadata)
                df = df[["channel_names", "channel_labels", "metal_names"]]

            if panel is None:
                panel = df
            else:
                assert panel.equals(df)

        targets = panel["channel_labels"].str.split("_")
        keep = targets.str.len() == 2

        assert keep.sum() == 43
        panel["target"] = targets.str[0].map(tidy_name).values
        # add suffix to target for duplicates (iridium)
        panel['target'] = panel.target + '_' + panel.groupby('target').cumcount().astype(str)
        panel.loc[:, 'target'] = panel.target.str.replace('_0$', '', regex=True)

        panel["keep"] = keep

        panel.columns = panel.columns.map(tidy_name)
        panel = panel.convert_dtypes()
        panel["page"] = range(len(panel))
        panel.index.name = "channel_index"

        panel.to_parquet(self.raw_acquisitions_dir / "panel.parquet")

        panel = panel[panel.keep].reset_index(drop=True)
        panel.index.name = "channel_index"
        panel_path = self.get_panel_path("published")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)

    def create_clinical_metadata(self):
        logger.info("Creating clinical metadata")

        raw_dir = self.raw_dir / "metadata" / "01_raw"
        processed_dir = self.raw_dir / "metadata" / "02_processed"

        # ROI matching
        filename = "Analyses_cohort 2.xlsx"
        roi_matching = pd.read_excel(raw_dir / filename, dtype=str, header=4)
        cont = []
        for i, _type in enumerate(["tur-b", "primary-tumor", "lymph-node-metastases"]):
            cols = [f"Localisation.{i}", f"Case.{i}"] if i else ["Localisation", "Case"]
            dat = roi_matching[cols]
            dat.columns = ["sample_id", "patient_id"]
            dat = dat.assign(type=_type)
            cont.append(dat)

        ids = pd.concat(cont)
        ids = ids.dropna(subset="sample_id")
        ids.loc[:, "patient_id"] = ids.patient_id.ffill()
        assert ids.sample_id.duplicated().any() == False
        assert ids.patient_id.value_counts().max() <= 6

        # mcd data
        mcd_metadata_path = self.raw_dir / 'metadata' / '02_processed' / 'mcd_metadata.parquet'
        mcd_metadata = pd.read_parquet(mcd_metadata_path)
        # NOTE: rename acquisitions that were split
        mcd_metadata = mcd_metadata.assign(sample_id=mcd_metadata.Description)
        mcd_metadata.loc[
            mcd_metadata.Description.str.startswith("1C2b"), "sample_id"
        ] = "1C2b"
        mcd_metadata.loc[
            mcd_metadata.Description.str.startswith("1C3i"), "sample_id"
        ] = "1C3i"

        # create acquisition_id
        mcd_metadata = mcd_metadata.assign(
            acquisition_id=mcd_metadata.file_name_napari.map(lambda x: Path(x).stem)
        )

        sample_ids_with_not_in_analysis = set(mcd_metadata.sample_id) - set(
            ids.sample_id
        )
        sample_ids_with_no_acquisition = set(ids.sample_id) - set(
            mcd_metadata.sample_id
        )

        metadata = pd.merge(ids, mcd_metadata, how="outer")
        metadata[metadata.sample_id.isin(sample_ids_with_no_acquisition)].to_csv(
            processed_dir / "samples_with_no_acquisition.csv"
        )
        metadata[metadata.sample_id.isin(sample_ids_with_not_in_analysis)].to_csv(
            processed_dir / "sample_ids_with_not_in_analysis.csv"
        )

        sample_ids_to_remove = sample_ids_with_not_in_analysis.union(
            sample_ids_with_no_acquisition
        )
        metadata = metadata[~metadata.sample_id.isin(sample_ids_to_remove)]

        # clinical metadata
        clinical_metadata = pd.read_excel(
            raw_dir / "metadata_NAC_TMA_11_08.xlsx", dtype=str
        )
        clinical_metadata = clinical_metadata.iloc[:59, :]
        clinical_metadata = clinical_metadata.dropna(
            axis=1, how="all"
        )  # remove NaN columns
        # NOTE: some patients have a `Nr old` but everything else is NaN
        clinical_metadata = clinical_metadata[
            ~clinical_metadata.drop(columns="Nr old").isna().all(axis=1)
        ]
        # NOTE: `Nr. old` are for the matching with TMA and new ones for the genomic profiling
        clinical_metadata = clinical_metadata.assign(
            patient_id=clinical_metadata["Nr old"].astype(int).astype(str)
        )

        patient_ids_with_no_acquisition = set(clinical_metadata.patient_id) - set(
            metadata.patient_id
        )
        patient_ids_with_no_clinical_metadata = set(metadata.patient_id) - set(
            clinical_metadata.patient_id
        )

        clinical_metadata[
            clinical_metadata.patient_id.isin(patient_ids_with_no_acquisition)
        ].to_csv(processed_dir / "patient_ids_with_no_acquisition.csv")
        metadata[
            metadata.patient_id.isin(patient_ids_with_no_clinical_metadata)
        ].to_csv(processed_dir / "patient_ids_with_no_clinical_metadata.csv")

        patient_ids_to_remove = patient_ids_with_no_acquisition.union(
            patient_ids_with_no_clinical_metadata
        )
        clinical_metadata = clinical_metadata[
            ~clinical_metadata.patient_id.isin(patient_ids_to_remove)
        ]
        metadata = metadata[~metadata.patient_id.isin(patient_ids_to_remove)]

        mask = clinical_metadata.columns.str.contains("date", case=False)
        date_cols = clinical_metadata.columns[mask]
        date_cols = date_cols.tolist() + ["Start Chemo", "End Chemo"]

        clinical_metadata.loc[
            clinical_metadata["End Chemo"] == "31.02.2010", "End Chemo"
        ] = pd.NaT
        for col in date_cols:
            clinical_metadata.loc[:, col] = pd.to_datetime(clinical_metadata[col])

        # metadata
        metadata_full = pd.merge(metadata, clinical_metadata, how="left")
        assert len(metadata) == len(metadata_full)

        # add group and group_id
        metadata_full['group'] = metadata_full['sample_id'].str.extract(r'([ABC])')
        group_map = {'A': 1, 'B': 2, 'C': 3}
        metadata_full['group_id'] = metadata_full['group'].map(group_map)

        # add responders and non responders
        metadata_full['RC: ypT'] = metadata_full['RC: ypT'].astype(int)
        metadata_full['Resp_Status'] = metadata_full['RC: ypT'].apply(lambda x: 1 if x == 0 else 0)

        # convert to parquet compatible types
        metadata_full = metadata_full.rename(columns=dict(sample_id="roi_id", acquisition_id='sample_id'))
        metadata_full = metadata_full.convert_dtypes()

        metadata_full.columns = metadata_full.columns.map(tidy_name)
        metadata_full = metadata_full.rename(columns=dict(rc_yp_t='rc_ypt'))

        metadata_full = metadata_full.set_index("sample_id")
        metadata_full.to_parquet(self.clinical_metadata_path, engine='fastparquet')

    def create_mask_version_for_all_cells_used_in_clustering(self):
        logger.info("Creating masks for clustered cells")

        cluster_ids = pd.read_parquet(
            self.raw_dir / 'clustering/2024-07-17_13-56/cluster-ids.parquet')

        object_ids = cluster_ids['id'].str.split('_').str[-1:].str.join('_')
        sample_ids = cluster_ids['id'].str.split('_').str[:-1].str.join('_')
        cluster_ids['sample_id'] = sample_ids
        cluster_ids['object_id'] = object_ids.astype(int)

        masks_dir = self.raw_dir / 'masks' / 'deepcell'
        save_dir = self.masks_dir / 'clustered'
        save_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, grp_data in cluster_ids.groupby('sample_id'):
            mask_deepcell = tifffile.imread(masks_dir / f'{sample_id}.tiff')
            object_ids = grp_data['object_id'].values

            valid_ids = np.isin(mask_deepcell, object_ids)
            mask = np.where(valid_ids, mask_deepcell, 0)
            tifffile.imwrite(save_dir / f'{sample_id}.tiff', mask)

        for path in save_dir.glob("*.tiff"):
            mask = tifffile.imread(path)
            uniq_objs = np.unique(mask)
            assert len(uniq_objs) > 1

    def create_mask_version_for_annotated_cells(self):
        logger.info("Creating masks for annotated cells")

        # labels = pd.read_parquet(self.raw_dir / 'metadata/02_processed/labels.parquet')
        # labels = labels.sort_index()

        # check with legacy annotations
        metadata_dir = self.metadata_dir / 'filtered-clustered'
        metadata = pd.concat(
            [
                pd.read_parquet(i, engine="fastparquet")
                for i in metadata_dir.glob("*.parquet")
            ]
        )
        assert not metadata.isna().all().any()

        metadata = metadata[metadata.exclude == False]
        metadata = metadata.sort_index()
        # assert labels.index.equals(metadata.index)

        # load the actual data we need for the mask generation
        metadata_dir = self.metadata_dir / 'filtered-annotated'
        annotated = pd.concat(
            [
                pd.read_parquet(i, engine="fastparquet")
                for i in metadata_dir.glob("*.parquet")
            ]
        )
        assert not annotated.isna().all().any()

        annotated = annotated.sort_index()
        assert metadata.index.equals(annotated.index)

        masks_dir = self.raw_dir / 'masks' / 'deepcell'
        save_dir = self.masks_dir / 'annotated'
        save_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, grp_data in annotated.groupby('sample_id'):
            mask_deepcell = tifffile.imread(masks_dir / f'{sample_id}.tiff')
            object_ids = grp_data.index.get_level_values('object_id').values

            valid_ids = np.isin(mask_deepcell, object_ids)
            mask = np.where(valid_ids, mask_deepcell, 0)
            tifffile.imwrite(save_dir / f'{sample_id}.tiff', mask)

    def create_metadata(self):
        logger.info("Creating metadata for clustered cells")

        cluster_ids = pd.read_parquet(
            self.raw_dir / 'clustering/2024-07-17_13-56/cluster-ids.parquet')

        with open(self.raw_dir / 'clustering/2024-07-17_13-56/cluster-annotations.yaml') as f:
            cluster_annos = yaml.safe_load(f)

        data = cluster_ids.set_index('id') 
        records = []
        for (idx, row) in data.iterrows():
            ids = list(filter(lambda x: x != None, row))
            assert len(ids) > 0
            id = ids[-1]
            records.append({'index': idx, 'cluster_id': id})

        records = pd.DataFrame(records)

        cluster_annos = pd.DataFrame(cluster_annos)
        cluster_annos = cluster_annos.rename(columns={'id': 'cluster_id'})
        records = pd.merge(records, cluster_annos, on='cluster_id', how='left')
        assert not records.name.isna().any()

        records.loc[:, 'exclude'] = [False if pd.isna(i) else i for i in records['exclude']]

        object_ids = records['index'].str.split('_').str[-1:].str.join('_')
        sample_ids = records['index'].str.split('_').str[:-1].str.join('_')
        records['sample_id'] = sample_ids
        records['object_id'] = object_ids.astype(int)
        records = records.set_index(['sample_id', 'object_id'])
        records = records.drop(columns=['index', 'subclustered'])

        records = records.rename(columns=dict(name='label'))
        records.columns = records.columns.map(tidy_name)
        records = records.convert_dtypes()

        # %%
        version_name = self.get_version_name(image_version="filtered", mask_version="clustered")
        save_dir = self.metadata_dir / version_name
        if save_dir.exists():
            return
        else:
            save_dir.mkdir(parents=True, exist_ok=True)

            for grp_name, grp_data in records.groupby("sample_id"):
                path = save_dir / f"{grp_name}.parquet"
                grp_data.to_parquet(path, engine='fastparquet')

        # %% HERE WE REMOVE CELLS THAT ARE EXCLUDE == True
        logger.info("Creating metadata for annotated cells")

        records = records[records['exclude'] == False]
        records = records.convert_dtypes()

        version_name = self.get_version_name(image_version="filtered", mask_version="annotated")
        save_dir = self.metadata_dir / version_name
        if save_dir.exists():
            return
        else:
            save_dir.mkdir(parents=True, exist_ok=True)

            for grp_name, grp_data in records.groupby("sample_id"):
                path = save_dir / f"{grp_name}.parquet"
                grp_data.to_parquet(path, engine='fastparquet')

    def create_panel(self):
        from ai4bmr_core.utils.tidy import tidy_name
        panel_paths = [p for p in self.images_dir.rglob("panel.csv")]
        for path in panel_paths:
            panel = pd.read_csv(path)
            panel.index.name = 'channel_index'
            panel = panel.rename(columns=dict(name='target'))
            panel.loc[:, 'target'] = panel.target.map(tidy_name)
            panel.to_parquet(path.with_suffix(".parquet"), engine='fastparquet')

    @property
    def name(self):
        return "BLCa"

    @property
    def id(self):
        return "BLCa"

    @property
    def doi(self):
        return "unpublished"
