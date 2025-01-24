from abc import abstractmethod
from pathlib import Path

import pandas as pd
from ai4bmr_core.utils.tidy import tidy_name

from .BaseDataset import BaseDataset
from ..datamodels.Image import Image, Mask


class IMCDataset(BaseDataset):

    def __init__(self,
                 base_dir: Path,
                 mask_version: str = 'cleaned',
                 image_version: str = 'filtered',
                 measurement_version: str = 'for_clustering'):
        super().__init__()

        self.base_dir = base_dir

        self.mask_version = mask_version
        self.masks_dir = self.base_dir / 'masks' / mask_version

        self.image_version = image_version
        self.images_dir = self.base_dir / 'images' / self.image_version

        self.panel_path = self.images_dir / 'panel.csv'

        self.annotations_path = self.base_dir / 'clustering' / 'annotations.parquet'

        self.measurement_version = measurement_version
        self.measurements_dir = self.base_dir / 'measurements' / self.measurement_version
        self.intensity_dir = self.measurements_dir / 'intensity'
        self.spatial_dir = self.measurements_dir / 'spatial'

        self.clinical_metadata_path = self.base_dir / 'metadata' / '02_processed' / 'clinical_metadata.parquet'
        self.mcd_metadata_path = self.base_dir / 'metadata' / '02_processed' / 'mcd_metadata.parquet'
        self.labels_path = self.base_dir / 'metadata' / '02_processed' / 'labels.parquet'

    def load(self):
        if not self.clinical_metadata_path.exists():
            self.create_clinical_metadata()
        clinical_metadata = pd.read_parquet(self.clinical_metadata_path)
        metadata = clinical_metadata

        panel = pd.read_csv(self.panel_path, index_col='channel')

        def process(df: pd.DataFrame):
            df.index = df.index.rename({'id': 'object_id'})

            sample_names = df.index.get_level_values('mask_path').map(lambda x: Path(x).stem)
            df['sample_name'] = sample_names
            df = df.set_index('sample_name', append=True)

            df.index = df.index.droplevel(list(set(df.index.names) - {'object_id', 'sample_name'}))
            return df

        intensity = pd.read_parquet(self.intensity_dir)
        intensity = process(intensity)
        intensity = intensity.rename(columns=panel['name'].to_dict())

        intensity, clinical_metadata = intensity.align(clinical_metadata, join='inner', axis=0)
        assert intensity.notna().any().any()

        spatial = pd.read_parquet(self.spatial_dir)
        spatial = process(spatial)
        intensity, spatial = intensity.align(spatial, join='inner', axis=0)
        assert spatial.notna().any().any()
        assert spatial.index.equals(clinical_metadata.index)

        # labels
        if self.labels_path.exists():
            labels = pd.read_parquet(self.labels_path)
            index_levels_to_drop = list(set(labels.index.names) - {'object_id', 'sample_name'})
            labels.index = labels.index.droplevel(index_levels_to_drop)
            intensity, labels = intensity.align(labels, join='right', axis=0)
            assert labels.isna().any().any() == False
        else:
            labels = None

        # NOTE: we need to align here again because not all cells in the mask are still in `labels`
        intensity, spatial = intensity.align(spatial, join='left', axis=0)
        assert spatial.index.equals(intensity.index)

        masks = {}
        for mask_path in self.masks_dir.glob('*.tiff'):
            # TODO: discuss how to store metadata for the masks, this is essentially the phenotyping aka metadata above
            # metadata_path = mask_path.with_suffix('.parquet')
            # assert metadata_path.exists()
            metadata_path = None
            masks[mask_path.stem] = Mask(id=mask_path.stem, data_path=mask_path, metadata_path=metadata_path)

        images = {}
        for image_path in self.images_dir.glob('*.tiff'):
            images[image_path.stem] = Image(id=image_path.stem, data_path=image_path,
                                            metadata_path=self.panel_path)

        assert set(images) == set(masks)

        return dict(intensity=intensity, spatial=spatial,
                    panel=panel, images=images, masks=masks,
                    clinical_metadata=clinical_metadata, metadata=metadata, labels=labels)

    @abstractmethod
    def create_clinical_metadata(self):
        pass


class PCa(IMCDataset):

    def create_clinical_metadata(self):
        """
        This computes the metadata for each object in the dataset. The results can still contain objects that have been
        removed after the clustering, i.e. all elements that where used in the clustering are included in the metadata.
        """

        raw_dir = self.base_dir / 'metadata' / '01_raw'

        # %% read metadata
        filename = 'ROI_matching_blockID.xlsx'
        roi_match_blockId = pd.read_excel(raw_dir / filename, dtype=str)
        roi_match_blockId = roi_match_blockId.drop(columns=['patient level code'])
        roi_match_blockId = roi_match_blockId.rename(
            columns={'Description (slidecode_coordinates_uniquecoreID)': 'description'})

        slide_code = roi_match_blockId['description'].str.split('_').str[0].str.strip()
        tma_coordinates = roi_match_blockId['description'].str.split('_').str[1].str.strip()
        tma_sample_ids = roi_match_blockId['description'].str.split('_').str[2].str.strip()
        roi_match_blockId = roi_match_blockId.assign(tma_sample_id=tma_sample_ids,
                                                     slide_code=slide_code,
                                                     tma_coordinates=tma_coordinates)

        patient_ids = roi_match_blockId['Original block number'].str.split('_').str[0].str.strip()
        roi_match_blockId = roi_match_blockId.assign(PAT_ID=patient_ids)

        # NOTE: these are the LP samples (i.e. test runs)
        assert roi_match_blockId.PAT_ID.isna().sum() == 4
        roi_match_blockId = roi_match_blockId.dropna(subset=['PAT_ID'])

        path_tma_tidy = raw_dir / 'TMA_tidy.txt'
        tma_tidy = pd.read_csv(path_tma_tidy, sep='\t', dtype=str)
        tma_tidy = tma_tidy.drop(columns=['DATE_OF_BIRTH', 'INITIALS'])

        tma_tidy.loc[:, 'PAT_ID'] = tma_tidy.PAT_ID.astype(str).str.strip()

        # NOTE: the `PATIENT_ID` of the first patient is NA for no obvious reason.
        assert tma_tidy.PATIENT_ID.isna().sum() == 1

        roi_match_blockId.loc[roi_match_blockId.tma_sample_id == '150 - split', 'tma_sample_id'] = '150_1'
        assert roi_match_blockId.tma_sample_id.str.contains('split').sum() == 0
        assert tma_tidy['AGE_AT_SURGERY'].isna().any() == False

        metadata = pd.merge(roi_match_blockId, tma_tidy, how='left')
        metadata = metadata.sort_values(['PAT_ID'])

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
            id_ = metadata['tma_sample_id'].values[i]
            ids_ = metadata[['UNIQUE_TMA_SAMPLE_ID_1', 'UNIQUE_TMA_SAMPLE_ID_2', 'UNIQUE_TMA_SAMPLE_ID_3',
                             'UNIQUE_TMA_SAMPLE_ID_4']].iloc[i]
            ids_ = set(ids_)
            if id_ not in ids_:
                pass
                print(id_, metadata.PAT_ID.iloc[i])

        filter_ = metadata['Original block number'] == metadata['DONOR_BLOCK_ID']
        mismatches = metadata[['PAT_ID', 'Gleason pattern (TMA core)', 'Original block number', 'DONOR_BLOCK_ID']][
            ~filter_]

        # %% tidy
        metadata = metadata.astype(str)

        metadata.columns = [tidy_name(c, split_camel_case=False) for c in metadata.columns]

        metadata['date_of_surgery'] = pd.to_datetime(metadata['date_of_surgery'])

        metadata['age_at_surgery'] = metadata['age_at_surgery'].astype(int)

        mapping = {
            'High Gleason pattern': 'high',
            'Low Gleason pattern': 'low',
            'Only 1 Gleason pattern': 'only_1',
            'unkown': 'unknown'
        }
        metadata['gleason_pattern_tma_core'] = metadata.gleason_pattern_tma_core.map(mapping)

        metadata['psa_at_surgery'] = metadata.psa_at_surgery.astype(float)
        assert metadata['psa_at_surgery'].isna().sum() == 0

        metadata['last_fu'] = metadata['last_fu'].astype(int)
        assert metadata.last_fu.isna().sum() == 0

        metadata['cause_of_death'] = metadata.cause_of_death.map({'0': 'alive', '1': 'non-PCa_death', '2': 'PCa_death'})
        assert metadata.cause_of_death.isna().sum() == 0

        metadata['os_status'] = metadata.os_status.map({'0': 'alive', '1': 'dead'})
        assert metadata.os_status.isna().sum() == 0

        metadata['psa_progr'] = metadata.psa_progr.astype(int)
        assert metadata.psa_progr.isna().sum() == 0

        metadata['psa_progr_time'] = metadata.psa_progr_time.astype(int)
        assert metadata.psa_progr_time.isna().sum() == 0

        metadata['clinical_progr'] = metadata.clinical_progr.astype(int)
        assert metadata.clinical_progr.isna().sum() == 0

        metadata['clinical_progr_time'] = metadata.clinical_progr_time.astype(int)
        assert metadata.clinical_progr_time.isna().sum() == 0

        metadata['disease_progr'] = metadata.disease_progr.astype(int)
        assert metadata.disease_progr.isna().sum() == 0

        metadata['disease_progr_time'] = metadata.disease_progr_time.astype(int)
        assert metadata.disease_progr_time.isna().sum() == 0

        metadata['recurrence_loc'] = metadata.recurrence_loc.map({'0': 'no_recurrence', '1': 'local', '2': 'metastasis',
                                                                  '3': 'local_and_metastasis'})
        assert metadata.recurrence_loc.isna().sum() == 0

        # NOTE: we do not have the mapping at this point
        # metadata['cgrading_biopsy'] = metadata.cgrading_biopsy.astype(int)

        metadata['cgs_pat_1'] = metadata.cgs_pat_1.astype(int)
        assert metadata.cgs_pat_1.isna().sum() == 0

        metadata['cgs_pat_2'] = metadata.cgs_pat_2.astype(int)
        assert metadata.cgs_pat_2.isna().sum() == 0

        metadata['cgs_score'] = metadata.cgs_score.astype(int)
        assert metadata.cgs_score.isna().sum() == 0

        # metadata['gs_grp'] = metadata.gs_grp.map({'1': '<=6', '2': '3+4=7', '3': '4+3=7', '4': '8', '5': '9/10'})
        # assert metadata.gs_grp.isna().sum() == 0

        metadata['pgs_score'] = metadata.pgs_score.astype(int)
        assert metadata.pgs_score.isna().sum() == 0

        metadata['ln_status'] = metadata.ln_status.astype(int)
        assert metadata.ln_status.isna().sum() == 0

        metadata['surgical_margin_status'] = metadata.surgical_margin_status.astype(float)
        assert metadata.surgical_margin_status.astype(float).isna().sum() == 87

        metadata['d_amico_risk'] = metadata['d_amico'].map({'Intermediate Risk': 'intermediate', 'High Risk': 'high'})
        metadata = metadata.drop('d_amico', axis=1)

        metadata = metadata.convert_dtypes()

        # %%
        # vals = metadata.cause_of_death.astype(int).values
        # cats = sorted(set(vals))
        # metadata['cause_of_death'] = pd.Categorical(vals, categories=cats, ordered=True)
        # assert metadata.cause_of_death.isna().sum() == 0

        # %%
        sample_names = [Path(i).stem for i in metadata.file_name_napari]
        metadata = metadata.assign(sample_name=sample_names)
        metadata = metadata.set_index('sample_name')

        # %%
        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path)


class BLCa(IMCDataset):

    def create_clinical_metadata(self):
        raw_dir = self.base_dir / 'metadata' / '01_raw'
        processed_dir = self.base_dir / 'metadata' / '02_processed'

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
        assert ids.patient_id.value_counts().max() <= 6

        # mcd data
        mcd_metadata = pd.read_parquet(self.mcd_metadata_path)
        # NOTE: rename acquisitions that were split
        mcd_metadata = mcd_metadata.assign(sample_id=mcd_metadata.Description)
        mcd_metadata.loc[mcd_metadata.Description.str.startswith('1C2b'), 'sample_id'] = '1C2b'
        mcd_metadata.loc[mcd_metadata.Description.str.startswith('1C3i'), 'sample_id'] = '1C3i'

        # create sample_name
        mcd_metadata = mcd_metadata.assign(sample_name=mcd_metadata.file_name_napari.map(lambda x: Path(x).stem))

        sample_ids_with_not_in_analysis = set(mcd_metadata.sample_id) - set(ids.sample_id)
        sample_ids_with_no_acquisition = set(ids.sample_id) - set(mcd_metadata.sample_id)

        metadata = pd.merge(ids, mcd_metadata, how="outer")
        metadata[metadata.sample_id.isin(sample_ids_with_no_acquisition)].to_csv(
            processed_dir / 'samples_with_no_acquisition.csv')
        metadata[metadata.sample_id.isin(sample_ids_with_not_in_analysis)].to_csv(
            processed_dir / 'sample_ids_with_not_in_analysis.csv')

        sample_ids_to_remove = sample_ids_with_not_in_analysis.union(sample_ids_with_no_acquisition)
        metadata = metadata[~metadata.sample_id.isin(sample_ids_to_remove)]

        # clinical metadata
        clinical_metadata = pd.read_excel(raw_dir / "metadata_NAC_TMA_11_08.xlsx", dtype=str)
        clinical_metadata = clinical_metadata.iloc[:59, :]
        clinical_metadata = clinical_metadata.dropna(axis=1, how="all")  # remove NaN columns
        # NOTE: some patients have a `Nr old` but everything else is NaN
        clinical_metadata = clinical_metadata[~clinical_metadata.drop(columns='Nr old').isna().all(axis=1)]
        # NOTE: `Nr. old` are for the matching with TMA and new ones for the genomic profiling
        clinical_metadata = clinical_metadata.assign(patient_id=clinical_metadata["Nr old"].astype(int).astype(str))

        patient_ids_with_no_acquisition = set(clinical_metadata.patient_id) - set(metadata.patient_id)
        patient_ids_with_no_clinical_metadata = set(metadata.patient_id) - set(clinical_metadata.patient_id)

        clinical_metadata[clinical_metadata.patient_id.isin(patient_ids_with_no_acquisition)].to_csv(
            processed_dir / 'patient_ids_with_no_acquisition.csv')
        metadata[metadata.patient_id.isin(patient_ids_with_no_clinical_metadata)].to_csv(
            processed_dir / 'patient_ids_with_no_clinical_metadata.csv')

        patient_ids_to_remove = patient_ids_with_no_acquisition.union(patient_ids_with_no_clinical_metadata)
        clinical_metadata = clinical_metadata[~clinical_metadata.patient_id.isin(patient_ids_to_remove)]
        metadata = metadata[~metadata.patient_id.isin(patient_ids_to_remove)]

        mask = clinical_metadata.columns.str.contains("date", case=False)
        date_cols = clinical_metadata.columns[mask]
        date_cols = date_cols.tolist() + ["Start Chemo", "End Chemo"]

        clinical_metadata.loc[clinical_metadata["End Chemo"] == "31.02.2010", "End Chemo"] = pd.NaT
        for col in date_cols:
            clinical_metadata.loc[:, col] = pd.to_datetime(clinical_metadata[col])

        # metadata
        metadata_full = pd.merge(metadata, clinical_metadata, how="left")
        assert len(metadata) == len(metadata_full)

        # convert to parquet compatible types
        metadata_full = metadata_full.convert_dtypes()
        metadata_full = metadata_full.set_index('sample_name')
        metadata_full.to_parquet(self.clinical_metadata_path)
