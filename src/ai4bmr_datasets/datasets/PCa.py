from pathlib import Path

import pandas as pd

from .BaseDataset import BaseDataset


class PCa(BaseDataset):

    def __init__(self,
                 base_dir: Path = Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/remotes/unibe/data/pca'),
                 mask_version: str = 'deepcell_cluster_clean',
                 image_version: str = 'compensated',
                 measurement_version: str = None):
        super().__init__()

        self.base_dir = base_dir

        self.mask_version = mask_version
        self.masks_dir = self.base_dir / 'masks'

        self.image_version = image_version
        self.images_dir = self.base_dir / 'images' / self.image_version
        self.panel_path = self.images_dir / 'panel.csv'

        self.object_labels_path = self.base_dir / 'clustering' / '2024-07-17_13-48' / 'object-labels.parquet'

        if measurement_version is None:
            self.measurement_version = f'i_{self.image_version}_m_{self.mask_version}'
        else:
            self.measurement_version = measurement_version
        self.measurements_dir = self.base_dir / 'measurements' / self.measurement_version
        self.intensity_dir = self.measurements_dir / 'intensity'
        self.spatial_dir = self.measurements_dir / 'spatial'

        self.clinical_metadata_path = self.base_dir / 'metadata' / 'metadata_full.parquet'

    def load(self):
        clinical_metadata = pd.read_parquet(self.clinical_metadata_path)

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
        # # TODO: check if this is needed if we use align later
        # if set(clinical_metadata.index.names) != set(intensity.index.names):
        #     assert set(clinical_metadata.index.names) <= set(intensity.index.names)
        # else:
        #     intensity.index = intensity.index.reorder_levels(clinical_metadata.index.names)
        clinical_metadata, intensity = clinical_metadata.align(intensity, join='inner', axis=0)
        assert intensity.notna().any().any()

        spatial = pd.read_parquet(self.spatial_dir)
        spatial = process(spatial)
        clinical_metadata, spatial = clinical_metadata.align(spatial, join='inner', axis=0)
        assert spatial.notna().any().any()

        assert spatial.index.equals(intensity.index)
        assert spatial.index.equals(clinical_metadata.index)

        return dict(intensity=intensity, spatial=spatial, clinical_metadata=clinical_metadata)

    def create_metadata(self):
        """
        This computes the metadata for each object in the dataset. The results can still contain objects that have been
        removed after the clustering, i.e. all elements that where used in the clustering are included in the metadata.
        """

        base_dir = self.base_dir

        # %% read metadata
        path_clinical_metadata = base_dir / 'clinical_metadata'

        filename = 'ROI_matching_blockID.xlsx'
        roi_match_blockId = pd.read_excel(path_clinical_metadata / filename, dtype=str)
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

        assert roi_match_blockId.PAT_ID.isna().sum() == 4
        roi_match_blockId = roi_match_blockId.dropna(subset=['PAT_ID'])

        path_tma_tidy = base_dir / 'clinical_metadata' / 'TMA_tidy.txt'
        tma_tidy = pd.read_csv(path_tma_tidy, sep='\t', dtype=str)
        tma_tidy = tma_tidy.drop(columns=['DATE_OF_BIRTH', 'INITIALS'])

        tma_tidy.loc[:, 'PAT_ID'] = tma_tidy.PAT_ID.astype(str).str.strip()

        # NOTE: the `PATIENT_ID` of the first patient is NA for no obvious reason. We assign 0 to it.
        assert tma_tidy.PATIENT_ID.isna().sum() == 1
        assert (tma_tidy.PATIENT_ID == 0).sum() == 0  # we ensure the id=0 is not used already
        assert (tma_tidy.PATIENT_ID == '0').sum() == 0  # we ensure the id=0 is not used already
        tma_tidy.loc[:, 'PATIENT_ID']= tma_tidy.PATIENT_ID.fillna('0')
        assert tma_tidy.PATIENT_ID.isna().sum() == 0

        roi_match_blockId.loc[roi_match_blockId.tma_sample_id == '150 - split', 'tma_sample_id'] = '150'
        assert roi_match_blockId.tma_sample_id.str.contains('split').sum() == 0
        assert tma_tidy['AGE_AT_SURGERY'].isna().any() == False

        metadata = pd.merge(roi_match_blockId, tma_tidy, how='left')
        metadata = metadata.sort_values(['PAT_ID'])

        # NOTE: all patients in the roi_match_block are present in the metadata
        assert metadata.PAT_ID.isna().sum() == 0
        # NOTE: 3 ROIs from 2 patients have no meta data
        assert metadata.PATIENT_ID.isna().sum() == 3
        assert metadata[metadata.PATIENT_ID.isna()].PAT_ID.nunique() == 2

        # NOTE: for 3 ROIs we do not have patient metadata
        #   thus from the 545 images we have, only 542 are used in downstream analysis
        metadata = metadata[metadata.PATIENT_ID.notna()]

        # %%
        sample_names = [Path(i).stem for i in metadata.file_name_napari]
        metadata = metadata.assign(sample_name=sample_names)
        metadata = metadata.set_index('sample_name')

        # %%
        if self.object_labels_path.exists():
            labels = pd.read_parquet(self.object_labels_path)
            metadata = metadata.join(labels, how='inner')
        # assert len(metadata) == len(labels)

        # %%
        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path)


