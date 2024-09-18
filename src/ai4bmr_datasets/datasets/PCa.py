from pathlib import Path

import pandas as pd

from .BaseDataset import BaseDataset


class PCa(BaseDataset):

    def __init__(self,
                 base_dir: Path = Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/remotes/unibe/data/pca')):
        super().__init__()

        self.base_dir = base_dir

        self.mask_version = 'deepcell_cluster_clean'
        self.masks_dir = self.base_dir / 'masks'

        self.image_version = 'compensated'
        self.images_dir = self.base_dir / 'images' / self.image_version
        self.panel_path = self.images_dir / 'panel.csv'

        self.object_labels_path = self.base_dir / 'clustering' / '2024-07-17_13-48' / 'object-labels.parquet'

        self.measurement_version = f'i_{self.image_version}_m_{self.mask_version}'
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
        intensity.index = intensity.index.reorder_levels(clinical_metadata.index.names)
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
        patient_ids = roi_match_blockId['Original block number'].str.split('_').str[0].str.strip()
        sample_ids = roi_match_blockId['description'].str.split('_').str[2].str.strip()
        roi_match_blockId = roi_match_blockId.assign(SAMPLE_ID=sample_ids, PAT_ID=patient_ids)

        assert roi_match_blockId.PAT_ID.isna().sum() == 4
        assert roi_match_blockId.SAMPLE_ID.isna().sum() == 4
        roi_match_blockId = roi_match_blockId.dropna(subset=['PAT_ID'])
        assert roi_match_blockId.SAMPLE_ID.isna().sum() == 0 and roi_match_blockId.PAT_ID.isna().sum() == 0

        path_tma_tidy = base_dir / 'clinical_metadata' / 'TMA_tidy.txt'
        tma_tidy = pd.read_csv(path_tma_tidy, sep='\t', dtype=str)
        tma_tidy = tma_tidy.drop(columns=['DATE_OF_BIRTH', 'INITIALS'])

        # sample_ids_vars = ['UNIQUE_TMA_SAMPLE_ID_1', 'UNIQUE_TMA_SAMPLE_ID_2',
        #                    'UNIQUE_TMA_SAMPLE_ID_3', 'UNIQUE_TMA_SAMPLE_ID_4']
        # id_vars = set(tma_tidy.columns) - set(sample_ids_vars)
        # tma_tidy = tma_tidy.melt(id_vars=id_vars, value_vars=sample_ids_vars, value_name='SAMPLE_ID')
        # tma_tidy = tma_tidy.dropna(subset=['SAMPLE_ID'])
        # tma_tidy.loc[:, 'SAMPLE_ID'] = tma_tidy.SAMPLE_ID.astype(int).astype(str).str.strip()
        tma_tidy.loc[:, 'PAT_ID'] = tma_tidy.PAT_ID.astype(str).str.strip()

        # date_of_surgery = pd.to_datetime(tma_tidy['DATE_OF_SURGERY')
        # last_fu = tma_tidy['LAST_FU']
        # fig, ax = plt.subplots();ax.scatter(date_of_surgery[mask], last_fu[mask]);ax.scatter(date_of_surgery[~mask], last_fu[~mask]);plt.plot([d1, d2], [100, 100 - (d2-d1).days / 30.44]);plt.show()

        # set(roi_match_blockId.SAMPLE_ID) - set(tma_tidy.SAMPLE_ID)
        roi_match_blockId.loc[roi_match_blockId.SAMPLE_ID == '150 - split', 'SAMPLE_ID'] = '150'
        # assert set(roi_match_blockId.SAMPLE_ID) - set(tma_tidy.SAMPLE_ID) == set()

        # roi_ids = set(roi_match_blockId[['PAT_ID', 'SAMPLE_ID']].to_records(index=False).tolist())
        # tma_ids = set(tma_tidy[['PAT_ID', 'SAMPLE_ID']].to_records(index=False).tolist())
        #
        # missing_ids = roi_ids - tma_ids
        # missing_ids = pd.DataFrame(missing_ids, columns=['PAT_ID', 'SAMPLE_ID']).sort_values(['PAT_ID', 'SAMPLE_ID'])

        # set(roi_match_blockId.PAT_ID) - set(tma_tidy.PAT_ID)
        # len(set(tma_tidy.PAT_ID) - set(roi_match_blockId.PAT_ID))
        # set(tma_tidy.PAT_ID) - set(roi_match_blockId.PAT_ID)

        assert tma_tidy['AGE_AT_SURGERY'].isna().any() == False

        metadata = pd.merge(roi_match_blockId, tma_tidy, how='left')
        metadata = metadata.sort_values(['PAT_ID'])

        assert metadata.PAT_ID.isna().sum() == 0
        assert metadata.SAMPLE_ID.isna().sum() == 0

        # NOTE: 3 ROIs from 2 patients have no meta data
        assert metadata['AGE_AT_SURGERY'].isna().sum() == 3
        assert metadata['PAT_ID'][metadata['AGE_AT_SURGERY'].isna()].nunique() == 2

        # %%
        labels = pd.read_parquet(self.object_labels_path)

        sample_names = [Path(i).stem for i in metadata.file_name_napari]
        metadata = metadata.assign(sample_name=sample_names)
        metadata = metadata.set_index('sample_name').join(labels, how='inner')
        assert len(metadata) == len(labels)

        # %%
        metadata.to_parquet(self.clinical_metadata_path)

