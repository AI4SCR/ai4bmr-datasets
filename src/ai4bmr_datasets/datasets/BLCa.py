from pathlib import Path

import pandas as pd

from .BaseDataset import BaseDataset


class BLCa(BaseDataset):

    def __init__(self,
                 base_dir: Path = Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/remotes/unibe/data/blca')):
        super().__init__()

        self.base_dir = base_dir

        self.mask_version = 'deepcell_cluster_clean'
        self.masks_dir = self.base_dir / 'masks'

        self.image_version = 'compensated'
        self.images_dir = self.base_dir / 'images' / self.image_version
        self.panel_path = self.images_dir / 'panel.csv'

        self.object_labels_path = self.base_dir / 'clustering/2024-07-17_13-56/object-labels.parquet'

        self.measurement_version = f'i_{self.image_version}_m_{self.mask_version}'
        self.measurements_dir = self.base_dir / 'measurements' / self.measurement_version
        self.intensity_dir = self.measurements_dir / 'intensity'
        self.spatial_dir = self.measurements_dir / 'spatial'

        self.clinical_metadata_path = self.base_dir / 'metadata/metadata_full.parquet'

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

        metadata_dir = base_dir / 'metadata'
        metadata = pd.read_csv(metadata_dir / 'clinical_metadata.csv').drop(columns='Unnamed: 0')

        labels = pd.read_parquet(self.object_labels_path)

        mcd_metadata_paths = list((base_dir / 'mcd_metadata').glob('**/*.txt'))
        mcd_metadata = pd.concat(pd.read_csv(i) for i in mcd_metadata_paths)

        # %%
        # NOTE: we drop the rows for cases that were excluded
        metadata = metadata[~metadata['Nr old'].isna()]

        # NOTE: remove test runs
        mcd_metadata = mcd_metadata[~mcd_metadata.Description.str.startswith('LP')]

        # %%
        set(metadata.sample_id) - set(mcd_metadata.Description)
        set(mcd_metadata.Description) - set(metadata.sample_id)

        # %% NOTE: rename acquisitions that were split
        mcd_metadata = mcd_metadata.assign(sample_id=mcd_metadata.Description)
        mcd_metadata.loc[mcd_metadata.Description.str.startswith('1C2b'), 'sample_id'] = '1C2b'
        mcd_metadata.loc[mcd_metadata.Description.str.startswith('1C3i'), 'sample_id'] = '1C3i'

        # %% create sample_name
        mcd_metadata = mcd_metadata.assign(sample_name=mcd_metadata.file_name_napari.map(lambda x: Path(x).stem))

        # %% NOTE: these are the images from patients that were excluded, as indicated in `Analyses_cohort 2.xlsx`
        samples_with_no_metadata = set(mcd_metadata.sample_id) - set(metadata.sample_id)
        pd.Series(list(samples_with_no_metadata)).to_csv(base_dir / 'metadata' / 'samples_with_no_metadata.csv')

        assert samples_with_no_metadata == {'1A4m', '1A4n', '1A6k', '1A6l', '1A7h', '1A8o', '1A8p', '1C4a', '1C4b'}
        assert set(mcd_metadata.sample_id) - set(metadata.sample_id) == samples_with_no_metadata

        removed_sample_names = set(mcd_metadata[mcd_metadata.sample_id.isin(samples_with_no_metadata)].sample_name)
        mcd_metadata = mcd_metadata[~mcd_metadata.sample_id.isin(samples_with_no_metadata)]

        assert set(mcd_metadata.sample_id) - set(metadata.sample_id) == set()

        # %%
        set(metadata.sample_id) - set(mcd_metadata.sample_id)
        len(set(metadata.sample_id) - set(mcd_metadata.sample_id))
        metadata_full = pd.merge(metadata, mcd_metadata, left_on='sample_id', right_on='sample_id', how='outer')

        col_order = ['sample_id', 'patient_id', 'type', 'Nr old', 'file_name', 'Description']
        cols = list(set(metadata_full.columns) - set(col_order))
        col_order = col_order + cols
        metadata_full = metadata_full[col_order]

        # %%
        samples_with_no_acquisition = metadata_full[~metadata_full.file_name_napari.notna()]
        samples_with_no_acquisition[['sample_id', 'patient_id']].to_csv(
            base_dir / 'metadata' / 'samples_with_no_acquisition.csv')

        metadata_full = metadata_full[metadata_full.file_name_napari.notna()]

        # %% add label info
        metadata_full = metadata_full.set_index('sample_name').join(labels, how='inner')

        # check that these are the same sample names that were removed form the mcd_metadata
        assert set(labels.index.get_level_values('sample_name')) - set(
            metadata_full.index.get_level_values('sample_name')) == removed_sample_names

        # %%
        cols = ['sample_id', 'patient_id', 'type', 'Nr old', 'file_name', 'Description', 'file_name_napari']
        cols = cols + list(set(metadata_full.columns) - set(cols))
        metadata_full = metadata_full[cols]
        metadata_full.to_parquet(self.clinical_metadata_path)

