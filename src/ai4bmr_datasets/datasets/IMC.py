from pathlib import Path

import pandas as pd

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

    def load(self):
        if not self.clinical_metadata_path.exists():
            self.create_clinical_metadata()
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

        clinical_metadata, intensity = clinical_metadata.align(intensity, join='inner', axis=0)
        assert intensity.notna().any().any()

        spatial = pd.read_parquet(self.spatial_dir)
        spatial = process(spatial)
        clinical_metadata, spatial = clinical_metadata.align(spatial, join='inner', axis=0)
        assert spatial.notna().any().any()

        assert spatial.index.equals(intensity.index)
        assert spatial.index.equals(clinical_metadata.index)

        # annotations
        if self.annotations_path.exists():
            metadata = pd.read_parquet(self.annotations_path)
            index_to_cols = list(set(metadata.index.names) - {'sample_name', 'object_id'})
            metadata = metadata.reset_index(index_to_cols, drop=True)

            # NOTE: legacy because we do not have group_id in the annotations for PCa
            if 'group_id' not in metadata.columns:
                group_id = metadata.groupby(['parent_group_name', 'group_name']).ngroup()
                metadata = metadata.assign(group_id=group_id)

            # NOTE: legacy to harmonize with BLCa data that does not provide this information at this point
            if 'parent_group_name' in metadata.columns:
                metadata = metadata.drop(columns=['parent_group_name'])

            clinical_metadata, metadata = clinical_metadata.align(metadata, join='inner', axis=0)
        else:
            metadata = None

        assert spatial.index.equals(metadata.index)

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
                    clinical_metadata=clinical_metadata, metadata=metadata)


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

        assert roi_match_blockId.PAT_ID.isna().sum() == 4
        roi_match_blockId = roi_match_blockId.dropna(subset=['PAT_ID'])

        path_tma_tidy = raw_dir / 'TMA_tidy.txt'
        tma_tidy = pd.read_csv(path_tma_tidy, sep='\t', dtype=str)
        tma_tidy = tma_tidy.drop(columns=['DATE_OF_BIRTH', 'INITIALS'])

        tma_tidy.loc[:, 'PAT_ID'] = tma_tidy.PAT_ID.astype(str).str.strip()

        # NOTE: the `PATIENT_ID` of the first patient is NA for no obvious reason. We assign 0 to it.
        assert tma_tidy.PATIENT_ID.isna().sum() == 1
        assert (tma_tidy.PATIENT_ID == 0).sum() == 0  # we ensure the id=0 is not used already
        assert (tma_tidy.PATIENT_ID == '0').sum() == 0  # we ensure the id=0 is not used already
        tma_tidy.loc[:, 'PATIENT_ID'] = tma_tidy.PATIENT_ID.fillna('0')
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
        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path)


class BLCa(IMCDataset):

    def create_clinical_metadata(self):
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
