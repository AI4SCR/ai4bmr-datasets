import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset
from ai4bmr_datasets.utils import io
from ai4bmr_datasets.utils.download import download_file_map, unzip_recursive
from ai4bmr_datasets.utils.tidy import filter_paths


class Cords2024(BaseIMCDataset):
    name = "Cords2024"

    @property
    def id(self):
        """
        Returns the unique identifier for the Cords2024 dataset.
        """
        return "Cords2024"

    @property
    def doi(self):
        """
        Returns the DOI for the Cords2024 dataset publication.
        """
        return "10.1016/j.ccell.2023.12.021"

    def __init__(self,
                 base_dir: Path | None = None,
                 image_version: str | None = None,
                 mask_version: str | None = None,
                 feature_version: str | None = None,
                 metadata_version: str | None = None,
                 load_intensity: bool = False,
                 load_spatial: bool = False,
                 load_metadata: bool = False,
                 align: bool = False,
                 join: str = "outer",
                 ):
        """
        Initializes the Cords2024 dataset.

        Args:
            base_dir (Path | None): Base directory for the dataset.
            image_version (str | None): Version of the images to load.
            mask_version (str | None): Version of the masks to load.
            feature_version (str | None): Version of the features to load.
            metadata_version (str | None): Version of the metadata to load.
            load_intensity (bool): Whether to load intensity features.
            load_spatial (bool): Whether to load spatial features.
            load_metadata (bool): Whether to load metadata.
            align (bool): Whether to align sample IDs across modalities.
            join (str): Type of join for alignment (e.g., 'outer').
        """

        super().__init__(base_dir=base_dir,
                         image_version=image_version,
                         mask_version=mask_version,
                         feature_version=feature_version,
                         metadata_version=metadata_version,
                         load_intensity=load_intensity,
                         load_spatial=load_spatial,
                         load_metadata=load_metadata,
                         align=align,
                         join=join)

        # raw data paths
        self.raw_clinical_metadata = self.raw_dir / "comp_csv_files" / "cp_csv" / "clinical_data_ROI.csv"
        self.raw_acquisitions_dir = self.raw_dir / "acquisitions"

    def prepare_data(self):
        """
        Prepares the Cords2024 dataset by downloading, processing, and creating
        various data components (clinical metadata, panel, images, masks, metadata).
        """
        self.download()
        self.process_rdata()
        self.process_acquisitions()

        self.create_clinical_metadata()

        self.create_panel()
        self.create_images()
        self.create_masks()
        self.create_metadata()
        self.create_features()

        self.create_annotated()

    def download(self, force: bool = False):
        """
        Downloads the raw data files for the Cords2024 dataset from Zenodo and Google Drive.

        Args:
            force (bool): If True, forces re-download even if files already exist.
        """
        download_dir = self.raw_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "https://zenodo.org/records/7961844/files/2020115_LC_NSCLC_TMA_86_A.mcd.zip?download=1": download_dir
                                                                                                     / "2020115_LC_NSCLC_TMA_86_A.mcd.zip",
            "https://zenodo.org/records/7961844/files/2020117_LC_NSCLC_TMA_86_B.mcd.zip?download=1": download_dir
                                                                                                     / "2020117_LC_NSCLC_TMA_86_B.mcd.zip",
            "https://zenodo.org/records/7961844/files/2020120_LC_NSCLC_TMA_86_C.mcd.zip?download=1": download_dir
                                                                                                     / "2020120_LC_NSCLC_TMA_86_C.mcd.zip",
            "https://zenodo.org/records/7961844/files/20201219_LC_NSCLC_TMA_178_A.mcd.zip?download=1": download_dir
                                                                                                       / "20201219_LC_NSCLC_TMA_178_A.mcd.zip",
            "https://zenodo.org/records/7961844/files/20201222_LC_NSCLC_TMA_178_B_2.mcd.zip?download=1": download_dir
                                                                                                         / "20201222_LC_NSCLC_TMA_178_B_2.mcd.zip",
            "https://zenodo.org/records/7961844/files/20201224_LC_NSCLC_TMA_178_C.mcd.zip?download=1": download_dir
                                                                                                       / "20201224_LC_NSCLC_TMA_178_C.mcd.zip",
            "https://zenodo.org/records/7961844/files/20201226_LC_NSCLC_TMA_175_A.mcd.zip?download=1": download_dir
                                                                                                       / "20201226_LC_NSCLC_TMA_175_A.mcd.zip",
            "https://zenodo.org/records/7961844/files/20201228_LC_NSCLC_TMA_175_B.mcd.zip?download=1": download_dir
                                                                                                       / "20201228_LC_NSCLC_TMA_175_B.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210101_LC_NSCLC_TMA_175_C.mcd.zip?download=1": download_dir
                                                                                                       / "20210101_LC_NSCLC_TMA_175_C.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210104_LC_NSCLC_TMA_176_A.mcd.zip?download=1": download_dir
                                                                                                       / "20210104_LC_NSCLC_TMA_176_A.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210109_LC_NSCLC_TMA_176_C.mcd.zip?download=1": download_dir
                                                                                                       / "20210109_LC_NSCLC_TMA_176_C.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210112_LC_NSCLC_TMA_176_B.mcd.zip?download=1": download_dir
                                                                                                       / "20210112_LC_NSCLC_TMA_176_B.mcd.zip",
            "https://zenodo.org/records/7961844/files/2020121_LC_NSCLC_TMA_87_A.mcd.zip?download=1": download_dir
                                                                                                     / "2020121_LC_NSCLC_TMA_87_A.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210123_LC_NSCLC_TMA_87_B.mcd.zip?download=1": download_dir
                                                                                                      / "20210123_LC_NSCLC_TMA_87_B.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210126_LC_NSCLC_TMA_87_C.mcd.zip?download=1": download_dir
                                                                                                      / "20210126_LC_NSCLC_TMA_87_C.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210129_LC_NSCLC_TMA_88_A.mcd.zip?download=1": download_dir
                                                                                                      / "20210129_LC_NSCLC_TMA_88_A.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210129_LC_NSCLC_TMA_88_B.mcd.zip?download=1": download_dir
                                                                                                      / "20210129_LC_NSCLC_TMA_88_B.mcd.zip",
            "https://zenodo.org/records/7961844/files/20210129_LC_NSCLC_TMA_88_C.mcd.zip?download=1": download_dir
                                                                                                      / "20210129_LC_NSCLC_TMA_88_C.mcd.zip",
            "https://zenodo.org/records/7961844/files/comp_csv_files.zip?download=1": download_dir
                                                                                      / "comp_csv_files.zip",
            "https://zenodo.org/records/7961844/files/Cell%20masks.zip?download=1": download_dir / "cell_masks_zenodo.zip",
            "https://drive.usercontent.google.com/download?id=1wXboMZpkLzk7ZP1Oib1q4NBwvFGK_h7G&confirm=xxx": download_dir
                                                                                                              / "cell_masks_google_drive.zip",
            "https://zenodo.org/records/7961844/files/SingleCellExperiment%20Objects.zip?download=1": download_dir / 'single_cell_experiment_objects.zip'
        }

        download_file_map(file_map, force=force)

        # Extract zip files
        for target_path in file_map.values():
            try:
                unzip_recursive(target_path)
            except Exception as e:
                logger.error(f'Failed to unzip {target_path}: {e}')

        shutil.rmtree(self.raw_dir / '__MACOSX', ignore_errors=True)

        logger.info("✅ Download and extraction completed.")

    def get_sample_id_from_path(self, path):
        """
        Extracts the sample ID from a given file path.

        Args:
            path (str): The file path.

        Returns:
            str: The extracted sample ID.
        """
        path = str(path)
        search = re.search(r"TMA_(\d+)_(\w+)_s0_a(\d+)", (path))
        return '_'.join(search.groups())

    def process_acquisitions(self):
        """
        Processes raw MCD acquisition files to extract and save image data and metadata.

        This method reads .mcd files, extracts individual acquisitions, and saves them as
        TIFF images and JSON metadata files in the raw acquisitions directory.
        """

        from readimc import MCDFile

        acquisitions_dir = self.raw_acquisitions_dir
        acquisitions_dir.mkdir(parents=True, exist_ok=True)

        num_failed_reads = 0

        mcd_paths = [i for i in self.raw_dir.rglob("*.mcd") if i.is_file() and not i.name.startswith(".")]
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

                        image_name = f"{mcd_id}_{acq.id}"  # should be equivalent to 'Tma_ac'
                        metadata_path = acquisitions_dir / f"{image_name}.json"
                        image_path = acquisitions_dir / f"{image_name}.tiff"

                        if image_path.exists():
                            logger.info(f"Skipping image {image_path}. Already exists.")
                            continue

                        try:
                            logger.info(f"Reading acquisition of image {image_name}")
                            img = f.read_acquisition(acq)
                            img = img.astype(np.float32)
                            io.imsave(save_path=image_path, img=img)
                            with open(metadata_path, "w") as f_json:
                                json.dump(item, f_json)
                        except OSError:
                            logger.error(f"Failed to read acquisition! ")
                            num_failed_reads += 1
                            continue

    @classmethod
    def validate_masks(cls):
        """
        Validates the consistency between mask objects and metadata objects.

        This method loads published image, mask, and metadata versions, then checks if the unique
        object IDs in the masks match those in the metadata for each sample.
        Logs mismatches for samples where the object IDs do not align.
        """
        import numpy as np
        version_name = 'published'
        ds = cls(image_version=version_name, mask_version=version_name, metadata_version=version_name,
                 load_metadata=True, load_spatial=False, load_intensity=False)

        mismatches = []
        for sample_id, mask in ds.masks.items():
            uniq_mask_objs = set(map(int, np.unique(mask.data))) - {0}
            meta = self.metadata.loc[sample_id]
            uniq_meta_objs = set(meta.index.astype(int))
            if not uniq_mask_objs == uniq_meta_objs:
                logger.info(f'Mask objects and metadata objects do not match for {sample_id}')
                mismatches.append((sample_id, uniq_meta_objs, uniq_mask_objs))

        for i, (sample_id, uniq_meta_objs, uniq_mask_objs) in enumerate(mismatches):
            logger.info(f"Sample ID: {sample_id}")
            logger.info(
                f"Unique mask objects: {len(uniq_mask_objs)} (min: {min(uniq_mask_objs)}, max: {max(uniq_mask_objs)})")
            logger.info(
                f"Unique metadata objects: {len(uniq_meta_objs)} (min: {min(uniq_meta_objs)}, max: {max(uniq_meta_objs)})")
            if i == 10:
                break

        item = list(filter(lambda x: x[0] == '178_B_71', mismatches))

    def create_panel(self):
        """
        Creates and saves the panel (channel metadata) for the Cords2024 dataset.

        This method reads channel information from raw acquisition JSON files, processes it,
        and saves the consolidated panel as a Parquet file.
        """
        from ai4bmr_datasets.utils.tidy import tidy_name
        import json
        import pandas as pd
        panel = None

        img_metadata_paths = [p for p in self.raw_acquisitions_dir.glob("*.json") if not p.name.startswith('.')]
        for img_metadata_path in img_metadata_paths:
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
        panel.loc[:, 'target'] = panel.target.str.replace('_0', '', regex=True)

        panel["keep"] = keep

        panel.columns = panel.columns.map(tidy_name)
        panel = panel.rename(columns={'channel_names': 'metal_tag',
                                      'channel_labels': 'channel_label',
                                      'metal_names': 'metal_name'})
        panel = panel.convert_dtypes()
        panel["page"] = range(len(panel))
        panel.index.name = "channel_index"

        panel.to_parquet(self.raw_acquisitions_dir / "panel.parquet")

        panel = panel[panel.keep].reset_index(drop=True)
        panel.index.name = "channel_index"
        panel_path = self.get_panel_path("published")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)

    def create_images(self):
        """
        Creates and saves processed image files for the Cords2024 dataset.

        This method reads raw image acquisitions, filters them based on the panel,
        and saves the processed images as TIFF files.
        """
        panel_path = self.get_panel_path("published")
        panel = pd.read_parquet(panel_path)

        acquisitions_dir = self.raw_acquisitions_dir
        acquisition_paths = [p for p in acquisitions_dir.glob("*.tiff") if not p.name.startswith('.')]

        logger.info(f'Creating images...')
        save_dir = self.get_image_version_dir("published")
        save_dir.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(acquisition_paths):
            save_path = save_dir / f"{img_path.name}"

            if save_path.exists():
                logger.info(f"Skipping image {img_path}. Already exists.")
                continue

            img = io.imread(img_path)
            img = img[panel.page.values]
            img = img.astype(np.float32)
            io.imsave(save_path=save_path, img=img)

    def create_masks(self):
        """
        Creates and saves processed mask files for the Cords2024 dataset.

        This method reads raw mask files, processes them (including handling specific
        naming conventions and potential mismatches), and saves the processed masks as TIFF files.
        """
        import tifffile

        raw_masks_dir = self.raw_dir / 'cell_masks_google_drive'
        mask_paths = [p for p in raw_masks_dir.rglob("*.tiff") if not p.name.startswith('.')]

        # note:
        # - 86_B_mask also contains 86_C masks
        # - 176_A_mask the filenames do not match the parent folder name but have TMA_176_B in the name filename
        mask_paths_from_176_A = [p for p in mask_paths if '176_A_mask' in str(p)]
        mask_paths = [p for p in mask_paths if p.parent.parent.name.replace('_mask', '') in p.name]

        mask_paths_with_ids = [(self.get_sample_id_from_path(path), path) for path in mask_paths]
        mask_paths_from_176_A_with_ids = [(self.get_sample_id_from_path(path), path) for path in mask_paths_from_176_A]
        mask_paths_from_176_A_with_ids = [(x[0].replace('176_B', '176_A'), x[1])
                                          for x in mask_paths_from_176_A_with_ids]

        mask_paths_with_ids = mask_paths_with_ids + mask_paths_from_176_A_with_ids

        # note: rename 178_B_2 to 178_B
        mask_paths_with_ids = [(x[0].replace('178_B_2', '178_B'), x[1]) for x in mask_paths_with_ids]

        # mask_paths_with_no_ids = list(filter(lambda x: x[0] is None, mask_paths_with_ids))
        assert (pd.Series([x[0] for x in mask_paths_with_ids]).value_counts() == 1).all()

        save_dir = self.get_mask_version_dir('published')
        save_dir.mkdir(parents=True, exist_ok=True)

        image_ids = [i.stem for i in self.get_image_version_dir('published').glob('*.tiff') if not i.name.startswith('.')]
        image_ids = set(image_ids)

        mask_ids = [i[0] for i in mask_paths_with_ids]
        mask_ids = set(mask_ids)

        # note: for some of the acquisitions that failed to read in we have masks, see `process_acquisitions`
        #   - we have 7 masks more than images
        #   - for '178_B_57' we have a mask but no acquisition
        # mask_ids - image_ids
        assert image_ids <= mask_ids

        save_dir = self.get_mask_version_dir('published')
        save_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.get_image_version_dir('published')
        for sample_id, mask_path in mask_paths_with_ids:

            save_path = save_dir / f"{sample_id}.tiff"
            image_path = image_dir / f"{sample_id}.tiff"

            if save_path.exists():
                logger.info(f"Skipping mask {mask_path}. Already exists.")
                continue

            elif image_path.exists():
                logger.info(f"Copying {mask_path} to {save_path}")
                with tifffile.TiffFile(image_path) as img, tifffile.TiffFile(mask_path) as mask:
                    shape1 = img.pages[0].shape
                    shape2 = mask.pages[0].shape
                    assert shape1 == shape2, f'Image and mask shapes do not match for {sample_id}'
                shutil.copy2(mask_path, save_path)
            else:
                logger.info(f'Skipping mask {mask_path}. No corresponding image found.')

    def create_clinical_metadata(self):
        """
        Creates and saves the clinical metadata for the Cords2024 dataset.

        This method reads raw clinical data, processes it to extract sample IDs,
        and saves the consolidated clinical metadata as a Parquet file.
        It also logs any samples for which clinical metadata or images are missing.
        """
        from ai4bmr_datasets.utils.tidy import tidy_name
        import re

        clinical_metadata = pd.read_csv(self.raw_clinical_metadata)

        assert clinical_metadata["Tma_ac"].duplicated().sum() == 0
        clinical_metadata = clinical_metadata.drop(columns='Unnamed: 0')

        sample_ids = clinical_metadata["Tma_ac"].to_list()
        sample_ids = [re.sub(r"^(\d+)([A-Za-z])", r"\1_\2", s) for s in sample_ids]

        clinical_metadata["sample_id"] = sample_ids
        clinical_metadata = clinical_metadata.set_index("sample_id")
        sample_ids = set(sample_ids)

        # check ids against images
        image_ids = {i.stem for i in self.get_image_version_dir('published').glob("*.tiff")}
        no_clinical_metadata = image_ids - sample_ids
        logger.info(
            f"No clinical metadata for {len(no_clinical_metadata)} ROIs found: {', '.join(no_clinical_metadata)}")
        no_image_ids = sample_ids - image_ids
        logger.info(f"No images for {len(no_image_ids)} patients found: {', '.join(no_image_ids)}")

        # ids = sample_ids.intersection(image_ids)
        # clinical_metadata = clinical_metadata.loc[list(ids)]
        clinical_metadata.columns = clinical_metadata.columns.map(tidy_name)

        # ['x', 'roi_xy', 'ac_id', 'roi_id', 'tma_x', 'tma_ac', 'patient_nr',
        #  'tma_y', 'x_spots', 'dx_name', 'x_y_localisation', 'age', 'gender',
        #  'typ', 'grade', 'size', 'vessel', 'pleura', 't_new', 'n', 'm_new',
        #  'stage', 'r', 'chemo', 'radio', 'chemo3', 'radio4', 'relapse', 'chemo5',
        #  'radio6', 'dfs', 'ev_o', 'os', 'smok', 'nikotin', 'roi', 'patient_id',
        #  'ln_met', 'dist_met', 'neo_adj']
        clinical_metadata['typ'].unique()

        clinical_metadata = clinical_metadata.assign(is_control=clinical_metadata.patient_id == 'Control')
        clinical_metadata = clinical_metadata.convert_dtypes()

        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        clinical_metadata.to_parquet(self.clinical_metadata_path, engine='fastparquet')

    def create_metadata(self):
        """
        Creates and saves processed metadata (cell types) for the Cords2024 dataset.

        This method reads raw single-cell experiment metadata, extracts sample and object IDs,
        and saves the processed metadata as Parquet files, grouped by sample ID.
        """
        path = self.raw_dir / 'single_cell_experiment_objects/SingleCellExperiment Objects/cell_types.parquet'
        metadata = pd.read_parquet(path)

        ids = metadata['object_id'].str.split('_')
        sample_id = ids.str[:-1].str.join('_')
        object_id = ids.str[-1]

        metadata['sample_id'] = sample_id
        metadata['object_id'] = object_id

        metadata = metadata.convert_dtypes()
        metadata = metadata.set_index(['sample_id', 'object_id'])

        version_name = 'published'
        save_dir = self.metadata_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in metadata.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_features_spatial(self):
        """
        Creates and saves processed spatial features for the Cords2024 dataset.

        This method reads raw spatial data, extracts sample and object IDs,
        applies tidy naming conventions, and saves the processed spatial features
        as Parquet files, grouped by sample ID.
        """
        logger.info('Creating spatial features')

        from ai4bmr_datasets.utils.tidy import tidy_name
        path = self.raw_dir / 'single_cell_experiment_objects/SingleCellExperiment Objects/spatial.parquet'
        spatial = pd.read_parquet(path)

        ids = spatial['object_id'].str.split('_')
        sample_id = ids.str[:-1].str.join('_')
        object_id = ids.str[-1]

        spatial['sample_id'] = sample_id
        spatial['object_id'] = object_id

        spatial.columns = spatial.columns.map(tidy_name)
        spatial = spatial.convert_dtypes()
        spatial = spatial.set_index(['sample_id', 'object_id'])

        version_name = 'published'
        save_dir = self.spatial_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in spatial.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_features_intensity(self):
        """
        Creates and saves processed intensity features for the Cords2024 dataset.

        This method reads raw intensity data, renames columns based on a mapping,
        extracts sample and object IDs, and saves the processed intensity features
        as Parquet files, grouped by sample ID.
        """
        logger.info('Creating intensity features')

        path = self.raw_dir / 'single_cell_experiment_objects/SingleCellExperiment Objects/intensity.parquet'
        intensity = pd.read_parquet(path)
        panel = pd.read_parquet(self.get_panel_path('published'))

        column_to_target = {
            'Myeloperoxidase (MPO)': 'myelope',
            'FSP1 / S100A4': 'fsp1_s1',
            'SMA': 'sma',
            'Histone H3': 'histone',
            'FAP': 'fap',
            'HLA-DR': 'hla_dr',
            'CD146': 'cd146',
            'Cadherin-11': 'cadheri',
            'Carbonic Anhydrase IX': 'carboni',
            'Collagen I + Fibronectin': 'collage',
            'VCAM1': 'vcam1',
            'CD20': 'cd20',
            'CD68': 'cd68',
            'Indoleamine 2- 3-dioxygenase (IDO)': 'indolea',
            'CD3': 'cd3',
            'Podoplanin': 'podopla',
            'MMP11': 'mmp11',
            'CD279 (PD-1)': 'cd279_p',
            'CD73': 'cd73',
            'MMP9': 'mmp9',
            'p75 (CD271)': 'p75_cd2',
            'TCF1/TCF7': 'tcf1tcf',
            'CD10': 'cd10',
            'Vimentin': 'vimenti',
            'FOXP3': 'foxp3',
            'CD45RA + CD45R0': 'cd45ra',
            'PNAd': 'pnad',
            'CD8a': 'cd8a',
            'CD248 / Endosialin': 'cd248_e',
            'LYVE-1': 'lyve_1',
            'PDGFR-b': 'cd140b',
            'CD34': 'cd34',
            'CD4': 'cd4',
            'vWF + CD31': 'cd31',
            'CXCL12': 'anti_hu',
            'CCL21': 'ccl21_6',
            'Pan Cytokeratin + Keratin Epithelial': 'pan_cyto',
            'Cadherin-6': 'k_cadhe',
            'Iridium_191': 'iridium',
            'Iridium_193': 'iridium_1',
            'Ki-67': 'ki_67',
            'Caveolin-1': 'caveoli',
            'CD15': 'cd15'
        }
        # target_to_column = {v: k for k, v in column_to_target.items()}
        intensity = intensity.rename(columns=column_to_target)

        ids = intensity['object_id'].str.split('_')
        sample_id = ids.str[:-1].str.join('_')
        object_id = ids.str[-1]

        intensity['sample_id'] = sample_id
        intensity['object_id'] = object_id

        intensity = intensity.convert_dtypes()
        intensity = intensity.set_index(['sample_id', 'object_id'])
        assert set(panel.target) - set(intensity.columns) == set()

        version_name = 'published'
        save_dir = self.intensity_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in intensity.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_features(self):
        """
        Orchestrates the creation of both spatial and intensity features.
        """
        self.create_features_spatial()
        self.create_features_intensity()

    def process_rdata(self, force: bool = False):
        """
        Processes R data files (RDS) to extract and convert single-cell experiment data
        into Parquet format for metadata, spatial, and intensity features.

        Args:
            force (bool): If True, forces re-processing even if Parquet files already exist.
        """
        import subprocess
        import textwrap

        sce_anno_path = self.raw_dir / 'single_cell_experiment_objects' / 'SingleCellExperiment Objects/sce_all_annotated.rds'
        assert sce_anno_path.exists()

        save_metadata_path = sce_anno_path.parent / 'cell_types.parquet'
        save_intensity_path = sce_anno_path.parent / 'intensity.parquet'
        save_spatial_path = sce_anno_path.parent / 'spatial.parquet'
        if save_metadata_path.exists() and save_intensity_path.exists() and save_spatial_path.exists() and not force:
            logger.info(f"RDS to Parquet conversion, files already exist. Skipping.")
            return

        logger.info('Converting RDS files to Parquet format')

        r_script = textwrap.dedent(f"""
            options(repos = c(CRAN = "https://cloud.r-project.org"))

            if (!requireNamespace("BiocManager", quietly = TRUE)) {{
                install.packages("BiocManager", quiet = TRUE)
            }}

            install_if_missing_cran <- function(pkg) {{
                if (!requireNamespace(pkg, quietly = TRUE)) {{
                    install.packages(pkg, dependencies = TRUE, quiet = TRUE)
                }}
            }}

            install_if_missing_bioc <- function(pkg) {{
                if (!requireNamespace(pkg, quietly = TRUE)) {{
                    BiocManager::install(pkg, ask = FALSE, update = FALSE)
                }}
            }}

            install_if_missing_cran("arrow")
            install_if_missing_bioc("SingleCellExperiment")

            library(arrow)
            library(SingleCellExperiment)

            data <- readRDS("{sce_anno_path}")
            coldat <- colData(data)
            cell_types <- coldat[, c("cell_category", "cell_type", "cell_subtype")]
            cell_types <- as.data.frame(cell_types)
            cell_types$object_id <- rownames(cell_types)
            write_parquet(cell_types, "{save_metadata_path}")
            
            spatial <- coldat[, c("Center_X", "Center_Y", "Area", "MajorAxisLength", "MinorAxisLength", "Compartment")]
            spatial <- as.data.frame(spatial)
            spatial$object_id <- rownames(spatial)
            write_parquet(spatial, "{save_spatial_path}")
            
            intensity = assay(data, "counts")
            intensity = t(intensity)
            intensity = as.data.frame(intensity)
            intensity$object_id <- rownames(intensity)
            write_parquet(intensity, "{save_intensity_path}")
        """)

        # Wrap the command in a shell to load modules and run the R script
        shell_command = textwrap.dedent(f"""
            # module load r-light/4.4.1
            Rscript -e '{r_script.strip()}'
        """)
        subprocess.run(shell_command, shell=True, check=True)

    def create_annotated(self, version_name: str = 'annotated', mask_version: str = "published"):
        """
        Creates an annotated version of the dataset by filtering masks, intensity, and metadata
        to include only objects that are present across all three modalities.

        Args:
            version_name (str): The name for the new annotated version (default: 'annotated').
            mask_version (str): The mask version to use for annotation (default: 'published').
        """
        from ai4bmr_datasets.utils import io
        import numpy as np

        logger.info(f'Creating new data version: `annotated`')

        metadata_version = 'published'
        metadata = pd.read_parquet(filter_paths(self.metadata_dir / metadata_version), engine='fastparquet')
        intensity = pd.read_parquet(filter_paths(self.intensity_dir / metadata_version), engine='fastparquet')

        # mask_version = 'published_cell'
        masks_dir = self.masks_dir / mask_version
        mask_paths = [p for p in masks_dir.glob("*.tiff") if not p.name.startswith('.')]

        image_version = 'published'
        images_dir = self.images_dir / image_version
        image_paths = list(images_dir.glob("*.tiff"))

        # collect sample ids
        sample_ids = set(metadata.index.get_level_values('sample_id').unique()) \
                     & set(intensity.index.get_level_values('sample_id').unique()) \
                     & set([i.stem for i in mask_paths]) \
                     & set([i.stem for i in image_paths])

        logger.info(f"Found {len(sample_ids)} samples with metadata, intensity, masks and images")

        # ANNOTATED DATA
        version = 'annotated'
        save_masks_dir = self.masks_dir / version
        save_masks_dir.mkdir(parents=True, exist_ok=True)

        save_intensity_dir = self.intensity_dir / version
        save_intensity_dir.mkdir(parents=True, exist_ok=True)

        save_metadata_dir = self.metadata_dir / version
        save_metadata_dir.mkdir(parents=True, exist_ok=True)

        for sample_id in tqdm(sample_ids):

            # NOTE: remove core 178 as the object ids in the annotated data are no longer aligned with the masks as they
            #   had to regenerate the masks and they no longer align with the metadata
            if sample_id.startswith('178_B'):
                continue

            save_path = save_masks_dir / f"{sample_id}.tiff"

            if save_path.exists():
                logger.info(f"Skipping annotated mask {save_path}. Already exists.")
                continue

            mask_path = masks_dir / f"{sample_id}.tiff"
            if not mask_path.exists():
                logger.warning(f'No corresponding mask found for {sample_id}. Skipping.')
                continue

            # logger.info(f"Saving annotated mask {save_path}")

            mask = io.imread(mask_path)
            intensity_ = intensity.xs(sample_id, level='sample_id', drop_level=False)
            metadata_ = metadata.xs(sample_id, level='sample_id', drop_level=False)

            intensity_objs = set(intensity_.index.get_level_values('object_id').astype(int).unique())
            metadata_objs = set(metadata_.index.get_level_values('object_id').astype(int).unique())
            mask_objs = set(np.unique(mask)) - {0}

            objs = metadata_objs.intersection(mask_objs).intersection(intensity_objs)
            missing_objs = mask_objs - objs
            if missing_objs:
                logger.warning(
                    f'{sample_id} has {len(objs)} common objects and {len(missing_objs)} mask objects without metadata')

            objs = np.asarray(sorted(objs), dtype=mask.dtype)
            mask_filtered = np.where(np.isin(mask, objs), mask, 0)
            assert len(np.unique(mask_filtered)) == len(objs) + 1
            io.save_mask(save_path=save_path, mask=mask_filtered)

            idx = pd.IndexSlice[:, objs.astype(str)]

            intensity_ = intensity_.loc[idx, :]
            intensity_.to_parquet(save_intensity_dir / f"{sample_id}.parquet", engine='fastparquet')

            metadata_ = metadata_.loc[idx, :]
            metadata_.to_parquet(save_metadata_dir / f"{sample_id}.parquet", engine='fastparquet')

