import json
from pathlib import Path

import pandas as pd
from ai4bmr_core.utils.tidy import tidy_name
from loguru import logger
from tifffile import imread
from tifffile import imwrite

from .BaseIMCDataset import BaseIMCDataset
import re


def unzip_recursive(zip_path: Path, extract_dir: Path = None):
    import zipfile
    from loguru import logger

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extracted_names = zip_ref.namelist()
        extract_dir = extract_dir or zip_path.parent / zip_path.stem
        is_extracted = all((extract_dir / name).exists() for name in extracted_names)
        if is_extracted:
            if logger:
                logger.info(f"Skipping extraction for {zip_path.name}, files already exist.")
        else:
            if logger:
                logger.info(f"Unzipping {zip_path.name}")
            zip_ref.extractall(extract_dir)

        # Recursively unzip any .zip files in extracted content
        for name in extracted_names:
            extracted_path = extract_dir / name
            if extracted_path.suffix == ".zip":
                unzip_recursive(extracted_path)

class Cords2024(BaseIMCDataset):

    @property
    def id(self):
        return "Cords2024"

    @property
    def doi(self):
        return "10.1016/j.ccell.2023.12.021"

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self.base_dir = base_dir

        # raw data paths
        self.raw_clinical_metadata = self.raw_dir / "comp_csv_files" / "cp_csv" / "clinical_data_ROI.csv"
        # self.raw_panel = self.raw_dir / "cp_csv" / "panel.csv"
        self.raw_acquisitions_dir = self.raw_dir / "acquisitions"

        # populated by `self.load()`
        self.sample_ids = None
        self.samples = None
        self.images = None
        self.masks = None
        self.panel = None
        self.intensity = None
        self.spatial = None

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.samples.index[idx]
        return {
            "sample_id": sample_id,
            # "sample": self.samples.loc[sample_id],
            "image": self.images[sample_id],
            # "mask": self.masks[sample_id],
            "panel": self.panel,
            # "intensity": self.intensity.loc[sample_id],
            # "spatial": self.spatial.loc[sample_id],
        }

    def process(self):
        self.download()
        self.process_acquisitions()

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
        search = re.search(r"TMA_(\d+)_(\w+)_s0_a(\d+)", (path))
        return '_'.join(search.groups())

    def process_acquisitions(self):
        """
            Failed reading acquisitions for:
            # have masks
            - 88_A_10
            - 88_A_9
            - 88_B_110
            - 86_B_1
            - 176_B_47
            - 175_B_90

            - 176_C_2
            - 87_A_49
        """

        from readimc import MCDFile

        acquisitions_dir = self.raw_acquisitions_dir
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
                            logger.info(f"Failed to read acquisition! ")
                            num_failed_reads += 1
                            continue

    def validate_masks(self):
        import numpy as np
        version_name = 'published'
        self.setup(image_version=version_name, mask_version=version_name, metadata_version=version_name,
                   load_metadata=True, load_spatial=False, load_intensity=False)

        mismatches = []
        for sample_id, mask in self.masks.items():
            uniq_mask_objs = set(map(int, np.unique(mask.data))) - {0}
            meta = self.metadata.loc[sample_id]
            uniq_meta_objs = set(meta.index.astype(int))
            if not uniq_mask_objs == uniq_meta_objs:
                logger.info(f'Mask objects and metadata objects do not match for {sample_id}')
                mismatches.append((sample_id, uniq_meta_objs, uniq_mask_objs))

        for i, (sample_id, uniq_meta_objs, uniq_mask_objs) in enumerate(mismatches):
            logger.info(f"Sample ID: {sample_id}")
            logger.info(f"Unique mask objects: {len(uniq_mask_objs)} (min: {min(uniq_mask_objs)}, max: {max(uniq_mask_objs)})")
            logger.info(f"Unique metadata objects: {len(uniq_meta_objs)} (min: {min(uniq_meta_objs)}, max: {max(uniq_meta_objs)})")
            if i == 10:
                break

        item = list(filter(lambda x: x[0] == '178_B_71', mismatches))

    def create_images(self):
        panel_path = self.get_panel_path("published")
        panel = pd.read_parquet(panel_path)

        acquisitions_dir = self.raw_acquisitions_dir
        # samples = pd.read_parquet(self.clinical_metadata_path())
        # sample_ids = set(samples.index)

        acquisition_ids = set(i.stem for i in acquisitions_dir.glob("*.tiff"))
        acquisition_paths = list(acquisitions_dir.glob("*.tiff"))
        # ids = sample_ids.intersection(acquisition_ids)

        save_dir = self.get_image_version_dir("published")
        save_dir.mkdir(parents=True, exist_ok=True)
        for img_path in acquisition_paths:
            save_path = save_dir / f"{img_path.name}"

            if save_path.exists():
                logger.info(f"Skipping image {img_path}. Already exists.")
                continue

            img = imread(img_path)
            img = img[panel.page.values]

            imwrite(save_path, img)

    def create_masks(self):
        import tifffile
        import shutil
        from loguru import logger

        raw_masks_dir = self.raw_dir / 'cell_masks_google_drive'
        mask_paths = list(raw_masks_dir.rglob("*.tiff"))

        # note:
        # - 86_B_mask also contains 86_C masks
        # - 176_A_mask the filenames do not match the parent folder name but have TMA_176_B in the name filename
        mask_paths_from_176_A = [p for p in mask_paths if '176_A_mask' in str(p)]
        mask_paths = [p for p in mask_paths if p.parent.parent.name.replace('_mask', '') in p.name]

        mask_paths_with_ids = [(self.get_sample_id_from_path(path), path) for path in mask_paths]
        mask_paths_from_176_A_with_ids = [(self.get_sample_id_from_path(path), path) for path in mask_paths_from_176_A]
        mask_paths_from_176_A_with_ids = [(x[0].replace('176_B', '176_A'), x[1])
                                          for x in mask_paths_from_176_A_with_ids
                                          ]

        mask_paths_with_ids = mask_paths_with_ids + mask_paths_from_176_A_with_ids

        # note: rename 178_B_2 to 178_B
        mask_paths_with_ids = [(x[0].replace('178_B_2', '178_B'), x[1]) for x in mask_paths_with_ids]

        # mask_paths_with_no_ids = list(filter(lambda x: x[0] is None, mask_paths_with_ids))
        assert (pd.Series([x[0] for x in mask_paths_with_ids]).value_counts() == 1).all()

        save_dir = self.get_mask_version_dir('published')
        save_dir.mkdir(parents=True, exist_ok=True)

        image_ids = [i.stem for i in self.get_image_version_dir('published').glob('*.tiff')]
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
        from ai4bmr_core.utils.tidy import tidy_name
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
        logger.info(f"No clinical metadata for {len(no_clinical_metadata)} ROIs found: {', '.join(no_clinical_metadata)}")
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
        path = self.raw_dir / 'cell_types.parquet'
        metadata = pd.read_parquet(path)

        ids = metadata['object_id'].str.split('_')
        sample_id = ids.str[:-1].str.join('_')
        object_id = ids.str[-1]

        metadata['sample_id'] = sample_id
        metadata['object_id'] = object_id

        metadata = metadata.convert_dtypes()
        metadata = metadata.set_index(['sample_id', 'object_id'])

        version_name = self.get_version_name(version='published')
        save_dir = self.metadata_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in metadata.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_features_spatial(self):
        pass

    def create_features_intensity(self):
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

