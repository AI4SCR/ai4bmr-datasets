import json
import re
from pathlib import Path

import pandas as pd
from ai4bmr_core.utils.tidy import tidy_name
from loguru import logger
from tifffile import imread
from tifffile import imwrite

from .BaseIMCDataset import BaseIMCDataset


class Cords2023(BaseIMCDataset):

    @property
    def id(self):
        return "Cords2023"

    @property
    def doi(self):
        return "10.1038/s41467-023-39762-1"

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self.base_dir = base_dir

        # raw data paths
        self.raw_clinical_metadata = self.raw_dir / "cp_csv" / "clinical_data_ROI.csv"
        self.raw_panel = self.raw_dir / "cp_csv" / "panel.csv"
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
        self.post_process_extracted_files()
        self.process_acquisitions()
        self.process_clinical_metadata()
        self.create_panel()
        self.create_images()

    def post_process_extracted_files(self):
        import shutil

        (self.raw_dir / 'Cell Masks').rename(self.raw_dir / 'cell_masks')
        (self.raw_dir / 'IMC').rename(self.raw_dir / 'imc_objects')
        (self.raw_dir / 'cp_30_new').rename(self.raw_dir / 'cp_output_config')
        (self.raw_dir / 'scRNA-seq').rename(self.raw_dir / 'scRNAseq_objects')

        shutil.rmtree(self.raw_dir / '__MACOSX')


    def download(self, force: bool = False):
        import zipfile
        import requests
        from tqdm import tqdm

        download_dir = self.raw_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            'https://zenodo.org/records/7540604/files/Cell%20Masks.zip?download=1': download_dir / 'cell_masks.zip',
            'https://zenodo.org/records/7540604/files/cp-output_config.zip?download=1': download_dir / 'cp_output_config.zip',
            'https://zenodo.org/records/7540604/files/IMC_dataobjects.zip?download=1': download_dir / 'imc_objects.zip',
            'https://zenodo.org/records/7540604/files/ometiff.zip?download=1': download_dir / 'ometiff.zip',
            'https://zenodo.org/records/7540604/files/scRNA-seq_dataobjects.zip?download=1': download_dir / 'scRNAseq_objects.zip',
        }

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
            with zipfile.ZipFile(target_path, "r") as zip_ref:
                extracted_names = zip_ref.namelist()

                # Check if all extracted files already exist
                is_extracted = all(
                    (download_dir / name).exists() for name in extracted_names
                )
                if is_extracted:
                    logger.info(
                        f"Skipping extraction for {target_path.name}, files already exist."
                    )
                else:
                    logger.info(f"Unzipping {target_path.name}")
                    zip_ref.extractall(download_dir)

        logger.info("✅ Download and extraction completed.")

    def process_clinical_metadata(self):
        clinical_metadata = pd.read_csv(self.raw_clinical_metadata)

        assert clinical_metadata["Tma_ac"].duplicated().sum() == 0

        sample_ids = clinical_metadata["Tma_ac"].to_list()
        sample_ids = [re.sub(r"^(\d+)([A-Za-z])", r"\1_\2", s) for s in sample_ids]
        num_sample_ids = len(sample_ids)
        sample_ids = [i.replace("178_B", "178_B_2") for i in sample_ids]
        assert len(sample_ids) == num_sample_ids
        clinical_metadata["sample_id"] = sample_ids
        clinical_metadata = clinical_metadata.set_index("sample_id")
        sample_ids = set(sample_ids)

        # check ids against images
        image_ids = {i.stem for i in self.raw_acquisitions_dir.glob("*.tiff")}
        ids = sample_ids.intersection(image_ids)
        no_clinical_metadata = image_ids - sample_ids
        logger.info(f"No clinical metadata for {len(no_clinical_metadata)} ROIs found")
        no_image_ids = sample_ids - image_ids
        logger.info(f"No images for {len(no_image_ids)} patients found")

        clinical_metadata = clinical_metadata.loc[list(ids)]
        clinical_metadata.columns = clinical_metadata.columns.map(tidy_name)
        clinical_metadata = clinical_metadata.convert_dtypes()

        samples_path = self.get_samples_path()
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        clinical_metadata.to_parquet(samples_path)

    def create_images(self):

        clinical = pd.read_csv(self.raw_dir / 'cp_output_config' / 'clinical_data.csv')
        cells = pd.read_csv(self.raw_dir / 'cp_output_config' / 'Cells.csv')
        # acq_meta = pd.read_csv(self.raw_dir / 'cp_output_config' / 'acquisition_metadata.csv')

        ['FileName_CellImage', 'FileName_CellMask', 'FileName_FullStack', 'FileName_TumourMask',
         'Group_Index', 'Group_Number',
         'Metadata_acid', 'Metadata_acname', 'Metadata_panoid', 'Metadata_roiid', 'Metadata_slideid',
         ]
        image = pd.read_csv(self.raw_dir / 'cp_output_config' / 'Image.csv')
        image[['Metadata_acid', 'Metadata_acname', 'Metadata_panoid', 'Metadata_roiid', 'Metadata_slideid']].drop_duplicates()
        image[['Metadata_acname']].drop_duplicates()

        meta = pd.read_csv(self.raw_dir / 'cp_output_config' / '20201013_LC_BC-Fibro_TBB075_s0_p8_r1_a1_ac_full.csv')
        ['Metadata_acid', 'Metadata_acid.1', 'Metadata_acname', 'Metadata_acname.1', 'Metadata_panoid', 'Metadata_roiid', 'Metadata_roiid.1', 'Metadata_slideid']
        cells['Metadata_FileLocation']
        cells[['Metadata_acid', 'Metadata_acname']]

        panel_path = self.get_panel_path("published")
        panel = pd.read_parquet(panel_path)

        acquisitions_dir = self.raw_acquisitions_dir
        samples = pd.read_parquet(self.get_samples_path())
        sample_ids = set(samples.index)

        acquisition_ids = set(i.stem for i in acquisitions_dir.glob("*.tiff"))
        ids = sample_ids.intersection(acquisition_ids)

        save_dir = self.get_images_dir("published")
        save_dir.mkdir(parents=True, exist_ok=True)
        for id_ in ids:
            img_path = self.raw_acquisitions_dir / f"{id_}.tiff"
            save_path = save_dir / f"{img_path.name}"

            if save_path.exists():
                logger.info(f"Skipping image {img_path}. Already exists.")
                continue

            img = imread(img_path)
            img = img[panel.page.values]

            imwrite(save_path, img)

    def create_masks(self):
        pass

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
        import pandas as pd
        from ai4bmr_core.utils.tidy import tidy_name

        panel_path = self.raw_dir / 'cp_output_config/LC_BC_fibro_study_final panel.csv'
        panel = pd.read_csv(panel_path)

        panel.columns = panel.columns.map(tidy_name)
        panel['target_original'] = panel['target']
        panel['target'] = panel.clean_target.map(tidy_name).to_list()

        panel = panel.convert_dtypes()
        panel.index.name = "channel_index"

        panel_path = self.get_panel_path("published")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)
