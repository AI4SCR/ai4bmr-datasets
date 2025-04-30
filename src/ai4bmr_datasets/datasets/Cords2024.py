import json
import re
from pathlib import Path

import pandas as pd
from ai4bmr_core.utils.tidy import tidy_name
from loguru import logger
from tifffile import imread
from tifffile import imwrite

from .BaseIMCDataset import BaseIMCDataset


class Cords2024(BaseIMCDataset):

    @property
    def id(self):
        return "NSCLC"

    @property
    def doi(self):
        return "10.1016/j.ccell.2023.12.021"

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
        self.process_acquisitions()
        self.process_clinical_metadata()
        self.create_panel()
        self.create_images()

    def download(self, force: bool = False):
        import wget
        import zipfile

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
            "https://drive.usercontent.google.com/download?id=1wXboMZpkLzk7ZP1Oib1q4NBwvFGK_h7G&confirm=xxx": download_dir
            / "cell_masks.zip",
        }

        # Download files
        for url, target_path in file_map.items():
            if not target_path.exists() or force:
                logger.info(f"Downloading {url} → {target_path}")
                wget.download(url, str(target_path))
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

    def process_acquisitions(self):
        from readimc import MCDFile

        acquisitions_dir = self.raw_acquisitions_dir
        acquisitions_dir.mkdir(parents=True, exist_ok=True)

        num_failed_reads = 0

        mcd_paths = list(self.raw_dir.glob("*.mcd"))
        for mcd_path in mcd_paths:
            logger.info(f"Processing {mcd_path.name}")
            mcd_id = re.search(r".+_TMA_(\d+_\w+).mcd", mcd_path.name).group(1)

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

    def create_images(self):
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
