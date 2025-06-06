import re
from pathlib import Path

import numpy as np
import pandas as pd
from ai4bmr_core.utils.saving import save_image, save_mask
from ai4bmr_core.utils.tidy import tidy_name
from tifffile import imread
from loguru import logger
from ai4bmr_datasets.utils.download import unzip_recursive
from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset


class Keren2018(BaseIMCDataset):
    name = 'Keren2018'
    id = 'Keren2018'
    doi = '10.1016/j.cell.2018.08.039'
    notes = """
    Download data from https://www.angelolab.com/mibi-data, unzip and place in `base_dir/01_raw`.
    Save Ionpath file as tnbc. We are waiting for the authors to provide the data in a more accessible format.
    The supplementary_table_1_path is downloaded as Table S1 from the paper.
    
    - patient_id 30 is missing due to noise
    - we have patient ids up to 44 in `raw_sca` but only have images for patient ids up to 41
    """

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)

    def prepare_data(self):
        logger.info('Waiting for the authors to provide the data in a more accessible format.')
        return
        self.create_panel()
        self.create_metadata()
        self.create_images()
        self.create_masks()
        self.create_clinical_metadata()
        self.create_features_intensity()
        self.create_filtered_masks()

    def download(self, force: bool = False):
        import requests
        import shutil

        download_dir = self.raw_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "https://ars.els-cdn.com/content/image/1-s2.0-S0092867418311000-mmc2.xlsx": download_dir / "1-s2.0-S0092867418311000-mmc1.xlsx",
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
            if target_path.suffix == '.zip':
                unzip_recursive(target_path)

        shutil.rmtree(self.raw_dir / '__MACOSX', ignore_errors=True)

        logger.info("✅ Download and extraction completed.")

    def create_panel(self):
        logger.info("Creating panel")

        # downloaded as Table S1 from paper
        supplementary_table_1_path = self.raw_dir / "1-s2.0-S0092867418311000-mmc1.xlsx"
        panel_1 = pd.read_excel(supplementary_table_1_path, header=3)[:31]
        panel_1.columns = panel_1.columns.str.strip()
        panel_1 = panel_1.rename(columns={"Antibody target": "target"})
        panel_2 = pd.read_excel(supplementary_table_1_path, header=37)
        panel_2.columns = panel_2.columns.str.strip()
        panel_2 = panel_2.rename(columns={"Label": "target", "Titer": "Titer (µg/mL)"})
        panel = pd.concat([panel_1, panel_2])

        panel = panel.rename(columns={"Mass channel": "mass_channel"})

        # note: Biotin has not Start,Stop,NoiseT
        #   for PD-L1, we replace the Mass channel and Isotope with the values in Biotin
        panel.loc[panel.target == "PDL1-biotin", "mass_channel"] = panel.loc[
            panel.target == "Biotin", "mass_channel"
        ].iloc[0]
        panel.loc[panel.target == "PDL1-biotin", "Isotope"] = panel.loc[
            panel.target == "Biotin", "Isotope"
        ].iloc[0]
        panel = panel[panel.Start.notna()]

        target_map = {"PDL1-biotin": "PD-L1", "Ki-67": "Ki67"}
        panel = panel.assign(target=panel.target.replace(target_map))

        panel = panel.sort_values("mass_channel", ascending=True)
        # panel = panel.assign(page=range(len(panel)))
        panel = panel.assign(
            target_original_name=panel.target, target=panel.target.map(tidy_name)
        )

        # extract original page numbers for markers
        raw_images_dir = self.raw_dir / "tnbc"
        img_path = sorted(
            filter(
                lambda f: not f.name.startswith("."), raw_images_dir.glob("*.tiff")
            )
        )[0]
        metadata = self.get_tiff_metadata(img_path)
        metadata = metadata.rename(columns={"target": "target_from_tiff"}).reset_index()

        panel = panel.merge(metadata, on="mass_channel", how="left")
        assert panel.isna().any().any() == False
        panel = panel.sort_values("page").reset_index(drop=True)

        panel.index.name = "channel_index"
        panel = panel.convert_dtypes()

        panel_path = self.get_panel_path(image_version='published')
        panel_path.parent.mkdir(exist_ok=True, parents=True)
        panel.to_parquet(panel_path, engine='fastparquet')

    @staticmethod
    def get_tiff_metadata(path):
        import tifffile

        tag_name = "PageName"
        metadata = pd.DataFrame()
        metadata.index.name = "page"
        with tifffile.TiffFile(path) as tif:
            for page_num, page in enumerate(tif.pages):
                page_name_tag = page.tags.get(tag_name)
                metadata.loc[page_num, tag_name] = page_name_tag.value
        metadata = metadata.assign(PageName=metadata.PageName.str.strip())
        metadata = metadata.PageName.str.extract(
            r"(?P<target>\w+)\s+\((?P<mass_channel>\d+)"
        )

        metadata.mass_channel = metadata.mass_channel.astype("int")
        metadata = metadata.convert_dtypes()

        return metadata

    def create_images(self):
        logger.info("Creating images")

        panel_path = self.get_panel_path(image_version='published')
        panel = pd.read_parquet(panel_path)

        images_dir = self.get_image_version_dir('published')
        images_dir.mkdir(exist_ok=True, parents=True)

        raw_images_dir = self.raw_dir / "tnbc"
        image_paths = sorted(
            filter(
                lambda f: not f.name.startswith("."), raw_images_dir.glob("*.tiff")
            )
        )
        for img_path in image_paths:
            logger.debug(f"processing image {img_path.stem}")
            sample_id = re.search(r"Point(\d+).tiff", img_path.name).groups()[0]

            if (images_dir / f"{sample_id}.tiff").exists():
                continue

            img = imread(img_path)
            img = img[panel.page.values]  # extract only the layers of the panel
            assert len(img) == len(panel)

            save_image(img, images_dir / f"{sample_id}.tiff")


    def create_masks(self):
        # NOTE: filter masks by objects in data
        #   - 0: cell boarders
        #   - 1: background
        #   - 2..N: cells
        from PIL import Image

        save_masks_dir = self.masks_dir / 'published'
        save_masks_dir.mkdir(exist_ok=True, parents=True)

        raw_images_dir = self.raw_dir / "tnbc"
        raw_masks_dir = self.raw_dir / "TNBC_shareCellData"
        mask_paths = sorted(list(raw_masks_dir.glob(r"p*_labeledcellData.tiff")))

        for mask_path in mask_paths:
            sample_id = re.match(r"p(\d+)", mask_path.stem).group(1)

            save_path = save_masks_dir / f"{sample_id}.tiff"
            if save_path.exists():
                logger.info(f'Mask for sample {sample_id} already exists, skipping.')
                continue

            logger.info(f"Creating mask {sample_id}")

            mask = imread(mask_path)

            # load segmentation mask from raw data
            segm = Image.open(raw_images_dir
                / f"TA459_multipleCores2_Run-4_Point{sample_id}"
                / "segmentation_interior.png")
            segm = np.array(segm)

            # note: set region with no segmentation to background
            # filter masks by objects in data
            #   - 0: cell boarders
            #   - 1: background
            #   - 2..n: cells
            mask[segm == 0] = 0
            assert 1 not in mask.flat

            save_mask(mask, save_path)

    def create_clinical_metadata(self):
        logger.info("Creating metadata")
        samples = pd.read_csv(self.raw_dir / "TNBC_shareCellData" / "patient_class.csv", header=None)
        samples.columns = ["patient_id", "cancer_type_id"]
        samples = samples.assign(sample_id=samples["patient_id"]).set_index("sample_id")
        samples.index = samples.index.astype(str)

        cancer_type_map = {0: "mixed", 1: "compartmentalized", 2: "cold"}
        samples = samples.assign(
            cancer_type=samples["cancer_type_id"].map(cancer_type_map)
        )

        path = self.clinical_metadata_path
        path.parent.mkdir(exist_ok=True, parents=True)
        samples.to_parquet(path, engine='fastparquet')

    def create_metadata(self):
        logger.info("Creating metadata [annotations]")
        raw_sca_path = self.raw_dir / "TNBC_shareCellData" / "cellData.csv"
        data = pd.read_csv(raw_sca_path)

        metadata_cols = [
            "SampleID",
            "cellLabelInImage",
            "cellSize",
            "tumorYN",
            "tumorCluster",
            "Group",
            "immuneCluster",
            "immuneGroup",
        ]
        metadata = data[metadata_cols]
        metadata = metadata.rename(columns={"SampleID": "sample_id", "cellLabelInImage": "object_id"})
        metadata.sample_id = metadata.sample_id.astype(str)
        metadata = metadata.set_index(["sample_id", "object_id"])

        grp_map = {
            1: "unidentified",
            2: "immune",
            3: "endothelial",
            4: "mesenchymal_like",
            5: "tumor",
            6: "keratin_positive_tumor",
        }
        metadata = metadata.assign(group_name=metadata["Group"].map(grp_map))

        immune_grp_map = {
            1: "Tregs",
            2: "CD4 T",
            3: "CD8 T",
            4: "CD3 T",
            5: "NK",
            6: "B",
            7: "Neutrophils",
            8: "Macrophages",
            9: "DC",
            10: "DC/Mono",
            11: "Mono/Neu",
            12: "Other immune",
        }
        metadata = metadata.assign(
            immune_group_name=metadata["immuneGroup"].map(immune_grp_map)
        )

        col_map = dict(
            cellSize="cell_size",
            tumorYN="tumor_yn",
            tumorCluster="tumor_cluster_id",
            Group="group_id",
            immuneCluster="immune_cluster_id",
            immuneGroup="immune_group_id",
        )
        metadata = metadata.rename(columns=col_map)
        metadata = metadata.assign(tumor_yn=metadata.tumor_yn.astype("bool"))

        label_name = metadata.group_name.copy()
        filter = metadata.immune_group_name.notna()
        label_name[filter] = metadata.immune_group_name[filter]
        metadata = metadata.assign(label_name=label_name)
        label_id_map = dict(zip(label_name.unique(), range(len(label_name.unique()))))
        metadata = metadata.assign(label_id=metadata.label_name.map(label_id_map))

        metadata = metadata.convert_dtypes()

        for grp_name, grp_data in metadata.groupby("sample_id"):
            path = self.metadata_dir / 'published' / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')

    def create_features_spatial(self):
        logger.info("Creating spatial features")

        samples = pd.read_parquet(self.clinical_metadata_path)
        spatial = pd.read_csv(self.raw_dir / "TNBC_shareCellData" / "cellData.csv")

        index_columns = ["sample_id", "object_id"]
        feat_cols_names = ["cellSize"]
        spatial = spatial.rename(
            columns={"SampleID": "sample_id", "cellLabelInImage": "object_id"}
        )
        spatial.sample_id = spatial.sample_id.astype(str)
        spatial = spatial.set_index(index_columns)

        spatial = spatial[feat_cols_names]
        spatial.columns = spatial.columns.map(tidy_name)
        spatial = spatial.convert_dtypes()

        valid_obs = set(spatial.index).intersection(self.observations_default)
        default = spatial.loc[list(valid_obs), :]

        valid_samples = set(default.index.get_level_values("sample_id")).intersection(
            samples.index
        )
        default = default.loc[list(valid_samples)]

        for grp_name, grp_data in default.groupby("sample_id"):
            dir = self.spatial_dir / 'published'
            path = dir / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')

        del default

        valid_obs = set(spatial.index).intersection(self.observations_unfiltered)
        unfiltered = spatial.loc[list(valid_obs), :]

        valid_samples = set(
            unfiltered.index.get_level_values("sample_id")
        ).intersection(samples.index)
        unfiltered = unfiltered.loc[list(valid_samples)]

        for grp_name, grp_data in unfiltered.groupby("sample_id"):
            dir = self.intensity_dir / 'published'
            path = dir / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')

    def create_features_intensity(self):
        logger.info("Creating intensity features")

        samples = pd.read_parquet(self.clinical_metadata_path)
        panel_path = self.get_panel_path(image_version='published')
        panel = pd.read_parquet(panel_path)
        intensity = pd.read_csv(self.raw_dir / "TNBC_shareCellData" / "cellData.csv")

        index_columns = ["sample_id", "object_id"]
        intensity = intensity.rename(
            columns={"SampleID": "sample_id", "cellLabelInImage": "object_id"}
        )
        intensity.sample_id = intensity.sample_id.astype(str)
        intensity = intensity.set_index(index_columns)

        intensity.columns = intensity.columns.map(tidy_name)

        intensity = intensity[panel.target]
        assert intensity.isna().sum().sum() == 0

        intensity = intensity.convert_dtypes()

        # filter for samples that are in both processed data and patient data
        # NOTE: patient 30 has been excluded from the analysis due to noise
        #   and there are patients 42,43,44 in the processed single cell data but we do not have images for them
        valid_obs = set(intensity.index).intersection(self.observations_default)
        default = intensity.loc[list(valid_obs), :]

        valid_samples = set(default.index.get_level_values("sample_id")).intersection(
            samples.index
        )
        default = default.loc[list(valid_samples)]

        for grp_name, grp_data in default.groupby("sample_id"):
            dir = self.intensity_dir / 'published'
            path = dir / f"{grp_name}.parquet"
            path.parent.mkdir(exist_ok=True, parents=True)
            grp_data.to_parquet(path, engine='fastparquet')


