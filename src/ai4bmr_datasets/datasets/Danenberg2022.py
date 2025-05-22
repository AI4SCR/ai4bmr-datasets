import re
from pathlib import Path

import pandas as pd
from loguru import logger
from tifffile import imread, imwrite

from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset
from ai4bmr_datasets.utils.download import unzip_recursive
from ai4bmr_core.utils.tidy import tidy_name


class Danenberg2022(BaseIMCDataset):

    @property
    def id(self):
        return "Danenberg2022"

    @property
    def name(self):
        return "Danenberg2022"

    @property
    def doi(self):
        return "10.1038/s41588-022-01041-y"

    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self.base_dir = base_dir

        self.sample_ids = None
        self.samples = None
        self.images = None
        self.masks = None
        self.panel = None
        self.intensity = None
        self.spatial = None

    def prepare_data(self):
        self.download()

        self.create_clinical_metadata()

        self.create_panel()
        self.create_images()
        self.create_masks()
        self.create_features()

    def download(self, force: bool = False):
        import requests
        import shutil

        download_dir = self.raw_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "https://zenodo.org/records/6036188/files/MBTMEStrIMCPublic.zip?download=1": download_dir / 'mbtme_imc_public.zip',
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

    def get_metabric_id(self, path):
        return re.search(r'MB\d+', path).group(0)

    def get_sample_id(self, path):
        return re.search(r'MB\d+_\d+', path).group(0)

    def get_mask_version(self, path):
        return re.search(r'MB\d+_\d+_(\w+)Mask.\tiff', path).group(1)

    def create_images(self):
        images_dir = self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/Images'
        img_paths = list(images_dir.rglob('*.tiff'))
        img_paths = sorted(filter(lambda x: 'FullStack' in str(x), img_paths))

        save_dir = self.images_dir / self.get_version_name(version='published')
        save_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            sample_id = self.get_sample_id(str(img_path))

            save_path = save_dir / f"{sample_id}.tiff"
            if save_path.exists():
                logger.warning(f"Image file {save_path} already exists. Skipping.")
                continue
            else:
                logger.info(f"Creating image {save_path}")

            img = imread(img_path)
            assert len(img) == 39

            imwrite(save_path, img)

    def create_masks(self):
        images_dir = self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/Images'
        img_paths = list(images_dir.rglob('*.tiff'))
        img_paths = sorted(filter(lambda x: 'Mask' in str(x), img_paths))

        for img_path in img_paths:
            sample_id = self.get_sample_id(str(img_path))

            version = self.get_mask_version(str(img_path)).strip().lower()
            version = f'published_{version}'
            save_dir = self.masks_dir / self.get_version_name(version=version)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample_id}.tiff"

            if save_path.exists():
                logger.warning(f"Image file {save_path} already exists. Skipping.")
                continue
            else:
                logger.info(f"Creating image {save_path}")

            img = imread(img_path)
            assert img.ndim == 2

            imwrite(save_path, img)

    def create_metadata(self):

        panel = pd.read_parquet(self.get_panel_path('published'))
        sc = pd.read_csv(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/SingleCells.csv')
        import pyreadr

        scfst = pyreadr.read_r('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Danenberg2022/01_raw/mbtme_imc_public/MBTMEIMCPublic/SingleCells.fst')
        clinicalfst = pyreadr.read_r('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Danenberg2022/01_raw/mbtme_imc_public/MBTMEIMCPublic/IMCClinical.fst')

        sample_ids = sc.metabric_id.str.replace('-', '_') + '_' + sc.ImageNumber.astype(str)
        sc['sample_id'] = sample_ids
        sc = sc.rename(columns={'ObjectNumber': 'object_id'})
        sc.columns = sc.columns.map(tidy_name)

        expr_cols = set(panel.target)
        expr_cols = list(expr_cols.intersection(set(sc.columns)))
        expr = sc[expr_cols]
        sc = sc.drop(columns=expr_cols)

        metadata_cols = ['is_tumour', 'metabric_id', 'her2_3b5', 'is_normal', 'ox40', 'is_dcis', 'pd_1', 'pan_ck',
                         'her2_d8f12', 'image_number', 'c_caspase3', 'sample_id', 'gitr', 'cd8', 'er', 'b2m',
                         'is_epithelial', 'ck5', 'is_interface', 'cell_phenotype', 'cd45ro', 'cxcl12',
                         'is_perivascular', 'pdgfrb', 'object_id', 'location_center_x', 'icos', 'ck8_18',
                         'location_center_y', 'is_hot_aggregate', 'area_shape_area']

        set(sc.columns)

        annot = pd.read_excel(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/SingleCellsAnnotation.xlsx')

        metadata.columns = metadata.columns.map(tidy_name)

        version_name = self.get_version_name(version='published')
        save_dir = self.metadata_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in metadata.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def add_sample_id_and_object_id(self, data):
        sample_ids = data.core.str.replace('ZTMA208_slide_', '')
        sample_ids = sample_ids.str.replace('BaselTMA_SP', '')

        data['sample_id'] = sample_ids.map(tidy_name)
        data = data.convert_dtypes()
        data = data.set_index(['sample_id'])

        if 'CellId' in data.columns:
            data = data.rename(columns={'CellId': 'object_id'})
            data = data.set_index(['object_id'], append=True)

        return data

    def create_clinical_metadata(self):

        metadata = pd.read_csv(
            self.raw_dir / 'single_cell_and_metadata/Data_publication/BaselTMA/Basel_PatientMetadata.csv')
        pass

        metadata = self.add_sample_id_and_object_id(metadata)

        metadata = metadata.convert_dtypes()
        for col in metadata.select_dtypes(include=['object', 'string']).columns:
            if col == 'file_name_full_stack':
                continue
            metadata.loc[:, col] = metadata[col].map(lambda x: tidy_name(str(x)) if not pd.isna(x) else x)

        metadata = metadata.convert_dtypes()
        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path, engine='fastparquet')

    def create_features(self):
        logger.info(f'Creating `published` features')
        pass

    def create_panel(self):
        panel = pd.read_csv(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/AbPanel.csv')
        assert (panel.full == 1).all()

        panel = panel.drop(columns=['ilastik', 'full', 'Unnamed: 10', 'tubenumber', 'to_sort', 'function_order'])
        panel = panel.rename(columns=dict(metaltag='metal_mass'))
        panel.loc[[37, 38], 'target'] = ['dna1', 'dna2']
        panel.loc[:, 'target'] = panel.target.map(tidy_name)

        panel = panel.convert_dtypes()
        panel.index.name = "channel_index"

        panel_path = self.get_panel_path("published")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)


base_dir = Path("/users/amarti51/prometex/data/datasets/Danenberg2022")
ds = self = Danenberg2022(base_dir=base_dir)
ds.download()
# ds.create_masks()
