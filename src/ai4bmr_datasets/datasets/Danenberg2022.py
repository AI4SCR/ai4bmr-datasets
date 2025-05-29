import re
from pathlib import Path
import numpy as np
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
        self.process_rdata()

        self.create_panel()
        self.create_images()
        self.create_masks()
        self.create_metadata_and_intensity()
        self.create_clinical_metadata()

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

    def get_metabric_id(self, path: str):
        return re.search(r'MB\d+', path).group(0)

    def get_sample_id(self, path: str):
        match = re.search(r'MB(\d+)_(\d+)', path)
        return f'MB_{match.group(1)}_{match.group(2)}'

    def get_mask_version(self, path):
        match = re.search(r'MB(\d+)_(\d+)_(\w+)Mask\.tiff', path)
        return match.group(3)

    def process_rdata(self):
        import subprocess
        import textwrap

        logger.info('Converting fts files to Parquet format')

        r_script = f"""
        options(repos = c(CRAN = "https://cloud.r-project.org"))

        if (!requireNamespace("BiocManager", quietly = TRUE)) {{
            install.packages("BiocManager", quiet = TRUE)
        }}

        install_if_missing_cran <- function(pkg) {{
            if (!requireNamespace(pkg, quietly = TRUE)) {{
                install.packages(pkg, dependencies = TRUE, quiet = TRUE)
            }}
        }}

        install_if_missing_cran("arrow")
        install_if_missing_cran("fst")
        
        library(fst)
        library(arrow)
        
        df <- read.fst("{self.raw_dir}/mbtme_imc_public/MBTMEIMCPublic/IMCClinical.fst")
        write_parquet(df, "{self.raw_dir}/mbtme_imc_public/MBTMEIMCPublic/IMCClinical.parquet")
        
        df <- read.fst("{self.raw_dir}/mbtme_imc_public/MBTMEIMCPublic/SingleCells.fst")
        write_parquet(df, "{self.raw_dir}/mbtme_imc_public/MBTMEIMCPublic/SingleCells.parquet")
        """
        r_script = textwrap.dedent(r_script)

        # TODO: this will run only on slurm
        # subprocess.run(["Rscript", "-e", r_script], check=True)
        # Wrap the command in a shell to load modules and run the R script
        shell_command = textwrap.dedent(f"""
            module load r-light/4.4.1
            Rscript -e '{r_script.strip()}'
        """)

        # Run using bash shell to ensure module environment is available
        subprocess.run(["bash", "-c", shell_command], check=True)

    def create_images(self):
        images_dir = self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/Images'
        img_paths = list(images_dir.rglob('*.tiff'))
        img_paths = sorted(filter(lambda x: 'FullStack' in str(x), img_paths))

        save_dir = self.images_dir / self.get_version_name(version='published')
        save_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            try:
                sample_id = self.get_sample_id(str(img_path))
            except Exception as e:
                print('debug', img_path)

            save_path = save_dir / f"{sample_id}.tiff"
            if save_path.exists():
                logger.info(f"Image file {save_path} already exists. Skipping.")
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
                logger.info(f"Mask file {save_path} already exists. Skipping.")
                continue
            else:
                logger.info(f"Creating mask {save_path}")

            img = imread(img_path)
            assert img.ndim == 2

            imwrite(save_path, img)

    def create_metadata_and_intensity(self):
        logger.info('Creating `published` metadata and intensity')

        panel = pd.read_parquet(self.get_panel_path('published'))
        sc = pd.read_csv(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/SingleCells.csv')

        sample_ids = sc.metabric_id.str.replace('-', '_') + '_' + sc.ImageNumber.astype(str)
        sc['sample_id'] = sample_ids
        sc = sc.rename(columns={'ObjectNumber': 'object_id'})
        sc = sc.set_index(['sample_id', 'object_id'])
        sc.columns = sc.columns.map(tidy_name)

        # INTENSITY
        logger.info('Creating `published` intensity')

        # note: some marker as not as in the panel.target.
        marker_to_target = {
            'er': "estrogen_receptor_alpha",
            'cd45ro': 'cd45',
            'her2_3b5': 'c_erb_b_2_her2_3b5',
            'her2_d8f12': 'c_erb_b_2_her2_d8f12',
            'ck5': 'cytokeratin_5',
            'ck8_18': 'cytokeratin_8_18',
            'icos': 'cd278_icos',
            'ox40': 'cd134',
            'pd_1': 'cd279_pd_1',
            'gitr': 'gitr_tnfrsf18',
            'b2m': 'beta_2_microglobulin',
            'cd8': 'cd8a',
            'pdgfrb': 'cd140b_pdgf_receptor_beta',
            'cxcl12': 'cxcl12_sdf_1',
            'pan_ck': 'pan_cytokeratin',
            'c_caspase3': 'cleaved_caspase3',
        }
        sc.columns = sc.columns.map(lambda x: marker_to_target.get(x, x))

        expr_cols = set(panel.target)
        expr_cols = list(expr_cols.intersection(set(sc.columns)))
        assert set(panel.target) == set(expr_cols)

        expr = sc[expr_cols]
        expr = expr.convert_dtypes()
        assert expr.columns == panel.target

        version_name = self.get_version_name(version='published')
        save_dir = self.intensity_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in expr.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

        # METADATA
        logger.info('Creating `published` metadata')

        sc = sc.drop(columns=expr_cols)

        metadata_cols = ['image_number', 'object_id', 'metabric_id', 'cell_phenotype', 'is_epithelial', 'is_tumour',
                         'is_normal', 'is_dcis', 'is_interface', 'is_perivascular', 'is_hot_aggregate',
                         'area_shape_area', 'sample_id']

        coord_cols = ['location_center_x', 'location_center_y']

        # note: we have 32 cell types in contrast to the 16 in the publication?
        #   - 'Ki67^{+}' but also 'Ep Ki67^{+}'
        #   - no 'ER^{hi}CXCL12^{+}'
        #   - 'CD57^{+}' but also 'MHC I^{hi}CD57^{+}', 'Ep CD57^{+}'
        # TODO: these seem to be the
        # df = pd.read_csv(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/SI_metaclusters.csv')

        sc['label'] = sc.cell_phenotype.str.replace('+', '_pos').str.replace('&', 'and').map(tidy_name)

        sc = sc.convert_dtypes()

        save_dir = self.metadata_dir / version_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for grp_name, grp_dat in sc.groupby('sample_id'):
            save_path = save_dir / f"{grp_name}.parquet"
            grp_dat.to_parquet(save_path, engine='fastparquet')

    def create_clinical_metadata(self):

        metadata = pd.read_parquet(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/IMCClinical.parquet')
        metadata.columns = metadata.columns.map(tidy_name)
        metadata['metabric_id'] = metadata.metabric_id.str.replace('-', '_')

        meta = pd.read_parquet(self.processed_dir / 'metadata/published').index.to_frame()
        meta = meta[['sample_id']].drop_duplicates().reset_index(drop=True)
        meta['metabric_id'] = meta.sample_id.str.split('_').str[:-1].str.join('_')

        metadata = pd.merge(metadata, meta, on='metabric_id', how='right')
        metadata = metadata.set_index('sample_id')

        metadata = metadata.convert_dtypes()
        for col in metadata.select_dtypes(include=['object', 'string']).columns:
            if col == 'file_name_full_stack':
                continue
            metadata.loc[:, col] = metadata[col].map(lambda x: tidy_name(str(x)) if not pd.isna(x) else x)

        metadata = metadata.convert_dtypes()
        self.clinical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.clinical_metadata_path, engine='fastparquet')

    def create_panel(self):
        panel = pd.read_csv(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/AbPanel.csv')
        order = pd.read_csv(self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/markerStackOrder.csv')
        assert (panel.metaltag == order.Isotope).all()
        assert (panel.full == 1).all()

        # based on supplementary table 1
        panel.loc[panel.metaltag == 'Ho165', 'target'] = 'estrogen_receptor_alpha'
        panel.loc[panel.metaltag == 'Eu151', 'target'] = 'c_erb_b_2_her2_3B5'
        panel.loc[panel.metaltag == 'Tb159', 'target'] = 'c_erb_b_2_her2_D8F12'

        panel = panel.drop(columns=['ilastik', 'full', 'Unnamed: 10', 'tubenumber', 'to_sort', 'function_order'])
        panel = panel.rename(columns=dict(metaltag='metal_mass'))
        panel.loc[[37, 38], 'target'] = ['dna1', 'dna2']
        panel.loc[:, 'target'] = panel.target.map(tidy_name)

        panel = panel.convert_dtypes()
        panel.index.name = "channel_index"
        assert (panel.target.value_counts() == 1).all()

        panel_path = self.get_panel_path("published")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path)

