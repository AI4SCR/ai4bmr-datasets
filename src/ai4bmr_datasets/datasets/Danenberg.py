import re
from pathlib import Path

import pandas as pd
from ai4bmr_datasets.utils.tidy import tidy_name
from ai4bmr_datasets.utils import io
import numpy as np
from loguru import logger

from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset
from ai4bmr_datasets.utils.download import unzip_recursive


class Danenberg2022(BaseIMCDataset):
    name = "Danenberg2022"
    id = "Danenberg2022"
    doi = "10.1038/s41588-022-01041-y"

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

    def prepare_data(self):
        """
        Prepares the Danenberg2022 dataset by downloading, processing R data,
        and creating various data components (panel, images, masks, metadata, intensity, clinical metadata).
        """
        self.download()
        self.process_rdata()

        self.create_panel()
        self.create_images()
        self.create_masks()
        self.create_metadata_and_intensity()
        self.create_clinical_metadata()

        self.create_annotated(version_name='annotated_cell', mask_version='published_cell')
        self.create_annotated(version_name='annotated_nucleus', mask_version='published_nucleus')

    def download(self, force: bool = False):
        """
        Downloads the raw data files for the Danenberg2022 dataset from Zenodo.

        Args:
            force (bool): If True, forces re-download even if files already exist.
        """
        import requests
        import shutil
        from ai4bmr_datasets.utils.download import download_file_map, unzip_recursive

        download_dir = self.raw_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "https://zenodo.org/records/6036188/files/MBTMEStrIMCPublic.zip?download=1": download_dir / 'mbtme_imc_public.zip',
            'https://zenodo.org/records/15615709/files/correctedPublicMasks.zip?download=1': download_dir / 'corrected_public_masks.zip',
        }

        download_file_map(file_map, force=force)

        # Extract zip files
        for target_path in file_map.values():
            unzip_recursive(target_path)

        shutil.rmtree(self.raw_dir / '__MACOSX', ignore_errors=True)

        logger.info("✅ Download and extraction completed.")

    def get_metabric_id(self, path: str) -> str:
        """
        Extracts the METABRIC ID from a given file path.

        Args:
            path (str): The file path.

        Returns:
            str: The extracted METABRIC ID.
        """
        return re.search(r'MB\d+', path).group(0)

    def get_sample_id(self, path: str) -> str:
        """
        Extracts the sample ID from a given file path.

        Args:
            path (str): The file path.

        Returns:
            str: The extracted sample ID in the format 'MB_X_Y'.
        """
        match = re.search(r'MB(\d+)_(\d+)', path)
        return f'MB_{match.group(1)}_{match.group(2)}'

    def get_mask_version(self, path: str) -> str:
        """
        Extracts the mask version from a given file path.

        Args:
            path (str): The file path.

        Returns:
            str: The extracted mask version.
        """
        match = re.search(r'MB(\d+)_(\d+)_(\w+)Mask\.tiff', path)
        return match.group(3)

    def process_rdata(self):
        """
        Processes R data files (FST) to extract and convert clinical and single-cell data
        into Parquet format.
        """
        import subprocess
        import textwrap

        clinical_path = self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/IMCClinical.parquet'
        sc_path = self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/SingleCells.parquet'

        if sc_path.exists() and clinical_path.exists():
            logger.info("Clinical and SingleCells data already processed. Skipping.")
            return

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
        write_parquet(df, "{clinical_path}")
        
        df <- read.fst("{self.raw_dir}/mbtme_imc_public/MBTMEIMCPublic/SingleCells.fst")
        write_parquet(df, "{sc_path}")
        """
        r_script = textwrap.dedent(r_script)

        # TODO: this will run only on slurm
        # subprocess.run(["Rscript", "-e", r_script], check=True)
        # Wrap the command in a shell to load modules and run the R script
        shell_command = textwrap.dedent(f"""
            # module load r-light/4.4.1
            Rscript -e '{r_script.strip()}'
        """)

        # Run using bash shell to ensure module environment is available
        subprocess.run(["bash", "-c", shell_command], check=True)

    def create_images(self):
        """
        Creates and saves processed image files for the Danenberg2022 dataset.

        This method reads raw image files, filters for 'FullStack' images,
        and saves them as TIFF files.
        """
        images_dir = self.raw_dir / 'mbtme_imc_public/MBTMEIMCPublic/Images'
        img_paths = [
            p for p in images_dir.rglob('*.tiff')
            if not p.name.startswith('.')
        ]
        img_paths = sorted(filter(lambda x: 'FullStack' in str(x), img_paths))

        save_dir = self.images_dir / self.get_version_name(version='published')
        save_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            try:
                sample_id = self.get_sample_id(str(img_path))
            except Exception as e:
                print('debug', img_path)  # TODO: remove

            save_path = save_dir / f"{sample_id}.tiff"
            if save_path.exists():
                logger.info(f"Image file {save_path} already exists. Skipping.")
                continue
            else:
                logger.info(f"Creating image {save_path}")

            img = io.imread(img_path)
            assert len(img) == 39

            img = img.astype(np.float32)
            io.imsave(save_path=save_path, img=img)

    def create_masks(self):
        """
        Creates and saves processed mask files for the Danenberg2022 dataset.

        This method reads raw mask files, extracts sample and mask version information,
        and saves the processed masks as TIFF files.
        """
        images_dir = self.raw_dir / 'corrected_public_masks' / 'correctedPublicMasks'
        img_paths = list(images_dir.rglob('*.tiff'))
        mask_paths = sorted(filter(lambda x: 'Mask' in str(x) and not x.name.startswith('.'), img_paths))

        for mask_path in mask_paths:
            sample_id = self.get_sample_id(str(mask_path))

            version = self.get_mask_version(str(mask_path)).strip().lower()
            version = f'published_{version}'
            save_dir = self.masks_dir / self.get_version_name(version=version)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample_id}.tiff"

            if save_path.exists():
                logger.info(f"Mask file {save_path} already exists. Skipping.")
                continue
            else:
                logger.info(f"Creating mask {save_path}")

            mask = io.imread(mask_path)
            assert mask.ndim == 2

            io.save_mask(save_path=save_path, mask=mask)

    def create_metadata_and_intensity(self):
        """
        Creates and saves processed metadata and intensity features for the Danenberg2022 dataset.

        This method reads raw single-cell data, processes it to extract sample and object IDs,
        renames intensity columns, and saves the processed intensity and metadata
        as Parquet files, grouped by sample ID.
        """
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

        # note: some marker are not as in the panel.target.
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
        """
        Creates and saves the clinical metadata for the Danenberg2022 dataset.

        This method reads raw clinical data, processes it to align with sample IDs,
        applies tidy naming conventions, and saves the consolidated clinical metadata
        as a Parquet file.
        """

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
        """
        Creates and saves the panel (antibody panel) for the Danenberg2022 dataset.

        This method reads raw antibody panel data, processes it to align with target names,
        and saves the consolidated panel as a Parquet file.
        """
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


