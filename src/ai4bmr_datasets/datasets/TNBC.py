from pathlib import Path

import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from .BaseDataset import BaseDataset
from ..datamodels.Image import Image, Mask
from ai4bmr_core.utils.symlink import symlink_dir_files
from ai4bmr_core.utils.logging import get_logger

class TNBC:
    """
    Download data from https://www.angelolab.com/mibi-data, unzip and place in `base_dir/01_raw`.
    Save Ionpath file as tnbc.
    The supplementary_table_1_path is downloaded as Table S1 from the paper.
    """
    name: str = 'TNBC'
    # TODO: check why patient_id 30 is missing in `data`.
    #   but we have patient ids up to 44 in `data`.
    #   we only have images for patient ids up to 41
    def __init__(self,
                 base_dir: Path = Path(
                     '/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/datasets/TNBC'),
                 verbose: int = 0):
        super().__init__()
        self.logger = get_logger('TNBC', verbose=verbose)

        # processed
        self.processed_dir = base_dir / '02_processed'
        self.images_dir = self.processed_dir / 'images'
        self.panel_path = self.images_dir / 'panel.parquet'
        self.masks_dir = self.processed_dir / 'masks'
        self.metadata_dir = self.processed_dir / 'metadata'

        # raw
        self.raw_dir = base_dir / '01_raw'
        self.raw_masks_dir = self.raw_dir / 'TNBC_shareCellData'
        self.raw_images_dir = self.raw_dir / 'tnbc'
        self.single_cell_data_path = self.raw_dir / 'TNBC_shareCellData' / 'cellData.csv'
        self.patient_data_path = self.raw_dir / 'TNBC_shareCellData' / 'patient_class.csv'
        self.supplementary_table_1_path = self.raw_dir / '1-s2.0-S0092867418311000-mmc1.xlsx'  # downloaded as Table S1 from paper


    def load(self):
        self.logger.info(f'Loading dataset {self.name}')

        patient_data = pd.read_csv(self.patient_data_path)
        patient_data.columns = ['patient_id', 'cancer_type_id']

        # TODO: verify mapping
        cancer_type_map = {0: 'cold', 1: 'mixed', 2: 'compartmentalized'}
        patient_data = patient_data.assign(cancer_type=patient_data['cancer_type_id'].map(cancer_type_map))

        data = pd.read_csv(self.single_cell_data_path)
        index_columns = ['SampleID', 'cellLabelInImage']
        data = data.set_index(index_columns)
        data.index = data.index.rename(dict(SampleID='sample_id', cellLabelInImage='object_id'))

        # metadata
        if self.metadata_dir.exists():
            metadata = pd.read_parquet(self.metadata_dir)
        else:
            self.create_metadata(data)
            metadata = pd.read_parquet(self.metadata_dir)
        # 19, 23, 24, (34),
        idcs = [19, 23, 24]
        # panel
        if self.panel_path.exists():
            panel = pd.read_parquet(self.panel_path)
        else:
            self.create_panel()
            panel = pd.read_parquet(self.panel_path)
        assert set(panel.target) - set(data.columns) == set()

        # data
        data = data[panel.target]
        assert data.isna().sum().sum() == 0

        # objects
        mask_paths = list(self.masks_dir.glob(r'*.tiff'))
        if len(mask_paths) != 40:
            self.create_filtered_masks(data)
            mask_paths = list(self.masks_dir.glob(r'*.tiff'))
            assert len(mask_paths) == 40

        masks = {}
        for mask_path in mask_paths:
            metadata_path = self.metadata_dir / f'{mask_path.stem}.parquet'
            assert metadata_path.exists()
            masks[int(mask_path.stem)] = Mask(id=int(mask_path.stem), data_path=mask_path, metadata_path=metadata_path)

        # images
        imgs_paths = list(self.images_dir.glob(r'*.tiff'))
        if len(imgs_paths) != 41:
            # self.create_images(panel)
            self.images_dir.mkdir(exist_ok=True, parents=True)
            paths = symlink_dir_files(src_dir=self.raw_images_dir, dest_dir=self.images_dir, glob='*.tiff')
            for path in paths:
                sample_id = re.search(r'Point(\d+).tiff', path.name).groups()[0]
                path.rename(path.with_stem(sample_id))

        images = {}
        for image_path in self.images_dir.glob('*.tiff'):
            images[int(image_path.stem)] = Image(id=int(image_path.stem), data_path=image_path, metadata_path=self.panel_path)

        return dict(data=data,
                    panel=panel,
                    metadata=metadata,
                    images=images,
                    masks=masks,
                    patient_data=patient_data)

    def create_images(self, panel):
        self.logger.info('Creating images')
        pass

    def create_filtered_masks(self, data):
        self.logger.info('Filtering masks')
        # NOTE:
        #   - 0: cell boarders
        #   - 1: background
        #   - 2: cells
        #   (19, 4812) is background in the mask and removed after removing objects not in `data`
        self.masks_dir.mkdir(exist_ok=True, parents=True)
        mask_paths = list(self.raw_masks_dir.glob(r'p*_labeledcellData.tiff'))
        sample_ids = set(data.index.get_level_values('sample_id'))
        for mask_path in mask_paths:
            sample_id = re.match(r'p(\d+)', mask_path.stem).groups()
            assert len(sample_id) == 1
            sample_id = int(sample_id[0])

            if sample_id not in sample_ids:
                continue
            if (self.masks_dir / f'{sample_id}.png').exists():
                continue

            mask = imread(mask_path, plugin='tifffile')

            # filter masks
            objs_in_data = set(data.loc[sample_id, :].index)
            removed_objects = set(mask.flat) - set(objs_in_data) - {0, 1}
            map = {i: 1 for i in removed_objects}
            vfunc = np.vectorize(lambda x: map.get(x, x))
            mask_filtered = vfunc(mask)
            mask_filtered[mask_filtered == 1] = 0

            assert set(mask_filtered.flat) - objs_in_data == {0}
            assert 1 not in mask_filtered.flat
            imsave(self.masks_dir / f'{sample_id}.tiff', mask_filtered)

            # diagnostic plot
            fig, axs = plt.subplots(1, 2, dpi=400)
            axs[0].imshow(mask > 1, interpolation=None, cmap='grey')
            axs[1].imshow(mask_filtered > 0, interpolation=None, cmap='grey')
            for ax in axs: ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(self.masks_dir / f'{sample_id}.png')
            plt.close(fig)

    def create_panel(self):
        self.logger.info('Creating panel')
        panel_1 = pd.read_excel(self.supplementary_table_1_path, header=3)[:31]
        panel_1.columns = panel_1.columns.str.strip()
        panel_1 = panel_1.rename(columns={'Antibody target': 'target'})
        panel_2 = pd.read_excel(self.supplementary_table_1_path, header=37)
        panel_2.columns = panel_2.columns.str.strip()
        panel_2 = panel_2.rename(columns={'Label': 'target', 'Titer': 'Titer (µg/mL)'})
        panel = pd.concat([panel_1, panel_2])

        # note: Biotin has not Start,Stop,NoiseT
        #   for PD-L1, we replace the Mass channel and Isotope with the values in Biotin
        panel.loc[panel.target == 'PDL1-biotin', 'Mass channel'] = \
        panel.loc[panel.target == 'Biotin', 'Mass channel'].iloc[0]
        panel.loc[panel.target == 'PDL1-biotin', 'Isotope'] = panel.loc[panel.target == 'Biotin', 'Isotope'].iloc[0]
        panel = panel[panel.Start.notna()]

        target_map = {'PDL1-biotin': 'PD-L1', 'Ki-67': 'Ki67'}
        panel = panel.assign(target=panel.target.replace(target_map))

        panel = panel.convert_dtypes()
        self.panel_path.parent.mkdir(exist_ok=True, parents=True)
        panel.to_parquet(self.panel_path)

    def create_metadata(self, data):
        self.logger.info('Creating metadata')
        metadata_cols = ['cellSize', 'tumorYN', 'tumorCluster', 'Group', 'immuneCluster', 'immuneGroup']
        metadata = data[metadata_cols]

        grp_map = {1: 'unidentified', 2: 'immune', 3: 'endothelial', 4: 'Mmsenchymal_like', 5: 'tumor',
                   6: 'keratin_positive_tumor'}
        metadata = metadata.assign(group_name= metadata['Group'].map(grp_map))

        immune_grp_map = {1: "Tregs", 2: "CD4 T", 3: "CD8 T", 4: "CD3 T", 5: "NK", 6: "B", 7: "Neutrophils",
                          8: "Macrophages", 9: "DC", 10: "DC/Mono", 11: "Mono/Neu", 12: "Other immune"}
        metadata = metadata.assign(immune_group_name=metadata['immuneGroup'].map(immune_grp_map))

        col_map = dict(
            cellSize='cell_size',
            tumorYN='tumor_yn',
            tumorCluster='tumor_cluster_id',
            Group='group_id',
            immuneCluster='immune_cluster_id',
            immuneGroup='immune_group_id')
        metadata = metadata.rename(columns=col_map)
        metadata = metadata.assign(tumor_yn=metadata.tumor_yn.astype('bool'))

        metadata = metadata.convert_dtypes()
        self.metadata_dir.mkdir(exist_ok=True, parents=True)
        for grp_name, grp_data in metadata.groupby('sample_id'):
            grp_data.to_parquet(self.metadata_dir / f'{grp_name}.parquet')