import re
from pathlib import Path

import numpy as np
import pandas as pd
from ai4bmr_core.utils.logging import get_logger
from ai4bmr_core.utils.tidy import tidy_names
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from skimage.io import imread, imsave
from ai4bmr_core.utils.saving import save_image, save_mask

from ..datamodels.Image import Image, Mask


# TODO: we still need to implement the single cell value extraction from the images based on the masks!
# at the moment we simply use the processed data from the publication.

class TNBC:
    """
    Download data from https://www.angelolab.com/mibi-data, unzip and place in `base_dir/01_raw`.
    Save Ionpath file as tnbc.
    The supplementary_table_1_path is downloaded as Table S1 from the paper.
    """
    name: str = 'TNBC'
    doi: str = '10.1016/j.cell.2018.08.039'

    # TODO: check why patient_id 30 is missing in `data`.
    #   but we have patient ids up to 44 in `data`.
    #   we only have images for patient ids up to 41
    def __init__(self,
                 base_dir: Path = Path('~/data/datasets/TNBC'),
                 mask_version: str = 'processed',
                 expression_from: str = 'processed',
                 verbose: int = 0):
        """
        :param base_dir:
        :param mask_version: `default` uses the cells that are present in the interior mask of the raw data. `v2` uses only the cells present in the processed data `cellData.csv`
        :param verbose:
        """
        if expression_from == 'processed':
            assert mask_version == 'processed'

        super().__init__()
        self.logger = get_logger('TNBC', verbose=verbose)

        # processed
        base_dir = Path(base_dir).expanduser().resolve()
        self.processed_dir = base_dir / '02_processed'
        self.images_dir = self.processed_dir / 'images'
        self.panel_path = self.images_dir / 'panel.parquet'
        self.masks_dir = self.processed_dir / 'masks'
        self.masks_version_dir = self.masks_dir / mask_version
        self.metadata_dir = self.processed_dir / 'metadata'
        self.misc = self.processed_dir / 'misc'
        self.misc.mkdir(exist_ok=True, parents=True)

        # raw
        self.raw_dir = base_dir / '01_raw'
        self.raw_masks_dir = self.raw_dir / 'TNBC_shareCellData'
        self.raw_images_dir = self.raw_dir / 'tnbc'
        self.processed_single_cell_data_path = self.raw_dir / 'TNBC_shareCellData' / 'cellData.csv'
        self.patient_data_path = self.raw_dir / 'TNBC_shareCellData' / 'patient_class.csv'
        self.supplementary_table_1_path = self.raw_dir / '1-s2.0-S0092867418311000-mmc1.xlsx'  # downloaded as Table S1 from paper

    def load(self):
        self.logger.info(f'Loading dataset {self.name}')

        patient_data = pd.read_csv(self.patient_data_path, header=None)
        patient_data.columns = ['patient_id', 'cancer_type_id']
        patient_data = patient_data.assign(sample_id=patient_data['patient_id']).set_index('sample_id')

        # TODO: verify mapping
        cancer_type_map = {0: 'cold', 1: 'mixed', 2: 'compartmentalized'}
        patient_data = patient_data.assign(cancer_type=patient_data['cancer_type_id'].map(cancer_type_map))

        processed_data = pd.read_csv(self.processed_single_cell_data_path)
        index_columns = ['SampleID', 'cellLabelInImage']
        processed_data = processed_data.set_index(index_columns)
        processed_data.index = processed_data.index.rename(dict(SampleID='sample_id', cellLabelInImage='object_id'))

        # filter for samples that are in both processed data and patient data
        # NOTE: patient 30 has been excluded from the analysis due to noise
        #   and there are patients 42,43,44 in the processed single cell data but we do not have images for them
        sample_ids = set(processed_data.index.get_level_values('sample_id')).intersection(set(patient_data.index))
        processed_data = processed_data.loc[list(sample_ids), :]
        patient_data = patient_data.loc[list(sample_ids), :]

        # mask metadata
        metadata_paths = list(self.masks_version_dir.glob(r'*.parquet'))
        if len(metadata_paths) == 40:
            metadata = pd.read_parquet(metadata_paths)
        else:
            self.create_metadata(processed_data)
            metadata_paths = list(self.masks_version_dir.glob(r'*.parquet'))
            assert len(metadata_paths) == 40
            metadata = pd.read_parquet(metadata_paths)

        # panel
        if self.panel_path.exists():
            panel = pd.read_parquet(self.panel_path)
        else:
            self.create_panel()
            panel = pd.read_parquet(self.panel_path)
        assert set(panel.target) - set(processed_data.columns) == set()

        # data
        processed_data = processed_data[panel.target]
        assert processed_data.isna().sum().sum() == 0

        # objects
        mask_paths = list(self.masks_version_dir.glob(r'*.tiff'))
        if len(mask_paths) != 40:
            self.create_filtered_masks(processed_data)
            mask_paths = list(self.masks_version_dir.glob(r'*.tiff'))
            assert len(mask_paths) == 40

        masks = {}
        for mask_path in mask_paths:
            metadata_path = mask_path.with_suffix('.parquet')
            assert metadata_path.exists()
            masks[int(mask_path.stem)] = Mask(id=int(mask_path.stem), data_path=mask_path, metadata_path=metadata_path)

        # images
        imgs_paths = list(self.images_dir.glob(r'*.tiff'))
        if len(imgs_paths) != 40:
            self.images_dir.mkdir(exist_ok=True, parents=True)
            imgs_paths = sorted(self.raw_images_dir.glob('*.tiff'))
            for img_path in imgs_paths:
                self.logger.debug(f'processing image {img_path.stem}')
                sample_id = re.search(r'Point(\d+).tiff', img_path.name).groups()[0]

                if (self.images_dir / f'{sample_id}.tiff').exists() or int(sample_id) not in sample_ids:
                    continue
                img = self.create_image(img_path, panel)
                save_image(img, self.images_dir / f'{sample_id}.tiff')

        images = {}
        for image_path in self.images_dir.glob('*.tiff'):
            images[int(image_path.stem)] = Image(id=int(image_path.stem), data_path=image_path,
                                                 metadata_path=self.panel_path)

        assert set(images) == set(masks)

        cols = processed_data.columns
        processed_data.columns = [tidy_names(col) for col in cols]
        panel = panel.assign(target_original_name=panel.target, target=panel.target.map(tidy_names))
        assert all(panel.target.tolist() == processed_data.columns)

        return dict(data=processed_data,
                    panel=panel,
                    metadata=metadata,
                    images=images,
                    masks=masks,
                    patient_data=patient_data)

    def create_image(self, path, panel) -> np.ndarray:
        self.logger.info('Creating images')
        img = imread(path)

        metadata = self.get_tiff_metadata(path)
        metadata = self.clean_metadata(metadata, panel)

        img = img[metadata.page]
        assert img.shape[0] == len(panel)

        return img

    def create_filtered_masks(self, data):
        self.logger.info('Filtering masks')

        self.masks_dir.mkdir(exist_ok=True, parents=True)
        mask_paths = sorted(list(self.raw_masks_dir.glob(r'p*_labeledcellData.tiff')))
        sample_ids = set(data.index.get_level_values('sample_id'))
        for mask_path in mask_paths:
            sample_id = re.match(r'p(\d+)', mask_path.stem).groups()
            assert len(sample_id) == 1
            sample_id = int(sample_id[0])

            self.logger.info(f'processing sample {sample_id}')
            if sample_id not in sample_ids:
                continue
            if (self.masks_version_dir / f'{sample_id}.tiff').exists():
                continue

            # load segmentation mask from raw data
            segm = imread(
                self.raw_images_dir / f'TA459_multipleCores2_Run-4_Point{sample_id}' / 'segmentation_interior.png')
            # load segmentation from processed data
            mask = imread(mask_path, plugin='tifffile')

            mask_filtered = mask.copy()
            # note: set region with no segmentation to background
            mask_filtered[segm == 0] = 0

            if mask_filtered.max() < 65535:
                mask_filtered = mask_filtered.astype('uint16')
            else:
                mask_filtered = mask_filtered.astype('uint32')

            assert 1 not in mask_filtered.flat
            masks_version_dir = self.masks_dir / 'processed'
            masks_version_dir.mkdir(exist_ok=True, parents=True)
            save_mask(mask_filtered, masks_version_dir / f'{sample_id}.tiff')

            # filter masks by objects in data
            # NOTE:
            #   - 0: cell boarders
            #   - 1: background
            #   - 2: cells
            #   (19, 4812) is background in the mask and removed after removing objects not in `data`
            objs_in_data = set(data.loc[sample_id, :].index)
            removed_objects = set(mask.flat) - set(objs_in_data) - {0, 1}
            map = {i: 0 for i in removed_objects}
            vfunc = np.vectorize(lambda x: map.get(x, x))
            mask_filtered_v2 = vfunc(mask)
            mask_filtered_v2[mask_filtered_v2 == 1] = 0

            if mask_filtered_v2.max() < 65535:
                mask_filtered_v2 = mask_filtered_v2.astype('uint16')
            else:
                mask_filtered_v2 = mask_filtered_v2.astype('uint32')

            assert set(mask_filtered_v2.flat) - objs_in_data == {0}
            assert 1 not in mask_filtered_v2.flat
            masks_version_dir = self.masks_dir / 'v2'
            masks_version_dir.mkdir(exist_ok=True, parents=True)
            save_mask(mask_filtered_v2, masks_version_dir / f'{sample_id}.tiff')

            # note: map objects in mask that are not present in data to 2
            objs_in_segm = set(mask[segm == 1])
            map = {i: 2 for i in objs_in_segm - objs_in_data}
            assert objs_in_data - objs_in_segm == set()
            map.update({i: 1 for i in objs_in_data.intersection(objs_in_segm)})
            z = mask.copy()
            z[segm != 1] = 0
            vfunc = np.vectorize(lambda x: map.get(x, x))
            z = vfunc(z)

            # diagnostic plot
            fig, axs = plt.subplots(2, 2, dpi=400)
            axs = axs.flat

            axs[0].imshow(mask > 1, interpolation=None, cmap='grey')
            axs[0].set_title('Segmentation from processed data', fontsize=8)

            axs[1].imshow(mask_filtered > 0, interpolation=None, cmap='grey')
            axs[1].set_title('Segmentation from raw data', fontsize=8)

            axs[2].imshow(z, interpolation=None)
            axs[2].set_title('Objects missing in `cellData.csv`', fontsize=8)

            axs[3].imshow(mask_filtered_v2 > 0, interpolation=None, cmap='grey')
            axs[3].set_title('Segmentation from processed\nfiltered with `cellData.csv`', fontsize=8)

            for ax in axs: ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(self.misc / f'{sample_id}.png')
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

        panel = panel.rename(columns={'Mass channel': 'mass_channel'})

        # note: Biotin has not Start,Stop,NoiseT
        #   for PD-L1, we replace the Mass channel and Isotope with the values in Biotin
        panel.loc[panel.target == 'PDL1-biotin', 'mass_channel'] = \
            panel.loc[panel.target == 'Biotin', 'mass_channel'].iloc[0]
        panel.loc[panel.target == 'PDL1-biotin', 'Isotope'] = panel.loc[panel.target == 'Biotin', 'Isotope'].iloc[0]
        panel = panel[panel.Start.notna()]

        target_map = {'PDL1-biotin': 'PD-L1', 'Ki-67': 'Ki67'}
        panel = panel.assign(target=panel.target.replace(target_map))

        panel = panel.sort_values('mass_channel', ascending=True)
        panel = panel.assign(page=range(len(panel)))

        # extract original page numbers for markers
        img_path = sorted(list(self.raw_images_dir.glob('*.tiff')))[0]
        metadata = self.get_tiff_metadata(img_path)
        metadata = self.clean_metadata(metadata, panel)
        metadata = metadata.rename(columns={'page': 'page_original'})

        panel = panel.merge(metadata[['page_original', 'mass_channel']], on='mass_channel')

        panel = panel.convert_dtypes()
        self.panel_path.parent.mkdir(exist_ok=True, parents=True)
        panel.to_parquet(self.panel_path)

    def get_tiff_metadata(self, path):
        import tifffile
        tag_name = 'PageName'
        metadata = pd.DataFrame()
        metadata.index.name = 'page'
        with tifffile.TiffFile(path) as tif:
            for page_num, page in enumerate(tif.pages):
                page_name_tag = page.tags.get(tag_name)
                metadata.loc[page_num, tag_name] = page_name_tag.value
        metadata = metadata.assign(PageName=metadata.PageName.str.strip())
        metadata = metadata.PageName.str.extract(r'(?P<target>\w+)\s+\((?P<mass_channel>\d+)')
        return metadata

    def create_metadata(self, data):
        self.logger.info('Creating metadata')
        metadata_cols = ['cellSize', 'tumorYN', 'tumorCluster', 'Group', 'immuneCluster', 'immuneGroup']
        metadata = data[metadata_cols]

        grp_map = {1: 'unidentified', 2: 'immune', 3: 'endothelial', 4: 'Mmsenchymal_like', 5: 'tumor',
                   6: 'keratin_positive_tumor'}
        metadata = metadata.assign(group_name=metadata['Group'].map(grp_map))

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

        label_name = metadata.group_name.copy()
        filter = metadata.immune_group_name.notna()
        label_name[filter] = metadata.immune_group_name[filter]
        metadata = metadata.assign(label_name=label_name)
        label_id_map = dict(zip(label_name.unique(), range(len(label_name.unique()))))
        metadata = metadata.assign(label_id=metadata.label_name.map(label_id_map))

        metadata = metadata.convert_dtypes()
        self.masks_version_dir.mkdir(exist_ok=True, parents=True)
        for grp_name, grp_data in metadata.groupby('sample_id'):
            grp_data.to_parquet(self.masks_version_dir / f'{grp_name}.parquet')

    @staticmethod
    def clean_metadata(metadata, panel):
        metadata = metadata.assign(mass_channel=metadata.mass_channel.astype('int'))
        filter = metadata.mass_channel.isin(panel['mass_channel'])
        metadata = metadata[filter]
        metadata = metadata.rename(columns={'target': 'target_from_tiff'})
        map = panel.set_index('mass_channel')['target'].to_dict()
        metadata = metadata.assign(target=metadata.mass_channel.map(map))
        # align with panel marker order, i.e. sort by mass channel
        panel = panel.sort_values('page')  # ensure panel is sorted by page
        metadata = metadata.reset_index().set_index('target').loc[panel.target]
        return metadata

    def reproduce_paper_figure_2E(self) -> Figure:
        """Try to reproduce the figure from the paper. Qualitative comparison looks good if data is loaded from
        """

        import pandas as pd
        import seaborn as sns
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        import colorcet as cc

        data = self.load()
        df = data['data']

        # %%
        cols = ['FoxP3', 'CD4', 'CD3', 'CD56', 'CD209', 'CD20', 'HLA-DR', 'CD11c', 'CD16', 'CD68', 'CD11b', 'MPO',
                'CD45',
                'CD31',
                'SMA', 'Vimentin', 'Beta catenin', 'Pan-Keratin', 'p53', 'EGFR', 'Keratin17', 'Keratin6']
        pdat = df[cols]
        pdat = pdat[pdat.index.get_level_values('sample_id') <= 41]

        pdat, row_colors = pdat.align(data['metadata'], join='left', axis=0)
        pdat = pdat.assign(group_name=pd.Categorical(row_colors.group_name,
                                                     categories=['immune', 'endothelial', 'Mmsenchymal_like', 'tumor',
                                                                 'keratin_positive_tumor', 'unidentified']))
        pdat = pdat.assign(immune_group_name=pd.Categorical(row_colors.immune_group_name,
                                                            categories=['Tregs', 'CD4 T', 'CD8 T', 'CD3 T', 'NK', 'B',
                                                                        'Neutrophils',
                                                                        'Macrophages', 'DC', 'DC/Mono', 'Mono/Neu',
                                                                        'Other immune']))
        pdat = pdat.set_index(['group_name', 'immune_group_name'], append=True)

        color_map = {'immune': np.array((187, 103, 30, 0)) / 255,
                     'endothelial': np.array((253, 142, 142, 0)) / 255,
                     'Mmsenchymal_like': np.array((1, 56, 54, 0)) / 255,
                     'tumor': np.array((244, 201, 254, 0)) / 255,
                     'keratin_positive_tumor': np.array((128, 215, 230, 0)) / 255,
                     'unidentified': np.array((255, 255, 255, 0)) / 255,
                     }
        row_colors = row_colors.assign(group_name=row_colors.group_name.map(color_map))

        color_map = {k: cc.glasbey_category10[i] for i, k in enumerate(row_colors.immune_group_name.unique())}
        row_colors = row_colors.assign(immune_group_name=row_colors.immune_group_name.map(color_map))
        row_colors = row_colors[['group_name', 'immune_group_name']]

        pdat = pdat.sort_values(['group_name', 'immune_group_name'])

        # %% filter
        pdat = pdat[pdat.index.get_level_values('group_name') == 'immune']
        pdat = pdat[
            ['FoxP3', 'CD4', 'CD3', 'CD56', 'CD209', 'CD20', 'HLA-DR', 'CD11c', 'CD16', 'CD68', 'CD11b', 'MPO', 'CD45']]

        # %%
        pdat = pd.DataFrame(MinMaxScaler().fit_transform(pdat), index=pdat.index, columns=pdat.columns)
        cg = sns.clustermap(pdat,
                            cmap='bone',
                            row_colors=row_colors,
                            col_cluster=False,
                            row_cluster=False,
                            # figsize=(20, 20)
                            )
        cg.ax_heatmap.set_facecolor('black')
        cg.ax_heatmap.set_yticklabels([])
        cg.figure.tight_layout()
        cg.figure.show()
        return cg.figure
