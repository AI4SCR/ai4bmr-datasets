from pathlib import Path
from ..datamodels.Image import Image, Mask
import pandas as pd

class BaseIMCDataset:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

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
            'sample_id': sample_id,
            'sample': self.samples.loc[sample_id],
            'image': self.images[sample_id],
            'mask': self.masks[sample_id],
            'panel': self.panel,
            'intensity': self.intensity.loc[sample_id],
            'spatial': self.spatial.loc[sample_id],
        }

    def load(self, image_version: str = 'published', mask_version: str = 'published', features_as_published: bool = True):
        self.logger.info(f'Loading dataset {self.name}')

        if features_as_published:
            if not (image_version == 'published' and mask_version == 'published'):
                raise ValueError('Features can only be loaded `as_published=True` when `image_version=published` and `mask_version=published`')
            intensity_path = self.intensity_dir / 'published'
            spatial_path = self.spatial_dir / 'published'
        else:
            intensity_path = self.get_intensity_dir(image_version=image_version, mask_version=mask_version)
            spatial_path = self.get_spatial_dir(mask_version=mask_version)

        panel_path = self.get_panel_path(image_version=image_version)
        panel = pd.read_parquet(panel_path)
        samples = pd.read_parquet(self.sample_metadata_path)

        # TODO: only load features, masks if they exist
        intensity = pd.read_parquet(intensity_path)
        spatial = pd.read_parquet(spatial_path)
        intensity, spatial = intensity.align(spatial, join='outer', axis=0)

        assert intensity.isna().any().any() == False
        assert spatial.isna().any().any() == False
        assert set(panel.target) == set(intensity.columns)

        sample_ids = intensity.index.get_level_values('sample_id').unique()
        sample_ids = sample_ids.sort_values()

        self.samples = samples.loc[sample_ids, :]
        self.intensity = intensity
        self.spatial = spatial
        self.panel = panel

        masks = {}
        masks_dir = self.get_masks_dir(mask_version=mask_version)

        images = {}
        images_dir = self.get_images_dir(image_version=image_version)

        if sample_ids is not None:
            for sample_id in sample_ids:
                mask_path = masks_dir / f'{sample_id}.tiff'
                image_path = images_dir / f'{sample_id}.tiff'

                masks[sample_id] = Mask(id=int(mask_path.stem), data_path=mask_path, metadata_path=None)
                images[sample_id] = Image(id=int(image_path.stem), data_path=image_path, metadata_path=panel_path)
        else:
            masks = {int(p.stem): Mask(id=int(p.stem), data_path=p, metadata_path=None) for p in masks_dir.glob('*.tiff')}
            images = {int(p.stem): Mask(id=int(p.stem), data_path=p, metadata_path=None) for p in images_dir.glob('*.tiff')}

        assert set(images) == set(masks)
        self.images = images
        self.masks = masks

        return {
            'samples': self.panel,
            'images': self.images,
            'masks': self.masks,
            'panel': self.panel,
            'intensity': self.intensity,
            'spatial': self.spatial
        }

    def process(self):
        self.create_panel()
        self.create_metadata()
        self.create_images()
        self.create_masks()
        self.create_features()

    def create_images(self):
        pass

    def create_masks(self):
        pass

    def create_metadata(self):
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
        pass

    @property
    def name(self):
        return 'name'

    @property
    def id(self):
        return 'id'

    @property
    def doi(self):
        return 'doi'

    @property
    def raw_dir(self):
        return self.base_dir / '01_raw'

    @property
    def processed_dir(self):
        return self.base_dir / '02_processed'

    @property
    def images_dir(self):
        return self.processed_dir / 'images'

    def get_images_dir(self, image_version: str):
        return self.images_dir / image_version

    @property
    def masks_dir(self):
        return self.processed_dir / 'masks'

    def get_masks_dir(self, mask_version: str):
        return self.masks_dir / mask_version

    @property
    def metadata_dir(self):
        return self.processed_dir / 'metadata'

    def get_samples_path(self):
        return self.metadata_dir / 'samples.parquet'

    @property
    def annotations_dir(self):
        return self.metadata_dir / 'annotations'

    def get_annotations_path(self, sample_name: str, image_version_name: str, mask_version_name: str):
        version_name = f'{image_version_name}-{mask_version_name}'
        return self.annotations_dir / version_name / f'{sample_name}.parquet'


    @property
    def graphs_dir(self):
        return self.processed_dir / 'graphs'

    def get_graph_path(self, sample_name: str, mask_version: str, graph_type: str):
        return self.graphs_dir / mask_version / graph_type / f'{sample_name}.npy'

    @property
    def features_dir(self):
        return self.processed_dir / 'features'

    @property
    def spatial_dir(self):
        return self.features_dir / 'spatial'

    def get_spatial_dir(self, mask_version: str):
        return self.spatial_dir / mask_version

    @property
    def intensity_dir(self):
        return self.features_dir / 'intensity'

    def get_intensity_dir(self, image_version: str, mask_version: str):
        version_name = f'{image_version}-{mask_version}'
        return self.intensity_dir / version_name

    @property
    def sample_metadata_path(self):
        return self.metadata_dir / 'samples.parquet'

    def get_panel_path(self, image_version: str):
        images_dir = self.get_images_dir(image_version)
        return images_dir / 'panel.parquet'