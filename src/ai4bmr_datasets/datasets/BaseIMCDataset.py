from pathlib import Path

class BaseIMCDataset:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def __len__(self):
        pass

    def process(self):
        self.create_panel()
        self.create_metadata()
        self.create_features()
        self.create_images()
        self.create_masks()

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