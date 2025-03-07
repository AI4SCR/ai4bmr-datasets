from pathlib import Path


class BaseIMCDataset:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def __len__(self):
        pass

    def create_images(self):
        pass

    def create_masks(self):
        pass

    def create_metadata(self):
        pass

    def create_graphs(self):
        pass

    def create_features(self):
        pass

    @property
    def name(self):
        return 'name'

    @property
    def id(self):
        return 'id'

    @property
    def raw_dir(self):
        return self.base_dir / '01_raw'

    @property
    def processed_dir(self):
        return self.base_dir / '02_processed'

    @property
    def images_dir(self):
        return self.processed_dir / 'images'

    @property
    def masks_dir(self):
        return self.processed_dir / 'masks'

    @property
    def metadata_dir(self):
        return self.processed_dir / 'metadata'

    @property
    def graphs_dir(self):
        return self.processed_dir / 'graphs'

    @property
    def features_dir(self):
        return self.processed_dir / 'features'

    @property
    def sample_metadata_path(self):
        return self.metadata_dir / 'samples.parquet'

    @property
    def panel_path(self):
        return self.metadata_dir / 'panel.parquet'