import re
from pathlib import Path
from ai4bmr_core.utils.tidy import tidy_name
from ai4bmr_core.utils.logging import get_logger
from .BaseIMCDataset import BaseIMCDataset
import pandas as pd


class NSCLCv2(BaseIMCDataset):

    def __init__(self, base_dir: Path):
        self.logger = get_logger('NSCLCv2')  # TODO: move to BaseIMCDataset init
        super().__init__(base_dir)
        self.base_dir = base_dir

        # raw file paths
        self.raw_panel_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/NSCLC/panel.csv')
        self.raw_samples_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/NSCLC/images.csv')
        self.raw_images_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/NSCLC/img')

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

    def create_images(self):
        # TODO: extract images from mcd with Theos low-level functions
        import shutil
        import re

        images_dir = self.get_images_dir('published')
        images_dir.mkdir(parents=True, exist_ok=True)

        images_paths = list(self.raw_images_dir.glob('*.tiff'))
        for img_path in images_paths:
            sample_id = re.search(r'.+_TMA_(\d+_\w_.+).tiff', img_path.name).group(1)
            save_path = images_dir / f"{sample_id}.tiff"
            assert not save_path.exists(), f"Image {save_path} already exists"
            _ = shutil.move(img_path, images_dir / f"{sample_id}.tiff")

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

    def create_metadata(self):
        import re
        samples = pd.read_csv(self.raw_samples_path)

        extract_sample_id = lambda x: re.search(r'.+_TMA_(\d+_\w_.+).tiff', x).group(1)
        samples['sample_id'] = samples['image'].map(extract_sample_id)
        assert samples['sample_id'].duplicated().any() == False
        samples = samples.set_index('sample_id')

        samples.columns = samples.columns.map(tidy_name)
        samples = samples.convert_dtypes()

        save_path = self.get_samples_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(save_path)

    def create_panel(self):
        # TODO: extract panel from mcd with Theos low-level functions
        panel = pd.read_csv(self.raw_panel_path)

        panel['keep'] = panel.name.str.split('_').map(len) > 1
        panel['target'] = panel.name.str.split('_').str[0].map(tidy_name)

        panel.columns = panel.columns.map(tidy_name)
        panel = panel.convert_dtypes()
        panel['page'] = range(len(panel))
        panel.index.name = 'channel_index'

        # TODO: remove after moving to low-level as this is only required by steinbock
        panel.to_csv(self.raw_panel_path, index=True)
        panel = panel[panel.keep]
        panel = panel.reset_index(drop=True)
        panel.index.name = 'channel_index'

        panel_path = self.get_panel_path('published')
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(self.get_panel_path('published'))

    @property
    def metadata(self):
        return self.samples

    @property
    def name(self):
        return "name"

    @property
    def id(self):
        return "id"

    @property
    def doi(self):
        return "doi"

    @property
    def raw_dir(self):
        return self.base_dir / "01_raw"

    @property
    def processed_dir(self):
        return self.base_dir / "02_processed"

    @property
    def images_dir(self):
        return self.processed_dir / "images"

    def get_images_dir(self, image_version: str):
        return self.images_dir / image_version

    @property
    def masks_dir(self):
        return self.processed_dir / "masks"

    def get_masks_dir(self, mask_version: str):
        return self.masks_dir / mask_version

    @property
    def metadata_dir(self):
        return self.processed_dir / "metadata"

    def get_samples_path(self):
        return self.metadata_dir / "samples.parquet"

    @property
    def annotations_dir(self):
        return self.metadata_dir / "annotations"

    def get_annotations_path(
            self, sample_name: str, image_version_name: str, mask_version_name: str
    ):
        version_name = f"{image_version_name}-{mask_version_name}"
        return self.annotations_dir / version_name / f"{sample_name}.parquet"

    @property
    def graphs_dir(self):
        return self.processed_dir / "graphs"

    def get_graph_path(self, sample_name: str, mask_version: str, graph_type: str):
        return self.graphs_dir / mask_version / graph_type / f"{sample_name}.npy"

    @property
    def features_dir(self):
        return self.processed_dir / "features"

    @property
    def spatial_dir(self):
        return self.features_dir / "spatial"

    def get_spatial_dir(self, mask_version: str):
        return self.spatial_dir / mask_version

    @property
    def intensity_dir(self):
        return self.features_dir / "intensity"

    def get_intensity_dir(self, image_version: str, mask_version: str):
        version_name = f"{image_version}-{mask_version}"
        return self.intensity_dir / version_name

    @property
    def sample_metadata_path(self):
        return self.metadata_dir / "samples.parquet"

    def get_panel_path(self, image_version: str):
        images_dir = self.get_images_dir(image_version)
        return images_dir / "panel.parquet"
