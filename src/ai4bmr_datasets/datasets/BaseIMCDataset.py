from pathlib import Path
from collections import namedtuple

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
        # TODO: this breaks if there is no self.samples
        sample_id = self.samples.index[idx]
        return {
            "sample_id": sample_id,
            "sample": self.samples.loc[sample_id],
            "image": self.images[sample_id],
            "mask": self.masks[sample_id],
            "panel": self.panel,
            "intensity": self.intensity.loc[sample_id],
            "spatial": self.spatial.loc[sample_id],
        }

    def resolve_paths(
            self, image_version: str, mask_version: str, features_as_published: bool = True
    ):
        if features_as_published:
            if not (image_version == "published" and mask_version == "published"):
                raise ValueError(
                    "Features can only be loaded `as_published=True` when `image_version=published` and `mask_version=published`"
                )
            intensity_path = self.intensity_dir / "published"
            spatial_path = self.spatial_dir / "published"
        else:
            intensity_path = self.get_intensity_dir(
                image_version=image_version, mask_version=mask_version
            )
            spatial_path = self.get_spatial_dir(mask_version=mask_version)

        samples_path = self.sample_metadata_path

        masks_dir = self.get_masks_dir(mask_version=mask_version)
        images_dir = self.get_images_dir(image_version=image_version)
        panel_path = self.get_panel_path(image_version=image_version)
        paths = dict(
            images_dir=images_dir,
            masks_dir=masks_dir,
            panel_path=panel_path,
            samples_path=samples_path,
            intensity_path=intensity_path,
            spatial_path=spatial_path,
        )

        # paths = {k: p if p.exists() else None for k,p in paths.items()}

        return paths

    def load(
            self,
            image_version: str = "published",
            mask_version: str = "published",
            features_as_published: bool = True,
    ):
        self.logger.info(
            f"Loading dataset {self.name} with image_version={image_version} and mask_version={mask_version}")

        paths = self.resolve_paths(
            image_version=image_version,
            mask_version=mask_version,
            features_as_published=features_as_published,
        )

        sample_ids = {}

        # load panel
        panel_path = paths["panel_path"]
        panel = pd.read_parquet(panel_path) if panel_path.exists() else None
        if panel is not None:
            assert panel.index.name == 'channel_index'
            assert 'target' in panel.columns
        self.panel = panel

        # load samples metadata
        samples_path = paths["samples_path"]
        samples = pd.read_parquet(samples_path) if samples_path.exists() else None

        if samples is not None and sample_ids:
            samples = samples.loc[list(sample_ids), :]
        elif samples is not None:
            # TODO: what should be the default location to get the sample_ids from?
            #   we want to support loading only certain `sample_ids`
            sample_ids = set(samples.index)

        # load features [intensity]
        intensity_path = paths["intensity_path"]
        intensity = pd.read_parquet(intensity_path) if intensity_path.exists() else None

        if intensity is not None:
            assert set(self.panel.target) == set(intensity.columns)
            assert intensity.index.names == ('sample_id', 'object_id')
            sample_ids = sample_ids & set(intensity.index.get_level_values('sample_id'))

        # load features [spatial]
        spatial_path = paths["spatial_path"]
        spatial = pd.read_parquet(spatial_path) if spatial_path.exists() else None
        if spatial is not None:
            assert spatial.index.names == ('sample_id', 'object_id')

        if intensity is not None and spatial is not None:
            assert set(intensity.index) == set(spatial.index)
            intensity, spatial = intensity.align(spatial, join="outer", axis=0)

        # load masks and images
        masks_dir = paths["masks_dir"]
        images_dir = paths["images_dir"]

        if sample_ids:
            masks = {}
            images = {}

            for sample_id in sample_ids:
                mask_path = masks_dir / f"{sample_id}.tiff"
                image_path = images_dir / f"{sample_id}.tiff"

                if masks_dir.exists():
                    masks[sample_id] = Mask(
                        id=mask_path.stem, data_path=mask_path, metadata_path=None
                    )

                if images_dir.exists():
                    images[sample_id] = Image(
                        id=image_path.stem,
                        data_path=image_path,
                        metadata_path=panel_path,
                    )
        else:

            if masks_dir.exists():
                masks = {
                    p.stem: Mask(id=p.stem, data_path=p, metadata_path=None)
                    for p in masks_dir.glob("*.tiff")
                }
            else:
                masks = {}

            if images_dir.exists():
                images = {
                    p.stem: Image(id=p.stem, data_path=p, metadata_path=None)
                    for p in images_dir.glob("*.tiff")
                }
            else:
                images = {}

        # TODO: either always require sample_ids or this might break
        if images and masks:
            assert set(images) == set(masks)
            assert set(images) == sample_ids
        if images:
            assert set(images) == sample_ids
        if masks:
            assert set(masks) == sample_ids

        self.sample_ids = sorted(sample_ids)
        # retain only the specified sample_ids
        self.samples = samples.loc[list(sample_ids)] if sample_ids else None
        self.images = {k: v for k, v in images.items() if k in sample_ids}
        self.masks = {k: v for k, v in masks.items() if k in sample_ids}
        self.intensity = intensity.loc[list(sample_ids)] if intensity else None
        self.spatial = spatial.loc[list(sample_ids)] if intensity else None

        return {
            "samples": self.samples,
            "images": self.images,
            "masks": self.masks,
            "panel": self.panel,
            "intensity": self.intensity,
            "spatial": self.spatial,
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
