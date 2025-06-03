from pathlib import Path
from loguru import logger
import pandas as pd

from ai4bmr_datasets.datamodels.Image import Image, Mask

class BaseIMCDataset:
    """
    Base class for managing Imaging Mass Cytometry (IMC) datasets,
    including loading images, masks, features (intensity, spatial), and metadata.
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the dataset with a base directory.

        Args:
            base_dir (Path): Base directory for the dataset that contains 01_raw and 02_processed folders.
        """

        self.base_dir = self.resolve_base_dir(base_dir=base_dir)

        self.sample_ids = None
        self.clinical_metadata = None
        self.metadata = None
        self.images = None
        self.masks = None
        self.panel = None
        self.intensity = None
        self.spatial = None

    def resolve_base_dir(self, base_dir: Path | None):
        import os
        if base_dir is None:
            base_dir = os.environ.get("AI4BMR_DATASETS_BASE_DIR", None)
            if base_dir is None:
                base_dir = Path.home().resolve() / '.cache' / 'ai4bmr_datasets' / self.name
            else:
                base_dir = Path(base_dir)
        return base_dir

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return {
            "sample_id": sample_id,
            "image": self.images[sample_id],
            "mask": self.masks[sample_id],
            "panel": self.panel,
            "intensity": self.intensity.loc[sample_id],
            "spatial": self.spatial.loc[sample_id],
            "metadata": dict(
                sample_level=self.clinical_metadata.loc[sample_id],
                observation_level=self.metadata.loc[sample_id],
            ),
        }

    def setup(
            self,
            image_version: str | None = None,
            mask_version: str | None = None,
            feature_version: str | None = None,
            metadata_version: str | None = None,
            load_intensity: bool = False,
            load_spatial: bool = False,
            load_metadata: bool = False,
            sample_ids: list[str] | None = None,
            align: bool = False,
            join: str = "outer",
    ):
        """
        Load and align data (images, masks, features, metadata) from disk.
        If only image_version and mask_version are provided features and metadata will be loaded from the combined
        version {image_version-mask_version}. To load specific versions of features and metadata please use the
        corresponding keywords. This is useful when you want to load process features as published by a publication.

        Align ensures that all loaded attributes share the same samples. This is useful because often not all samples
        are present in images, masks or provided metadata.

        Args:
            image_version (str | None): Image version identifier.
            mask_version (str | None): Mask version identifier.
            feature_version (str | None): Feature version identifier.
            metadata_version (str | None): Metadata version identifier.
            load_intensity (bool): Whether to load intensity features.
            load_spatial (bool): Whether to load spatial features.
            load_metadata (bool): Whether to load observation-level metadata.
            align (bool): Whether to align and intersect all available sample IDs.
        """


        feature_version = feature_version or self.get_version_name(image_version=image_version, mask_version=mask_version)
        metadata_version = metadata_version or self.get_version_name(image_version=image_version, mask_version=mask_version)

        intensity = spatial = metadata = None
        ids_from_metadata = ids_from_intensity = ids_from_spatial = set()

        logger.info(
            f"Loading dataset {self.name} with image_version={image_version} and mask_version={mask_version}"
        )

        # load panel
        panel_path = self.get_panel_path(image_version=image_version)
        panel = pd.read_parquet(panel_path, engine="fastparquet")

        assert panel.index.name == "channel_index"
        assert "target" in panel.columns

        self.panel = panel

        # load clinical metadata
        clinical = pd.read_parquet(self.clinical_metadata_path, engine="fastparquet")
        assert clinical.index.name == 'sample_id'
        ids_from_clinical = set(clinical.index)

        # load features [intensity]
        if load_intensity:
            intensity_dir = self.intensity_dir / feature_version
            intensity = pd.concat(
                [
                    pd.read_parquet(i, engine="fastparquet")
                    for i in intensity_dir.glob("*.parquet")
                ]
            )

            assert set(self.panel.target) == set(intensity.columns)
            assert intensity.index.names == ("sample_id", "object_id")
            ids_from_intensity = set(intensity.index.get_level_values("sample_id"))

        # load features [spatial]
        if load_spatial:
            spatial_dir = self.spatial_dir / feature_version
            spatial = pd.concat(
                [
                    pd.read_parquet(i, engine="fastparquet")
                    for i in spatial_dir.glob("*.parquet")
                ]
            )

            assert spatial.index.names == ("sample_id", "object_id")
            ids_from_spatial = set(spatial.index.get_level_values("sample_id"))

        # load images
        images_dir = self.get_image_version_dir(image_version=image_version)
        images = {}
        for image_path in images_dir.glob("*.tiff"):
            images[image_path.stem] = Image(
                id=image_path.stem,
                data_path=image_path,
                metadata_path=panel_path,
            )
        ids_from_images = set(images.keys())

        # load masks
        masks_dir = self.get_mask_version_dir(mask_version=mask_version)
        masks = {}
        for mask_path in masks_dir.glob("*.tiff"):
            masks[mask_path.stem] = Mask(id=mask_path.stem, data_path=mask_path, metadata_path=None)
        ids_from_masks = set(masks.keys())

        # load metadata
        if load_metadata:
            metadata_dir = self.metadata_dir / metadata_version
            metadata = pd.concat(
                [
                    pd.read_parquet(i, engine="fastparquet")
                    for i in metadata_dir.glob("*.parquet")
                ]
            )
            ids_from_metadata = set(metadata.index.get_level_values("sample_id"))

        if align:
            from functools import reduce
            list_of_sets = [ids_from_images,
                            ids_from_masks,
                            ids_from_metadata,
                            ids_from_intensity,
                            ids_from_spatial,]

            sample_ids = reduce(lambda x, y: x.intersection(y) if y else x, list_of_sets, ids_from_clinical)
            sample_ids = list(sample_ids)

            self.sample_ids = sample_ids
            self.clinical = clinical.loc[sample_ids]
            self.images = {k: v for k, v in images.items() if k in sample_ids}
            self.masks = {k: v for k, v in masks.items() if k in sample_ids}

            if intensity is not None and spatial is not None:
                assert set(intensity.index) == set(spatial.index)
                intensity, spatial = intensity.align(spatial, join="outer", axis=0)

            if metadata is not None and intensity is not None:
                assert set(metadata.index) <= set(intensity.index)
                assert set(intensity.index) <= set(metadata.index)
                assert set(metadata.index) == set(intensity.index)
                intensity, metadata = intensity.align(metadata, join="outer", axis=0)

            if metadata is not None and spatial is not None:
                assert set(metadata.index) == set(intensity.index)
                spatial, metadata = spatial.align(metadata, join="outer", axis=0)

            self.intensity = intensity.loc[sample_ids] if intensity is not None else None
            self.spatial = spatial.loc[sample_ids] if spatial is not None else None
            self.metadata = metadata.loc[sample_ids] if metadata is not None else None

        else:
            self.sample_ids = clinical.index.to_list()
            self.clinical = clinical
            self.images = images
            self.masks = masks
            self.intensity = intensity
            self.spatial = spatial
            self.metadata = metadata

    def process(self):
        raise NotImplementedError("process method must be implemented")

    def create_images(self):
        pass

    def create_masks(self):
        pass

    def create_clinical_metadata(self):
        pass

    def create_metadata(self):
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

    def create_annotated(self):
        metadata_version = 'published'
        metadata = pd.read_parquet(self.metadata_dir / metadata_version, engine='fastparquet')
        intensity = pd.read_parquet(self.intensity_dir / metadata_version, engine='fastparquet')

        mask_version = 'published_nucleus'
        masks_dir = self.masks_dir / mask_version

        # ANNOTATED DATA
        version = 'annotated'
        save_masks_dir = self.masks_dir / version
        save_masks_dir.mkdir(parents=True, exist_ok=True)

        save_intensity_dir = self.intensity_dir / version
        save_intensity_dir.mkdir(parents=True, exist_ok=True)

        save_metadata_dir = self.metadata_dir / version
        save_metadata_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, sample_data in metadata.groupby(level='sample_id'):
            save_path = save_masks_dir / f"{sample_id}.tiff"

            if save_path.exists():
                logger.info(f"Skipping annotated mask {save_path}. Already exists.")
                continue

            mask_path = masks_dir / f"{sample_id}.tiff"
            if not mask_path.exists():
                logger.warning(f'No corresponding mask found for {sample_id}. Skipping.')
                continue

            logger.info(f"Saving annotated mask {save_path}")

            mask = imread(mask_path)
            objs = set(sample_data.index.get_level_values('object_id').astype(int).unique())
            mask_objs = set([int(i) for i in mask.flatten()]) - {0}
            intx = objs.intersection(mask_objs)
            missing_objs = objs.union(mask_objs) - intx
            if missing_objs:
                logger.warning(f'{sample_id} has {len(missing_objs)} missing objects')

            mask_filtered = np.where(np.isin(mask, objs), mask, 0)
            imwrite(save_path, mask_filtered)

            anno_data = sample_data.xs(list(intx), level='object_id')
            anno_data.to_parquet(save_metadata_dir / f"{sample_id}.parquet", engine='fastparquet')

            intensity_data = intensity.xs(sample_id, level='sample_id').xs(list(intx), level='object_id')


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

    def get_image_version_dir(self, image_version: str):
        return self.images_dir / image_version

    @property
    def masks_dir(self):
        return self.processed_dir / "masks"

    def get_mask_version_dir(self, mask_version: str):
        return self.masks_dir / mask_version

    @property
    def metadata_dir(self):
        return self.processed_dir / "metadata"

    @staticmethod
    def get_version_name(version: str | None = None, image_version: str | None = None, mask_version: str | None = None):
        if image_version is not None and mask_version is not None:
            return f"{image_version}-{mask_version}"
        else:
            return version

    @property
    def features_dir(self):
        return self.processed_dir / "features"

    @property
    def spatial_dir(self):
        return self.features_dir / "spatial"

    @property
    def intensity_dir(self):
        return self.features_dir / "intensity"

    def get_panel_path(self, image_version: str):
        images_dir = self.get_image_version_dir(image_version)
        return images_dir / "panel.parquet"

    @property
    def clinical_metadata_path(self):
        return self.metadata_dir / "clinical.parquet"

    def compute_features(self, image_version: str, mask_version: str, force: bool = False):
        from ai4bmr_datasets.utils.imc.features import intensity_features, spatial_features
        from tifffile import imread

        version_name = self.get_version_name(image_version=image_version, mask_version=mask_version)
        save_intensity_dir = self.intensity_dir / version_name
        save_spatial_dir = self.spatial_dir / version_name

        if (save_intensity_dir.exists() or save_spatial_dir.exists()) and not force:
            logger.error(f"Features directory {save_intensity_dir} already exists. Set force=True to overwrite to recompute.")
            return

        save_intensity_dir.mkdir(exist_ok=True, parents=True)
        save_spatial_dir.mkdir(exist_ok=True, parents=True)

        # PATHS
        img_dir = self.get_image_version_dir(image_version)
        assert img_dir.exists()

        mask_dir = self.get_mask_version_dir(mask_version)
        assert mask_dir.exists()

        panel_path = self.get_panel_path(image_version=image_version)
        assert panel_path.exists()

        panel = pd.read_parquet(panel_path, engine="fastparquet")
        panel = panel.reset_index().set_index("target")

        image_ids = set([i.stem for i in img_dir.glob("*.tiff")])
        mask_ids = set([i.stem for i in mask_dir.glob("*.tiff")])
        sample_ids = image_ids.intersection(mask_ids)

        for i, sample_id in enumerate(sample_ids):
            logger.info(f"[{i + 1}/{len(sample_ids)}] Processing {sample_id}")

            img_path = img_dir / f"{sample_id}.tiff"
            assert img_path.exists()

            mask_path = mask_dir / f"{sample_id}.tiff"
            assert mask_path.exists()

            img = imread(img_path)
            mask = imread(mask_path)
            assert img.shape[1:] == mask.shape

            intensity = intensity_features(img=img, mask=mask, panel=panel)
            intensity = intensity.assign(sample_id=sample_id).set_index("sample_id", append=True)
            intensity = intensity.reorder_levels(["sample_id", "object_id"], axis=0)

            spatial = spatial_features(mask=mask)
            spatial = spatial.assign(sample_id=sample_id).set_index("sample_id", append=True)
            spatial = spatial.reorder_levels(["sample_id", "object_id"], axis=0)

            assert intensity.index.equals(spatial.index)

            intensity.to_parquet(save_intensity_dir / f'{sample_id}.parquet', engine="fastparquet")
            spatial.to_parquet(save_spatial_dir / f'{sample_id}.parquet', engine="fastparquet")
