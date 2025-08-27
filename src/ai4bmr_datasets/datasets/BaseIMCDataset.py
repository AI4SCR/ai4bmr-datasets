from pathlib import Path
from loguru import logger
import pandas as pd

from ai4bmr_datasets.datamodels.Image import Image, Mask
from ai4bmr_datasets.utils.tidy import filter_paths
from ai4bmr_datasets.utils import io

class BaseIMCDataset:
    name: str = None
    id: str = None
    doi: str = None

    """
    Base class for managing Imaging Mass Cytometry (IMC) datasets.

    This class provides a standardized interface for loading and processing IMC data,
    including images, masks, features (intensity, spatial), and clinical metadata.
    It handles directory resolution, data versioning, and alignment of different data modalities.

    Attributes:
        name (str): The name of the dataset.
        id (str): A unique identifier for the dataset.
        doi (str): The Digital Object Identifier (DOI) for the dataset's publication.
    """

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
        """
        Initializes the BaseIMCDataset with specified configurations.

        Args:
            base_dir (Path | None): Base directory for the dataset, containing '01_raw' and '02_processed' folders.
                                    If None, it defaults to an environment variable or a cache directory.
            image_version (str | None): Identifier for the image version to load.
            mask_version (str | None): Identifier for the mask version to load.
            feature_version (str | None): Identifier for the feature version to load.
            metadata_version (str | None): Identifier for the metadata version to load.
            load_intensity (bool): If True, intensity features will be loaded.
            load_spatial (bool): If True, spatial features will be loaded.
            load_metadata (bool): If True, observation-level metadata will be loaded.
            align (bool): If True, aligns and intersects all available sample IDs across modalities.
            join (str): Specifies the type of join to perform during alignment (e.g., 'outer').
        """

        self.base_dir = self.resolve_base_dir(base_dir=base_dir)
        self.image_version = image_version
        self.mask_version = mask_version
        self.feature_version = feature_version
        self.metadata_version = metadata_version
        self.load_intensity = load_intensity
        self.load_spatial = load_spatial
        self.load_metadata = load_metadata
        self.align = align
        self.join = join

        self.sample_ids = None
        self.clinical = None
        self.metadata = None
        self.images = None
        self.masks = None
        self.panel = None
        self.intensity = None
        self.spatial = None

    def resolve_base_dir(self, base_dir: Path | None) -> Path:
        """
        Resolves the base directory for the dataset.

        If `base_dir` is None, it attempts to use the 'AI4BMR_DATASETS_DIR' environment variable.
        If the environment variable is not set, it defaults to a cache directory within the user's home directory.

        Args:
            base_dir (Path | None): The initial base directory provided.

        Returns:
            Path: The resolved absolute path to the base directory.
        """
        import os
        if base_dir is None:
            base_dir = os.environ.get("AI4BMR_DATASETS_DIR", None)
            if base_dir is None:
                base_dir = Path.home().resolve() / '.cache' / 'ai4bmr_datasets' / self.name
            else:
                base_dir = Path(base_dir) / self.name
        return base_dir

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the sample's ID, image, mask, panel, intensity,
                  spatial features, and metadata.
        """
        sample_id = self.sample_ids[idx]
        return {
            "sample_id": sample_id,
            "image": self.images[sample_id],
            "mask": self.masks[sample_id],
            "panel": self.panel,
            "intensity": self.intensity.loc[sample_id],
            "spatial": self.spatial.loc[sample_id],
            "metadata": dict(
                sample_level=self.clinical.loc[sample_id],
                observation_level=self.metadata.loc[sample_id],
            ),
        }

    def setup(self, sample_ids: list[str] | None = None):
        """
        Loads data (images, masks, features, metadata) from disk based on the configured versions.

        This method populates the dataset's attributes (`images`, `masks`, `intensity`, `spatial`,
        `metadata`, `clinical`, `panel`) by reading data from the file system.
        If `align` is True (configured in `__init__`), it ensures that all loaded attributes
        share the same set of sample IDs.

        Args:
            sample_ids (list[str] | None): Optional list of specific sample IDs to load. If None,
                                            all available samples will be considered.
        """
        feature_version = self.feature_version or self.get_version_name(image_version=self.image_version,
                                                                   mask_version=self.mask_version)
        metadata_version = self.metadata_version or self.get_version_name(image_version=self.image_version,
                                                                     mask_version=self.mask_version)

        intensity = spatial = metadata = None
        ids_from_metadata = ids_from_intensity = ids_from_spatial = set()

        logger.info(
            f"Loading dataset {self.name} with image_version={self.image_version} and mask_version={self.mask_version}"
        )

        # load panel
        panel_path = self.get_panel_path(image_version=self.image_version)
        panel = pd.read_parquet(panel_path, engine="fastparquet")

        assert panel.index.name == "channel_index"
        assert "target" in panel.columns

        self.panel = panel

        # load clinical metadata
        clinical = pd.read_parquet(self.clinical_metadata_path, engine="fastparquet")
        assert clinical.index.name == 'sample_id'
        ids_from_clinical = set(clinical.index)

        # load features [intensity]
        if self.load_intensity:
            intensity_dir = self.intensity_dir / feature_version
            intensity = pd.concat(
                [
                    pd.read_parquet(i, engine="fastparquet")
                    for i in intensity_dir.glob("*.parquet")
                    if not i.name.startswith(".")
                ]
            )

            assert set(self.panel.target) == set(intensity.columns)
            assert intensity.index.names == ("sample_id", "object_id")
            ids_from_intensity = set(intensity.index.get_level_values("sample_id"))

        # load features [spatial]
        if self.load_spatial:
            spatial_dir = self.spatial_dir / feature_version
            spatial = pd.concat(
                [
                    pd.read_parquet(i, engine="fastparquet")
                    for i in spatial_dir.glob("*.parquet")
                    if not i.name.startswith(".")
                ]
            )

            assert spatial.index.names == ("sample_id", "object_id")
            ids_from_spatial = set(spatial.index.get_level_values("sample_id"))

        # load images
        images_dir = self.get_image_version_dir(image_version=self.image_version)
        images = {}
        for image_path in [p for p in images_dir.glob("*.tiff") if not p.name.startswith('.')]:
            images[image_path.stem] = Image(
                id=image_path.stem,
                data_path=image_path,
                metadata_path=panel_path,
            )
        ids_from_images = set(images.keys())

        # load masks
        masks_dir = self.get_mask_version_dir(mask_version=self.mask_version)
        masks = {}
        for mask_path in [p for p in masks_dir.glob("*.tiff") if not p.name.startswith('.')]:
            masks[mask_path.stem] = Mask(id=mask_path.stem, data_path=mask_path, metadata_path=None)
        ids_from_masks = set(masks.keys())

        # load metadata
        if self.load_metadata:
            metadata_dir = self.metadata_dir / metadata_version
            metadata = pd.concat(
                [
                    pd.read_parquet(i, engine="fastparquet")
                    for i in metadata_dir.glob("*.parquet")
                    if not i.name.startswith(".")
                ]
            )
            ids_from_metadata = set(metadata.index.get_level_values("sample_id"))

        if self.align:
            from functools import reduce
            list_of_sets = [ids_from_images,
                            ids_from_masks,
                            ids_from_metadata,
                            ids_from_intensity,
                            ids_from_spatial, ]

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
                if intensity is not None:
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

    def create_annotated(self, version_name: str = 'annotated', mask_version: str = "published"):
        """
        Creates an annotated version of the dataset by filtering masks, intensity, and metadata
        to include only objects that are present across all three modalities.

        Args:
            version_name (str): The name for the new annotated version (default: 'annotated').
            mask_version (str): The mask version to use for annotation (default: 'published').
        """
        from skimage.io import imread, imsave
        import numpy as np

        logger.info(f'Creating new data version: `{version_name}`')

        metadata_version = 'published'
        metadata = pd.read_parquet(filter_paths(self.metadata_dir / metadata_version), engine='fastparquet')
        intensity = pd.read_parquet(filter_paths(self.intensity_dir / metadata_version), engine='fastparquet')

        # mask_version = 'published_cell'
        masks_dir = self.masks_dir / mask_version
        mask_paths = [p for p in masks_dir.glob("*.tiff") if not p.name.startswith('.')]

        image_version = 'published'
        images_dir = self.images_dir / image_version
        image_paths = [p for p in images_dir.glob("*.tiff") if not p.name.startswith('.')]

        # collect sample ids
        sample_ids = set(metadata.index.get_level_values('sample_id').unique()) \
                     & set(intensity.index.get_level_values('sample_id').unique()) \
                     & set([i.stem for i in mask_paths]) \
                     & set([i.stem for i in image_paths])

        logger.info(f"Found {len(sample_ids)} samples with metadata, intensity, masks and images")

        # ANNOTATED DATA
        save_masks_dir = self.masks_dir / version_name
        save_masks_dir.mkdir(parents=True, exist_ok=True)

        save_intensity_dir = self.intensity_dir / version_name
        save_intensity_dir.mkdir(parents=True, exist_ok=True)

        save_metadata_dir = self.metadata_dir / version_name
        save_metadata_dir.mkdir(parents=True, exist_ok=True)

        for sample_id in sample_ids:
            save_path = save_masks_dir / f"{sample_id}.tiff"

            if save_path.exists():
                logger.info(f"Skipping annotated mask {save_path}. Already exists.")
                continue

            mask_path = masks_dir / f"{sample_id}.tiff"
            if not mask_path.exists():
                logger.warning(f'No corresponding mask found for {sample_id}. Skipping.')
                continue

            logger.info(f"Saving annotated mask {save_path}")

            mask = io.imread(mask_path)
            intensity_ = intensity.xs(sample_id, level='sample_id', drop_level=False)
            metadata_ = metadata.xs(sample_id, level='sample_id', drop_level=False)

            intensity_objs = set(intensity_.index.get_level_values('object_id').astype(int).unique())
            metadata_objs = set(metadata_.index.get_level_values('object_id').astype(int).unique())
            mask_objs = set(np.unique(mask)) - {0}

            objs = metadata_objs.intersection(mask_objs).intersection(intensity_objs)
            missing_objs = mask_objs - objs
            if missing_objs:
                logger.warning(
                    f'{sample_id} has {len(objs)} common objects and {len(missing_objs)} mask objects without metadata')

            objs = np.asarray(sorted(objs), dtype=mask.dtype)
            mask_filtered = np.where(np.isin(mask, objs), mask, 0)
            assert len(np.unique(mask_filtered)) == len(objs) + 1
            io.save_mask(mask=mask_filtered, save_path=save_path)

            idx = pd.IndexSlice[:, objs]

            intensity_ = intensity_.loc[idx, :]
            intensity_.to_parquet(save_intensity_dir / f"{sample_id}.parquet", engine='fastparquet')

            metadata_ = metadata_.loc[idx, :]
            metadata_.to_parquet(save_metadata_dir / f"{sample_id}.parquet", engine='fastparquet')

    @property
    def raw_dir(self) -> Path:
        """
        Returns the path to the raw data directory (01_raw).
        """
        return self.base_dir / "01_raw"

    @property
    def processed_dir(self) -> Path:
        """
        Returns the path to the processed data directory (02_processed).
        """
        return self.base_dir / "02_processed"

    @property
    def images_dir(self) -> Path:
        """
        Returns the path to the images directory within the processed data.
        """
        return self.processed_dir / "images"

    def get_image_version_dir(self, image_version: str) -> Path:
        """
        Returns the path to a specific image version directory.

        Args:
            image_version (str): The identifier for the image version.

        Returns:
            Path: The path to the specified image version directory.
        """
        return self.images_dir / image_version

    @property
    def masks_dir(self) -> Path:
        """
        Returns the path to the masks directory within the processed data.
        """
        return self.processed_dir / "masks"

    def get_mask_version_dir(self, mask_version: str) -> Path:
        """
        Returns the path to a specific mask version directory.

        Args:
            mask_version (str): The identifier for the mask version.

        Returns:
            Path: The path to the specified mask version directory.
        """
        return self.masks_dir / mask_version

    @property
    def metadata_dir(self) -> Path:
        """
        Returns the path to the metadata directory within the processed data.
        """
        return self.processed_dir / "metadata"

    @staticmethod
    def get_version_name(version: str | None = None, image_version: str | None = None, mask_version: str | None = None) -> str:
        """
        Generates a version name based on image and mask versions, or a provided version string.

        Args:
            version (str | None): A direct version string to use if provided.
            image_version (str | None): The image version identifier.
            mask_version (str | None): The mask version identifier.

        Returns:
            str: The generated version name.
        """
        if image_version is not None and mask_version is not None:
            return f"{image_version}-{mask_version}"
        else:
            return version

    @property
    def features_dir(self) -> Path:
        """
        Returns the path to the features directory within the processed data.
        """
        return self.processed_dir / "features"

    @property
    def spatial_dir(self) -> Path:
        """
        Returns the path to the spatial features directory.
        """
        return self.features_dir / "spatial"

    @property
    def intensity_dir(self) -> Path:
        """
        Returns the path to the intensity features directory.
        """
        return self.features_dir / "intensity"

    def get_panel_path(self, image_version: str) -> Path:
        """
        Returns the path to the panel file for a given image version.

        Args:
            image_version (str): The identifier for the image version.

        Returns:
            Path: The path to the panel.parquet file.
        """
        images_dir = self.get_image_version_dir(image_version)
        return images_dir / "panel.parquet"

    @property
    def clinical_metadata_path(self) -> Path:
        """
        Returns the path to the clinical metadata file.
        """
        return self.metadata_dir / "clinical.parquet"

    def compute_features(self, image_version: str, mask_version: str, force: bool = False):
        """
        Computes and saves intensity and spatial features for the dataset.

        Args:
            image_version (str): The image version to use for feature computation.
            mask_version (str): The mask version to use for feature computation.
            force (bool): If True, recomputes features even if they already exist.
        """
        from ai4bmr_datasets.utils.imc.features import intensity_features, spatial_features
        from tifffile import imread

        version_name = self.get_version_name(image_version=image_version, mask_version=mask_version)
        save_intensity_dir = self.intensity_dir / version_name
        save_spatial_dir = self.spatial_dir / version_name

        if (save_intensity_dir.exists() or save_spatial_dir.exists()) and not force:
            logger.error(
                f"Features directory {save_intensity_dir} already exists. Set force=True to overwrite to recompute.")
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

        image_ids = set([i.stem for i in img_dir.glob("*.tiff") if not i.name.startswith('.')])
        mask_ids = set([i.stem for i in mask_dir.glob("*.tiff") if not i.name.startswith('.')])
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

