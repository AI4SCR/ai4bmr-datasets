import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path


class DummyPoints:

    def __init__(self, save_dir: Path | None = None, height: int = 224, width: int = 224, num_samples: int = 100,
                 num_points: int = 1000, num_features: int = 10, num_point_classes: int = 5, num_classes: int = 2):

        self.save_dir = save_dir or Path('~/data/ai4bmr-learn/DummyPoints').expanduser()
        self.data_dir = self.save_dir / 'data'

        self.height = height
        self.width = width
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_point_classes = num_point_classes
        self.points_class_labels = [f'type_{i}' for i in range(self.num_point_classes)]

        self.num_features = num_features
        self.num_classes = num_classes
        self.class_labels = [f'type_{i}' for i in range(self.num_classes)]

    def prepare_data(self, force: bool = False):
        save_metadata_path = self.save_dir / 'metadata.parquet'
        save_data_dir = self.save_dir / 'data'
        # save_metadata_dir = self.save_dir / 'metadata'

        # if save_metadata_path.exists() and save_data_dir.exists() and save_metadata_dir.exists() and not force:
        if save_metadata_path.exists() and save_data_dir.exists() and not force:
            return
        else:
            self.data_dir.mkdir(exist_ok=True, parents=True)

            rng = np.random.default_rng(seed=42)

            metadata = dict(label=rng.choice(self.class_labels, self.num_samples))
            metadata = pd.DataFrame(metadata, dtype='category')
            metadata = metadata.convert_dtypes()
            metadata.index = metadata.index.astype(str)
            metadata.index.name = 'sample_id'
            metadata.to_parquet(save_metadata_path, engine='fastparquet')

            sample_ids = metadata.index.to_list()

            for sample_id in sample_ids:
                df = pd.DataFrame(rng.random((self.num_points, self.num_features)))
                df['label'] = rng.choice(self.points_class_labels, self.num_points)
                df['label'] = df['label'].astype('category')

                xy = rng.random((self.num_points, 2))

                df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(xy[:, 0] * self.width, xy[:, 1] * self.height),
                                      crs='EPSG:4326')
                df.columns = df.columns.astype(str)

                index = pd.Index(range(self.num_points), dtype=str, name='object_id')
                df.index = index

                df.to_file(str(self.data_dir / f'{sample_id}.gpkg'))

    def setup(self):
        self.metadata = pd.read_parquet(self.save_dir / 'metadata.parquet')
        self.sample_ids = self.metadata.index.to_list()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        points = gpd.read_file(self.data_dir / f'{sample_id}.gpkg')

        return {
            'sample_id': sample_id,
            'points': points,
            'metadata': self.metadata.loc[sample_id].to_dict(),
        }

# points = DummyPoints()
# points.prepare_data()
# points.setup()
# points[0]
