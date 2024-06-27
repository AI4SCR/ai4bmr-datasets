from ai4bmr_datasets.datasets.Dummy import DummyTabular
from ai4bmr_datasets.datasets.PCA import PCA
from pathlib import Path

base_dir = Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/remotes/unibe/data/pca')
base_dir = Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/ai4scr/cell-embeddings/01_datasets/t1')
ds = PCA(base_dir=base_dir, img_version='', mask_version='').setup()
s = ds[0]
