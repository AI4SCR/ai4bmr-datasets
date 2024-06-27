from ai4bmr_datasets.datasets.IMC import IMC
from pathlib import Path

base_dir = Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/ai4scr/cell-embeddings/01_datasets/t1')
ds = IMC(base_dir=base_dir, img_version='', mask_version='').setup()
s = ds[0]

from ai4bmr_datasets.datasets.CropCollectionView import CropCollectionView

ccv = CropCollectionView(dataset=ds, padding=0, use_centroid=False).setup()
ccv = CropCollectionView(dataset=ds, padding=10, use_centroid=True).setup()

