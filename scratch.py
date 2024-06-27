from pathlib import Path

from ai4bmr_datasets.datasets.IMC import IMC

base_dir = Path(
    '/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/ai4scr/cell-embeddings/01_datasets/t1')
ds = IMC(base_dir=base_dir, img_version='', mask_version='').setup()
s = ds[0]

from ai4bmr_datasets.datasets.CropCollectionView import CropCollectionView

ds = IMC(base_dir=base_dir, img_version='', mask_version='', in_memory=False).setup()
ccv = CropCollectionView(dataset=ds, padding=0, use_centroid=False).setup()

assert ccv[0].img.sample_name == ccv[1].img.sample_name
assert ccv[0].img is ccv[1].img
assert ccv[0].img.img is not ccv[1].img.img

ds = IMC(base_dir=base_dir, img_version='', mask_version='', in_memory=True).setup()
ccv = CropCollectionView(dataset=ds, padding=0, use_centroid=False).setup()

assert ccv[0].img.sample_name == ccv[1].img.sample_name
assert ccv[0].img is ccv[1].img
assert ccv[0].img.img is ccv[1].img.img
