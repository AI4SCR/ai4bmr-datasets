from ai4bmr_datasets.datasets.NSCLCv2 import NSCLCv2
from pathlib import Path

base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/NSCLCv2')
ds = NSCLCv2(base_dir=base_dir)
data = ds.load()
ds.create_images()
sample_id = ds.sample_ids[0]
image = ds.images[sample_id]
assert image.data.shape[0] == len(ds.panel)
