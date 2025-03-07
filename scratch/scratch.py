# from pathlib import Path
# from ai4bmr_datasets.datasets.SegmentedImages import SegmentedImages
# from ai4bmr_datasets.datasets.SegmentedImagesCrops import SegmentedImagesCrops
#
# raw_dir = Path(
#     '/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/ai4scr/cell-embeddings/01_datasets/t1')
# processed_dir = raw_dir
# ds = SegmentedImages(processed_dir=processed_dir, img_version='', mask_version='', in_memory=False)
# s = ds[0]
# ss = next(iter(ds))
# ccv = SegmentedImagesCrops(dataset=ds, padding=0, use_centroid=False).setup()
# import pandas as pd
# #
# # assert ccv[0].source.sample_name == ccv[1].source.sample_name
# # assert ccv[0].source is ccv[1].source
# # assert ccv[0].source.source is not ccv[1].source.source
# #
# # ds = SegmentedImages(base_dir=base_dir, img_version='', mask_version='', in_memory=True).setup()
# # ccv = SegmentedImagesCrops(dataset=ds, padding=0, use_centroid=False).setup()
# #
# # assert ccv[0].source.sample_name == ccv[1].source.sample_name
# # assert ccv[0].source is ccv[1].source
# # assert ccv[0].source.source is ccv[1].source.source
# #
# # # %%
# # from pathlib import Path
# #
# # from ai4bmr_datasets.datasets.IMC import IMCImages
# # from ai4bmr_datasets.datasets.Views import SegmentedImagesCrops
# #
# # base_dir = Path(
# #     '/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/ai4scr/cell-embeddings/01_datasets/t1')
# # ds = IMCImages(base_dir=base_dir, img_version='', mask_version='').setup()
#
# # from ai4bmr_datasets.datasets.PCa import PCa
# # dataset = PCa()
# # # dataset.create_metadata()
# # dataset = dataset.setup()
# # a = dataset._data['intensity']
# # b = dataset._data['clinical_metadata']
#
# # from ai4bmr_datasets.datasets.TNBC import TNBC
# #
# # dataset = TNBC(verbose=1)
# # data = dataset.load()
# #
# # image = data['images'][14]
# # masks = data['masks'][14]
# #
# # # image to look closer at
# # 34, 26, 8, 7
#
# # from ai4bmr_datasets.datasets.Dummy import DummyImages
# # di = DummyImages(num_samples=10)
# # data = di.load()
#
# # %%
# # from ai4bmr_datasets.datasets.TNBC import TNBC
# # tnbc = TNBC(verbose=1)
# # data = tnbc.load()
# # data['panel']
#
#
# # %%
# from pathlib import Path
# from ai4bmr_datasets.datasets import PCa
# base_dir = Path('/Users/adrianomartinelli/data/datasets/PCa')
# dataset = PCa(base_dir=base_dir)
# data = dataset.load()
#
# # %%
# from ai4bmr_datasets.datasets import BLCa
# base_dir = Path('/Users/adrianomartinelli/data/datasets/BLCa')
# dataset = BLCa(base_dir=base_dir)
# data = dataset.load()

# %%
from pathlib import Path
from ai4bmr_datasets.datasets import TNBCv2
ds = TNBCv2(base_dir=Path('/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/datasets/TNBC'))
# ds.process()
x = ds.load()
s = ds[0]

# %% to compute new feature version use
from ai4bmr_imc.measure import intensity_features, spatial_features

version_name = 'published'
x = ds.load(image_version=version_name, mask_version=version_name, features_as_published=False)
for sample in ds:
    sample_id = sample['sample_id']
    intensity_path = ds.get_intensity_dir(image_version=version_name, mask_version=version_name) / f'{sample_id}.parquet'
    intensity = intensity_features(sample['image'].data, sample['mask'].data, sample['panel'])

    spatial = spatial_features(sample['mask'].data)
