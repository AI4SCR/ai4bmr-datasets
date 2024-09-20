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

#
# assert ccv[0].source.sample_name == ccv[1].source.sample_name
# assert ccv[0].source is ccv[1].source
# assert ccv[0].source.source is not ccv[1].source.source
#
# ds = SegmentedImages(base_dir=base_dir, img_version='', mask_version='', in_memory=True).setup()
# ccv = SegmentedImagesCrops(dataset=ds, padding=0, use_centroid=False).setup()
#
# assert ccv[0].source.sample_name == ccv[1].source.sample_name
# assert ccv[0].source is ccv[1].source
# assert ccv[0].source.source is ccv[1].source.source
#
# # %%
# from pathlib import Path
#
# from ai4bmr_datasets.datasets.IMC import IMCImages
# from ai4bmr_datasets.datasets.Views import SegmentedImagesCrops
#
# base_dir = Path(
#     '/Users/adrianomartinelli/Library/CloudStorage/OneDrive-ETHZurich/oneDrive-documents/data/ai4scr/cell-embeddings/01_datasets/t1')
# ds = IMCImages(base_dir=base_dir, img_version='', mask_version='').setup()

from ai4bmr_datasets.datasets.BLCa import BLCa
# from ai4bmr_datasets.datasets.PCa import PCa
# dataset = PCa()
# # dataset.create_metadata()
# dataset = dataset.setup()
# a = dataset._data['intensity']
# b = dataset._data['clinical_metadata']

from ai4bmr_datasets.datasets.TNBC import TNBC
dataset = TNBC()
dataset.load()
