def test_Danenberg2022():
    from pathlib import Path
    from ai4bmr_datasets.datasets.Danenberg2022 import Danenberg2022
    base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Danenberg2022')
    ds = self = Danenberg2022(base_dir=base_dir)
    ds.prepare_data()
    ds.setup(image_version='published', mask_version='published_cell',
             metadata_version='published', feature_version='published',
             load_metadata=True, load_intensity=True)
