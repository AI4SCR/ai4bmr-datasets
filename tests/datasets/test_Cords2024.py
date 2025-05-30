
def test_Cords2024():
    from ai4bmr_datasets import Cords2024
    from pathlib import Path
    image_version = mask_version = metadata_version = 'published'
    base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/Cords2024')
    ds = self = Cords2024(base_dir=base_dir)
    # ds.prepare_data()
    ds.setup(image_version=image_version, mask_version=mask_version,
             metadata_version=metadata_version, load_metadata=True, load_intensity=False)

