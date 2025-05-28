
def test_BLCa():
    from ai4bmr_datasets.datasets.BLCa import BLCa
    from pathlib import Path

    base_dir = Path("/users/amarti51/prometex/data/datasets/BLCa")
    ds = BLCa(base_dir=base_dir)
    # ds.prepare_data()

    ds.setup(image_version='filtered',
             mask_version='annotated',
             load_intensity=True, load_metadata=True, align=True)

    ds.setup(image_version='filtered',
             mask_version='clustered',
             load_intensity=True, load_metadata=True)

