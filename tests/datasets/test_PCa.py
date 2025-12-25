def test_PCa():
    from ai4bmr_datasets import PCa
    from pathlib import Path

    ds = dataset = self = PCa(
        base_dir=Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa'),
        image_version="filtered", mask_version="annotated",
        load_metadata=True, load_intensity=True, align=True)
    # ds.prepare_data()
    ds.setup()
    ds.images.keys()

    len(ds.sample_ids)
    sample_id = ds.sample_ids[0]
    image = ds.images[sample_id]
    ds.metadata.shape
    ds.intensity.shape
    ds.metadata.columns
    ds.clinical.notna().sum()

    list(filter(lambda x: '240208_IIIBL_X4Y6_55'.lower() in x, ds.sample_ids))
