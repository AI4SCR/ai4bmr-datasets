def test_PCa():
    from ai4bmr_datasets import PCa

    ds = dataset = self = PCa()
    ds.prepare_data()
    ds.setup(image_version="filtered", mask_version="annotated",
             load_metadata=True, load_intensity=True, align=True)

    len(ds.sample_ids)
    sample_id = ds.sample_ids[0]
    image = ds.images[sample_id]
    ds.metadata.shape
    ds.intensity.shape
