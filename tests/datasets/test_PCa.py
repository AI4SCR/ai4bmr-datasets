def test_PCa():
    from ai4bmr_datasets import PCa

    ds = dataset = self = PCa()
    ds.prepare_data()
    ds.setup(image_version="filtered", mask_version="annotated",
             load_metadata=True, load_intensity=True, align=True)
