def test_PCa():
    from ai4bmr_datasets import PCa
    import pandas as pd

    ds = dataset = self = PCa()
    ds.process_acquisitions()
    ds.label_transfer()
    ds.setup(image_version="filtered", mask_version="filtered", load_metadata=True)
    # ds.metadata.shape
