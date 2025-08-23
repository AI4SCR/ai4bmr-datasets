def test_BEAT():
    from ai4bmr_datasets import BEAT
    ds = self = BEAT()
    ds.prepare_tools()
    ds.prepare_clinical()
    ds.prepare_wsi(force=True)
    # ds.setup()
