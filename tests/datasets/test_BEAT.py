def test_Keren2018():
    from ai4bmr_datasets import BEAT
    ds = self = BEAT()
    ds.prepare_data()
    ds.setup()
