
def test_ChEMBL():
    from ai4bmr_datasets import ChEMBL as Dataset

    ds = Dataset()
    ds.prepare_data()
    ds.setup()

    item = ds[0]
    assert 'smiles' in item
    assert 'chembl_id' in item

