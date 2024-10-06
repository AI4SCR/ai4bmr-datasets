from ai4bmr_core.datasets.Dataset import BaseDataset


def test_additional_fields():
    class D(BaseDataset):
        # required fields
        _id: str = "Hello"
        _name: str = "World"
        _data: str = "None"

        # additional fields
        myfield: str

        def __init__(self, myfield):
            super().__init__(myfield=myfield)

        def load(self):
            return "Hello World"

    d = D(myfield="Hello World")
    assert d.myfield == "Hello World"
    assert d._data == "Hello World"
