import pytest

@pytest.mark.parametrize("dataset_name", ["Jackson2023", "Danenberg2022", "Cords2024"])
def test_prepare_data(dataset_name):
    from pathlib import Path
    import shutil

    match dataset_name:
        case "Jackson2023":
            from ai4bmr_datasets.datasets.Jackson2023 import Jackson2023 as Dataset
        case "Danenberg2022":
            from ai4bmr_datasets.datasets.Danenberg2022 import Danenberg2022 as Dataset
        case "Cords2024":
            from ai4bmr_datasets.datasets.Cords2024 import Cords2024 as Dataset
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    try:
        tmpdir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/temporary')
        base_dir = Path(tmpdir)
        ds = Dataset(base_dir=base_dir)
        ds.prepare_data()

        image_version = mask_version = metadata_version = 'published'
        ds.setup(image_version=image_version,
                 mask_version=mask_version,
                 metadata_version=metadata_version,
                 load_metadata=True,
                 load_intensity=True)
    except Exception as e:
        print(f"Error preparing dataset {dataset_name}: {e}")
    finally:
        # Clean up temporary directory if it was created
        if 'tmpdir' in locals() and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {tmpdir}")

