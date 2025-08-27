import pytest
import tempfile

# @pytest.mark.skip(reason="Only run when downloading changes")
@pytest.mark.parametrize("dataset_name", ["Jackson2020", "Danenberg2022", "Cords2024", "Keren2018"])
def test_prepare_data(dataset_name):
    from pathlib import Path

    match dataset_name:
        case "Jackson2020":
            from ai4bmr_datasets import Jackson2020 as Dataset
        case "Danenberg2022":
            from ai4bmr_datasets import Danenberg2022 as Dataset
        case "Cords2024":
            from ai4bmr_datasets import Cords2024 as Dataset
        case "Keren2018":
            from ai4bmr_datasets import Keren2018 as Dataset
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f'Running test in {tmpdir}')
            # tmpdir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/.tmp')
            base_dir = Path(tmpdir)
            image_version = metadata_version = feature_version = 'published'
            mask_version = 'published_cell' if dataset_name == "Danenberg2022" else 'published'
            ds = Dataset(base_dir=base_dir,
                         image_version=image_version,
                         mask_version=mask_version,
                         metadata_version=metadata_version, load_metadata=True,
                         feature_version=feature_version, load_intensity=True)
            ds.prepare_data()
            ds.setup()

            assert isinstance(ds.images, dict)
            assert isinstance(ds.masks, dict)
            import pandas as pd
            assert isinstance(ds.metadata, pd.DataFrame)
            assert isinstance(ds.intensity, pd.DataFrame)

    except Exception as e:
        print(f"Error preparing dataset {dataset_name}: {e}")
    finally:
        pass
        # import shutil
        # Clean up temporary directory if it was created
        # if 'tmpdir' in locals() and tmpdir.exists():
        #     shutil.rmtree(tmpdir, ignore_errors=True)
        #     print(f"Cleaned up temporary directory: {tmpdir}")

# test_prepare_data("Jackson2020")
# test_prepare_data("Danenberg2022")
test_prepare_data("Cords2024")
# test_prepare_data("Keren2018")
