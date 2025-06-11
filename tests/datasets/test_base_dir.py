def test_base_dir_resolve_default():
    from pathlib import Path
    from ai4bmr_datasets import Cords2024

    ds = Cords2024(base_dir=None)
    default_dir = Path.home() / '.cache' / 'ai4bmr_datasets' / Cords2024.name
    assert ds.base_dir.resolve() == default_dir.resolve(), f"Expected base_dir to be {default_dir.resolve()}, but got {ds.base_dir.resolve()}"

def test_base_dir_resolve_from_env_var():
    from pathlib import Path
    from ai4bmr_datasets import Cords2024
    import os

    ai4bmr_datasets_dir = Path('/tmp/ai4bmr_datasets')
    os.environ['AI4BMR_DATASETS_DIR'] = str(ai4bmr_datasets_dir)

    ds = Cords2024(base_dir=None)
    assert ds.base_dir == ai4bmr_datasets_dir / Cords2024.name, f"Expected base_dir to be {ai4bmr_datasets_dir}, but got {ds.base_dir}"
