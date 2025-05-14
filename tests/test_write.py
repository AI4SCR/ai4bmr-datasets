from pathlib import Path
import os

username = os.getenv("USER")
test_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/BLCa/02_processed/features/intensity')
with open(test_dir / f'{username}.txt', 'w') as f:
    f.write('test')

subdir = test_dir / f'{username}_dir'
subdir.mkdir(parents=True, exist_ok=True)
