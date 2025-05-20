# echo "umask 002" >> ~/.bashrc
# source ~/.bashrc
# find /work/FAC/FBM/DBC/mrapsoma/prometex/data ss-exec chgrp prometex_101454-pr-g "{}" +
# find /work/FAC/FBM/DBC/mrapsoma/prometex/data -type f -exec chmod ug+rw {} +
# find /work/FAC/FBM/DBC/mrapsoma/prometex/data -type d -exec chmod ug+rws {} +

from pathlib import Path
import os
# os.umask(0o002)
username = os.getenv("USER")
test_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/BLCa/02_processed/features/intensity')
with open(test_dir / f'{username}.txt', 'w') as f:
    f.write('test')

subdir = test_dir / f'{username}_dir'
subdir.mkdir(parents=True, exist_ok=True)
