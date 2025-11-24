from dotenv import load_dotenv
assert load_dotenv()

import hashlib
from pathlib import Path
import os

dataset_dirs = [i for i in Path(os.getenv("AI4BMR_DATASETS_DIR")).iterdir() if i.is_dir()]

records = []
for dir_ in dataset_dirs:
    zip_files = list(dir_.rglob("*.zip"))
    for zip_file in zip_files:
        # Compute SHA‑256 directly from the ZIP file
        sha = hashlib.sha256()
        with open(zip_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        checksum = sha.hexdigest()
        records.append({
            'dataset': dir_.name,
            'file_path': str(zip_file),
            'file_name': str(zip_file.name),
            'checksum': checksum
        })