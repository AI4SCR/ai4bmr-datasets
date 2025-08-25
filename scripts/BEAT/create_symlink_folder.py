from ai4bmr_datasets.datasets.BEAT import BEAT
from pathlib import Path
import os

# Base dataset directory
base_dir = Path("/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed_v2/datasets/beat")

# Target directory where wsi_slides should be created
target_dir = Path("/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed_v2/datasets")
symlink_dir = target_dir / "wsi_slides"
symlink_dir.mkdir(exist_ok=True)

# Iterate over subfolders in base_dir
for sample_dir in base_dir.iterdir():

    sample_id = sample_dir.name
    if sample_id in BEAT.INVALID_SAMPLE_IDS:
        continue

    if sample_dir.is_dir():
        wsi_path = sample_dir / "wsi.tiff"

        if wsi_path.exists():
            symlink_path = symlink_dir / f"{sample_dir.name}.tiff"
            # Remove existing symlink if it already exists
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            os.symlink(wsi_path, symlink_path)
            print(f"Linked {wsi_path} -> {symlink_path}")



