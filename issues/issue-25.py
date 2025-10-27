from ai4bmr_datasets import Keren2018
from pathlib import Path

# Define the base directory for storing the dataset
base_dir = None  # use defaults
base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/issues/ai4bmr-datasets/25/Keren2018')
base_dir.mkdir(parents=True, exist_ok=True)

# Initialize the dataset
dataset = Keren2018(base_dir=base_dir,
                  image_version="published",
                  mask_version="published",
                  feature_version="published",
                  load_intensity=True,
                  metadata_version="published",
                  load_metadata=True
                  )

# Download and preprocess the data. This only needs to be run once.
dataset.prepare_data()