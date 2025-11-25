# SpatialProteomicsNet

**SpatialProteomicsNet** is an open-source Python package providing a unified interface to access, prepare, and load
spatial proteomics datasets (e.g., Imaging Mass Cytometry and MIBI) for machine learning workflows.

For a detailed description, motivation, and full list of supported technologies and use cases, please refer
to [paper.md](./paper.md).

---

## 📦 Supported Datasets

The package supports the following public spatial proteomics datasets:

- **Keren et al. 2018** – IMC of triple-negative breast cancer [@Keren2018]
- **Jackson et al. 2020** – IMC of breast cancer [@jacksonSinglecellPathologyLandscape2020]
- **Danenberg et al. 2022** – IMC of breast cancer [@danenbergBreastTumorMicroenvironment2022]
- **Cords et al. 2024** – IMC of NSCLC [@cordsCancerassociatedFibroblastPhenotypes2024]

|  | name          | images | masks | markers | annotated cells | clinical samples |
|-:|:--------------|-------:|------:|--------:|----------------:|-----------------:|
|  | Danenberg2022 |    794 |   794 |      39 |         1123466 |              794 |
|  | Cords2024     |   2070 |  2070 |      43 |         5984454 |             2072 |
|  | Jackson2020   |    735 |   735 |      35 |         1224411 |              735 |
|  | Keren2018     |     41 |    41 |      36 |          201656 |               41 |

<figcaption><strong>Table 1:</strong> Summary statistics of supported spatial proteomics datasets in the package.</figcaption>

---

## 🧱 Data Model

All datasets are represented in a **consistent format**, enabling the same code to work across different
datasets. After preparing and setting up the dataset instance, the following attributes are available:

- `images` – Dictionary of image objects
- `panel` – Imaging panel description (DataFrame), including a key `target`
- `masks` – Dictionary of mask objects
- `metadata` – Observation-level metadata (pandas DataFrame) including a key `label`
- `clinical` – Clinical / Sample-level metadata (pandas DataFrame)
- `intensity` – Object / Single-cell intensity features (DataFrame)
- `spatial` – Object / Single-cell spatial features (DataFrame)

These components are enforced and standardized across datasets.

### Data Versions

A dataset can have different versions of images, masks, features, and metadata, allowing for flexibility in data
access.  
To load the data *as published*, use `{image,mask,metadata,feature}_version="published"`. Be aware that for most
datasets, this means the loaded data is not harmonized. You may encounter inconsistencies such as:

- objects present in the masks that are not listed in the metadata,
- images without matching annotations,
- or cells in the metadata with no corresponding segmentation.

To work with a cleaned and more consistent dataset, some datasets offer an `annotated` version. This version:

- has cleaned masks to include only objects found in both the segmentation and the metadata,
- only contains images that have corresponding metadata.

**Efficient Data Loading:** Images and masks are **lazily loaded** from disk to avoid unnecessary memory use. Tabular
data are stored in Parquet format for fast access.

---

## ⚙️ Installation

### Prerequisites

- Python >= 3.10
- R >= 4.4.1

To install R, please visit the [R project website](https://www.r-project.org/) and follow the installation instructions
for your operating system.
Alternatively, you can install R using conda.

### Environment Setup

We recommend using a virtual environment to install the required dependencies.

#### Using `venv`

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git
```

#### Using `conda`

```bash
# Create a conda environment
conda create -n ai4bmr-datasets python=3.12 -y
conda activate ai4bmr-datasets

# Install the package
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git
#conda install conda-forge::r-base  # optional, if R is not already installed
```

---

## 🚀 Quickstart

This section provides a brief guide to get started with the package.

> **Note on Memory Requirements**
>
> Please be aware that storing the datasets can consume a significant amount of memory. The approximate requirements are
> as follows:
> - **Keren2018:** ~10GB
> - **Jackson2020:** ~180GB
> - **Danenberg2022:** ~66GB
> - **Cords2024:** ~785GB

### 1. Initialize and Prepare a Dataset

First, import the desired dataset and initialize it. You can specify a `base_dir` to control where the data is stored.
If not provided, it defaults to `~/.cache/ai4bmr-datasets` or the path set in the `AI4BMR_DATASETS_DIR` environment
variable.
We recommend configuring the environment variables in a `.env` file for convenience.

```python
from ai4bmr_datasets import Keren2018
from pathlib import Path

# Define the base directory for storing the dataset
base_dir = None  # use defaults
base_dir = Path('path/to/data/')  # define your own path

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
```

### 2. Setup and Access Data

After preparing the data, use the `setup()` method to load the data. By setting `load_intensity` and `load_metadata` to
`True`, you can access cell-level features and clinical data. You can list the available data versions using 
`list_{image, mask, metadata, features}_versions()`.

```python
# List available data versions
dataset.list_image_versions()
dataset.list_mask_versions()
dataset.list_metadata_versions()
dataset.list_features_versions()

# Setup the dataset with a specific version and load features
dataset.setup()

# Access core components of the dataset
print("Sample IDs:", dataset.sample_ids)
print("Available images:", list(dataset.images.keys()))
print("Available masks:", list(dataset.masks.keys()))
```

### 3. Work with Image and Mask Objects

Image and mask objects are loaded lazily. The data is only loaded into memory when you access the `.data` attribute.

```python
import matplotlib.pyplot as plt

# Get the first sample ID
sample_id = dataset.sample_ids[0]

# Access the image and mask data
img = dataset.images[sample_id].data
mask = dataset.masks[sample_id].data

print(f"Image shape for sample {sample_id}: {img.shape}")
print(f"Mask shape for sample {sample_id}: {mask.shape}")

panel = dataset.panel
idx = panel.target == 'ds_dna'

# Visualize the image and mask for the DNA channel
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].imshow(img[idx].squeeze())
axs[1].imshow(mask > 0)
for ax, title in zip(axs, ['channel', 'masks']):
    ax.set_title(title)
    ax.set_axis_off()
ax.figure.tight_layout()
ax.figure.show()

```

### 4. Access Cell-Level Features and Metadata

Once loaded, you can directly access the dataframes for intensity, metadata, and clinical data.

```python
# Access the intensity matrix (cells x markers)
print("Intensity matrix shape:", dataset.intensity.shape)

# Access the metadata (cells x annotations)
print("Metadata shape:", dataset.metadata.shape)

# Access the clinical data (patients x annotations)
print("Clinical data shape:", dataset.clinical.shape)
```

---

## 🤝 Contributing

We welcome contributions to improve the project, including:

- Adding new datasets
- Fixing bugs
- Improving performance or usability

### 📌 Contribution Guidelines

If you'd like to contribute:

1. **Fork the repository** and create a feature branch.
2. Add or update relevant tests if applicable.
3. Submit a **pull request** with a clear description of your changes.

Please open a discussion or issue before implementing large features to ensure alignment with project goals.

### 🧬 Adding a New Dataset

The community is encouraged to contribute additional spatial proteomics datasets. When preparing a pull request:

1. **Check the expected data assets**
   - **Required**: multiplexed images in OME-TIFF, TIFF, or another lossless format.
   - **Strongly recommended**: matching segmentation masks and any preprocessed (filtered or transformed) intensity tables published with the dataset.
   - Optional but helpful: spatial features, clinical metadata, or annotation tables that appeared in the associated publication.

2. **Stage the data for download**
   - Host the files at a stable location such as [Figshare](https://figshare.com/) or [Zenodo](https://zenodo.org/); both services provide persistent DOIs and long-term access.
   - Verify that public download links work without authentication and document any required access steps in your dataset class docstring.

3. **Create the dataset implementation**
   - Add a new module under `src/ai4bmr_datasets/datasets/<YourDataset>.py` that subclasses [`BaseIMCDataset`](./src/ai4bmr_datasets/datasets/BaseIMCDataset.py).
   - Implement a `prepare_data()` pipeline that downloads the raw assets into `01_raw/`, processes them into the standardized structure inside `02_processed/`, and saves panel, metadata, images, masks, and features using the helper utilities in `ai4bmr_datasets.utils`.
   - Register the dataset in `src/ai4bmr_datasets/datasets/__init__.py` and expose it from `src/ai4bmr_datasets/__init__.py` so it can be imported via `from ai4bmr_datasets import YourDataset`.

4. **Document dataset-specific details**
   - Include module-level docstrings describing the source publication, licensing, and any preprocessing decisions.
   - Update this README or the documentation to list the new dataset with a short summary of its contents.

sts**
   - Mirror the style of the existing tests in `tests/datasets/` to cover download stubs, preprocessing helpers, and the `prepare_data()` + `setup()` flow.
   - Run `pytest tests/datasets -k YourDataset` locally (or simply `pytest`) to confirm that the end-to-end processing succeeds using the provided sample files or fixtures.

Pull requests that add datasets should include the DOI or URL for the hosted files, describe the available modalities (images, masks, intensities), and enumerate any deviations from the shared data model.

### 🐛 Reporting Issues

If you encounter bugs, incorrect behavior, or missing functionality:

- [Open an issue](https://github.com/AI4SCR/ai4bmr-datasets/issues) with a clear title and minimal reproducible example.
- Include relevant error messages and version information if possible.

We monitor issues regularly and appreciate detailed, constructive reports.

### 💬 Seeking Support

If you need help using the software:

- Check the existing [issues](https://github.com/AI4SCR/ai4bmr-datasets/issues)
  and [discussions](https://github.com/AI4SCR/ai4bmr-datasets/discussions) first.
- For general questions or usage advice, feel free to open
  a [discussion topic](https://github.com/AI4SCR/ai4bmr-datasets/discussions/new).
- For dataset-specific problems, include dataset identifiers and loading code if applicable.

We aim to foster a welcoming and respectful community. Please be kind and collaborative in your interactions.

---

## 📬 Contact

For questions, issues, or contributions, feel free to contact:

**Adriano Martinelli**  
📧 adriano.martinelli@chuv.ch  
📧 adrianom@student.ethz.ch