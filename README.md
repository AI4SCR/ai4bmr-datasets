# SpatialProteomicsNet

**SpatialProteomicsNet** is an open-source Python package providing a unified interface to access, prepare, and load spatial proteomics datasets (e.g., Imaging Mass Cytometry and MIBI) for machine learning workflows.

For a detailed description, motivation, and full list of supported technologies and use cases, please refer to [paper.md](./paper.md).

---

## 📦 Supported Datasets

The package supports the following public spatial proteomics datasets:

- **Keren et al. 2018** – IMC of triple-negative breast cancer [@Keren2018]
- **Jackson et al. 2020** – IMC of breast cancer [@jacksonSinglecellPathologyLandscape2020]
- **Danenberg et al. 2022** – IMC of breast cancer [@danenbergBreastTumorMicroenvironment2022]
- **Cords et al. 2024** – IMC of NSCLC [@cordsCancerassociatedFibroblastPhenotypes2024]

|  | name          |  images |  masks |  markers | annotated cells | clinical samples |
|-:|:--------------|--------:|-------:|---------:|----------------:|-----------------:|
|  | Danenberg2022 |     794 |    794 |       39 |         1123466 |              794 |
|  | Cords2024     |    2070 |   2070 |       43 |         5984454 |             2072 |
|  | Jackson2020   |     735 |    735 |       35 |         1224411 |              735 |
|  | Keren2018     |      41 |     41 |       36 |          201656 |               41 |

<figcaption><strong>Table 1:</strong> Summary statistics of supported spatial proteomics datasets in the package.</figcaption>

Dummy datasets are also included for development and testing purposes.

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

A dataset can have different versions of images, masks, features, and metadata, allowing for flexibility in data access.  
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

To install R, please visit the [R project website](https://www.r-project.org/) and follow the installation instructions for your operating system.

### Environment Setup

We recommend using a virtual environment to install the required dependencies.

#### Using `venv`

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git@v2.0.1
```

#### Using `conda`

```bash
# Create a conda environment
conda create -n ai4bmr-datasets python=3.12 -y
conda activate ai4bmr-datasets

# Install the package
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git@v2.0.1
```

---

## 🚀 Quickstart

### 1. Load and prepare a dataset

All datasets are initialized by passing the desired versions and data loading flags to the constructor.

```python
from ai4bmr_datasets import Keren2018
from pathlib import Path

base_dir = Path("/path/to/storage")  # can be None, resolves to ~/.cache/ai4bmr_datasets by default
dataset = Keren2018(
    base_dir=base_dir,
    image_version="published",
    mask_version="published",
    # optionally, load intensity features and metadata
    feature_version="published", 
    load_intensity=True,
    metadata_version="published", 
    load_metadata=True
)
dataset.prepare_data()  # Downloads and preprocesses data if needed, only needs to be run once
dataset.setup()  # load the data
```

### 2. Access core components

```python
print(dataset.sample_ids)  # List of sample IDs
print(dataset.images.keys())  # Dictionary of images
print(dataset.masks.keys())  # Dictionary of masks
```

### 3. Work with image and mask objects

```python
sample_id = dataset.sample_ids[0]  # get the first sample ID
img = dataset.images[sample_id].data
print("Image shape:", img.shape)

mask = dataset.masks[sample_id].data
print("Mask shape:", mask.shape)
```

### 4. Access optional cell-level features and metadata

```python
print(dataset.intensity.shape)  # Cell x marker matrix
print(dataset.metadata.shape)   # Cell x annotation matrix
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

### 🐛 Reporting Issues

If you encounter bugs, incorrect behavior, or missing functionality:

- [Open an issue](https://github.com/AI4SCR/ai4bmr-datasets/issues) with a clear title and minimal reproducible example.  
- Include relevant error messages and version information if possible.

We monitor issues regularly and appreciate detailed, constructive reports.

### 💬 Seeking Support

If you need help using the software:

- Check the existing [issues](https://github.com/AI4SCR/ai4bmr-datasets/issues) and [discussions](https://github.com/AI4SCR/ai4bmr-datasets/discussions) first.  
- For general questions or usage advice, feel free to open a [discussion topic](https://github.com/AI4SCR/ai4bmr-datasets/discussions/new).  
- For dataset-specific problems, include dataset identifiers and loading code if applicable.

We aim to foster a welcoming and respectful community. Please be kind and collaborative in your interactions.

---

## 📬 Contact

For questions, issues, or contributions, feel free to contact:

**Adriano Martinelli**  
📧 adriano.martinelli@chuv.ch  
📧 adrianom@student.ethz.ch
