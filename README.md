---
title: "ai4bmr-datasets: A unified interface for spatial omics data access for computer vision and machine learning"
tags:
  - Python
  - spatial omics
  - imaging mass cytometry
  - spatial transcriptomics
  - computational pathology
authors:
  - name: Adriano Martinelli
    orcid: 0000-0002-9288-5103
    affiliation: 1
  - name: Marianna Rapsomaniki
    orcid: 0000-0003-3883-4871
    affiliation: 1, 3
affiliations:
  - index: 1
    name: University Hospital Lausanne (CHUV), Lausanne, Switzerland
  - index: 2
    name: ETH Zürich, Zürich, Switzerland
  - index: 3
    name: University of Lausanne (UNIL), Lausanne, Switzerland
date: 2025-05-28
bibliography: paper.bib
#csl: https://raw.githubusercontent.com/citation-style-language/styles/master/nature.csl
---

# Summary

**ai4bmr-datasets** is an open-source Python package that provides a harmonized and standardized interface for accessing
spatial omics datasets, including imaging mass cytometry (IMC) and multiplexed ion beam imaging (MIBI) data. The
package enables researchers to load raw spatially-resolved omics data from multiple studies in a unified format, apply
and retrieve data structures ready for downstream analysis or model training. By
focusing on open-source raw data processing and enforcing common data schemas (e.g., standardized image and single-cell
data formats), `ai4bmr-datasets` promotes reproducible and efficient research in computational and spatial
biology. The library is currently used internally within our group, but is designed to serve the broader community
working on spatial omics by easing data access and integration into machine learning workflows.

# Supported Datasets

The package supports the following public spatial omics datasets:

# TODO: add stats

- **Keren et al. 2018** – IMC of triple-negative breast cancer [@Keren2018]
- **Jackson et al. 2023** – IMC of breast cancer [@Jackson2020]
- **Danenberg et al. 2022** – IMC of breast cancer [@Danenberg2022]
- **Cords et al. 2024** – IMC of NSCLC [@Cords2024]

Additionally, dummy datasets are provided to mimic real data structure for development and testing purposes.

Each dataset is accessible through a standardized class interface that mimics the lightning philosophy and includes
methods for downloading, preparing, and
accessing processed components (images, masks, metadata, and spatial coordinates). These datasets follow consistent
naming conventions and data schemas, making them immediately usable for downstream tasks.

## Data

**Data Model:** All datasets are represented in a **consistent format**, enabling the same code to work across different
cancer types. Each loaded dataset is essentially a dictionary with the following keys:

- `metadata` – Observation-level metadata (pandas DataFrame) including a key `label`
- `clinical` – Clinical / Sample-level metadata (pandas DataFrame)
- `images` – Dictionary of image objects
- `masks` – Dictionary of mask objects
- `panel` – Imaging panel description (DataFrame), including a key `target`
- `intensity` – Object / Single-cell intensity features (DataFrame)
- `spatial` – Object / Single-cell spatial features (DataFrame)

These components are enforced and standardized across datasets.

**Efficient Data Loading:** Images and masks are **lazily loaded** from disk to avoid unnecessary memory use. Tabular
data are stored in Parquet format for fast access.

## Installation

You can install the package as follows:

```bash
pip install git+https://github.com/AI4SCR/ai4bmr-core.git 
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git
```

## Quickstart

1**Load the dataset:**

```python
from pathlib import Path
from ai4bmr_datasets.datasets import Keren2018  # or another dataset class

base_dir = Path("/path/to/TNBC")  # if none saved to `~/.cache/ai4bmr-datasets`
ds = Keren2018(base_dir=base_dir)
ds.prepare_data()  # download and prepare the dataset, only needed once
ds.setup(image_version='published', mask_version='published')

print(ds.images)  # list of images
print(ds.masks)  # list of masks
print(ds.intensity.shape)  # cell x marker matrix
print(ds.metadata.shape)  # cell x annotation matrix
```

3. **Access components:**

```python
images = ds.images
masks = ds.masks

sample_id = next(iter(images.keys()))
image_obj = images[sample_id]
mask_obj = masks[sample_id]

img_array = image_obj.data
mask_array = mask_obj.data

print("Image shape:", img_array.shape)
print("Mask shape:", mask_array.shape)
```

## Contact
Adriano Martinelli <adriano.martinelli@chuv.ch> or <adrianom@student.ethz.ch>


