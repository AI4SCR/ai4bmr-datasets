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

# Statement of Need

Spatially-resolved omics technologies generate high-dimensional datasets with complex formats that often vary greatly
between studies. In emerging fields like spatial proteomics (e.g., IMC) and spatial transcriptomics, researchers face a
lack of unified frameworks to access and process heterogeneous datasets, hindering reproducibility and cross-study
analyses. Each dataset typically requires custom scripts to handle its specific data structure, which creates barriers
to entry, complicates usage and hinders robust benchmarking. `ai4bmr-datasets` addresses this gap by providing a single interface to
multiple well-known spatial omics datasets, abstracting away dataset-specific idiosyncrasies. This unified approach
allows scientists to focus on biological and analytical questions rather than data wrangling, and it supports consistent
preprocessing pipelines across different studies that people can easily adapt and extend by providing predefined interfaces for common processing steps.

The importance of data standardization in spatial omics has been highlighted by recent efforts in the community (e.g.,
developing standards for sharing spatial transcriptomics data [@SharingSpatial2023]), underscoring the need
for tools like `ai4bmr-datasets` that facilitate data sharing, reproducibility, and comparative analysis. By harmonizing
data access, our package enables more straightforward integration of spatial omics data into machine learning and
statistical modeling frameworks, ultimately accelerating biomedical discovery.

# Supported Datasets

The package supports the following public spatial omics datasets:
# TODO: add stats

- **Keren et al. 2018** – IMC of triple-negative breast cancer [@Keren2018]
- **Jackson et al. 2023** – IMC of breast cancer [@Jackson2020]
- **Danenberg et al. 2022** – IMC of breast cancer [@Danenberg2022]
- **Cords et al. 2024** – IMC of NSCLC [@Cords2024]

Additionally, dummy datasets are provided to mimic real data structure for development and testing purposes.

Each dataset is accessible through a standardized class interface that mimics the lightning philosophy and includes methods for downloading, preparing, and
accessing processed components (images, masks, metadata, and spatial coordinates). These datasets follow consistent
naming conventions and data schemas, making them immediately usable for downstream tasks.

# Example Usage

```python
from ai4bmr_datasets import Keren2018
from pathlib import Path

dataset = Keren2018(base_dir=Path("<PATH>"))
dataset.prepare_data()
dataset.setup(image_version="v1", mask_version="v1")

print(dataset.images)  # list of images
print(dataset.masks)  # list of masks
print(dataset.intensity.shape)  # cell x marker matrix
print(dataset.metadata.shape)  # cell x annotation matrix
```

# Conclusion

`ai4bmr-datasets` lowers the technical barrier to working with spatial omics data by providing unified, open access to
several published datasets and processing routines. Its modular design and standardized outputs make it a practical tool
for researchers developing computational methods in spatial biology. We welcome contributions and extensions from the
community and envision this package as a foundation for reproducible spatial omics analysis.

# References


