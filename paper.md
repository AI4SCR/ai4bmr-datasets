---
title: "**SpatialOmicsNet** : A unified interface for spatial proteomics data access for computer vision and machine learning"
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

`SpatialOmicsNet` is an open-source Python package that provides a harmonized and standardized interface for accessing
spatial proteomics and multiplexed imaging datasets, including imaging mass cytometry (
IMC)[@giesenHighlyMultiplexedImaging2014] and multiplexed ion beam
imaging (MIBI-TOF)[@kerenMIBITOFMultiplexedImaging2019] data. The package enables researchers to load raw spatially-resolved
proteomics data from multiple
studies in a unified format, apply and retrieve data structures ready for downstream machine learning analysis or model
training. By focusing on open-source raw data processing and enforcing common data schemas (e.g., standardized image and
single-cell data formats), `SpatialOmicsNet` promotes reproducible and efficient research in computational and spatial
biology. The library is designed to serve the broader community working on spatial proteomics by easing data access and
integration into machine learning workflows.

# Statement of Need

Spatially-resolved proteomics, recently named Nature Method of the Year 2024[@MethodYear20242024], enable the
quantification of proteins in
single cells within their tissue context, revealing intricate aspects of spatial cellular arrangement and communication.
In the context of cancer, these advancements provide unprecedented insights into the heterogeneity of the tumor and its
microenvironment, and the underlying mechanisms affecting tumor initiation, progression, and response to
treatment[@lewisSpatialOmicsMultiplexed2021].
IMC and MIBI-TOF are among the most popular
technologies, with dozens of high-dimensional datasets made publicly available per year. The increasing availability of
these datasets has fueled algorithmic development in machine learning and computer vision. Numerous models that perform
a variety of tasks, such as cell segmentation[@greenwaldWholecellSegmentationTissue2022], cell type
annotation[@geuenichAutomatedAssignmentCell2021], representation learning[@wencksternAIpoweredVirtualTissues2025a] or
heterogeneity
analysis[@martinelliATHENAAnalysisTumor2022a] tailored to spatial proteomics data have been recently developed, with
corresponding widely used packages.

However, a critical gap hindering model development, reproducibility and cross-study analyses is the lack of unified
frameworks to access and process the data. Spatial proteomics datasets, often deposited in public repositories such as
Zenodo[@https://doi.org/10.25495/7gxk-rd71] or Figshare[@FigshareCreditAll], typically contain a collection of
components, such as raw and preprocessed images, segmentation
masks, extracted single-cell intensities, panel descriptions and associated clinical metadata, uploaded in disparate,
non-standardized formats (e.g., mixed .tiff, .csv, custom JSONs), with varying metadata structures and inconsistent
preprocessing that vary greatly between studies and labs. Working with these fragmented datasets implies a significant
time investment for researchers to locate and download the data, and write custom scripts to handle their specific data
structure, creating barriers to entry, complicating usage and hindering robust benchmarking. While existing data
frameworks developed by the spatial transcriptomics community such as
SpatialData[@marconatoSpatialDataOpenUniversal2025] and Pysodb[@yuanSODBFacilitatesComprehensive2023] are gaining
popularity
and can be extended to spatial proteomics, they often come with heavier dependencies and general-purpose abstractions
that may be unnecessarily complex for researchers focused on fast, standardized access to real-world IMC or MIBI
datasets.

`SpatialOmicsNet` is an open-source Python package that addresses these gaps by:

  - Providing a lightweight, unified interface to widely-used curated spatial proteomics datasets.
  - Abstracting dataset-specific structure, letting users access data components (images, masks, metadata) through a
consistent schema.
  - Supporting reproducible preprocessing via modular, reusable interfaces for common pipeline steps.
  - Facilitating integration in machine learning and computer vision models by streamlining dataset loading into standard
formats.
  - Encouraging community contributions for expanding and maintaining harmonized dataset access.

This unified approach allows scientists to abstract away dataset-specific idiosyncrasies and focus on biological and
analytical questions rather than data wrangling. `SpatialOmicsNet` is intentionally minimal, tailored to machine
learning and computer vision workflows (e.g., loading images, masks, and cell-level metadata with minimal setup) without depending on larger ecosystem packages (e.g., anndata, xarray, zarr, dask).
`SpatialOmicsNet` gives immediate access to curated datasets with ready-to-use utilities, eliminating the need to write custom loaders or
parse inconsistent formats. As such, it is particularly friendly to the growing community of ML developers, researchers,
and engineers entering the emerging field of spatial biology. By harmonizing data access, our package enables more
straightforward integration of spatial proteomics data into machine learning and modeling frameworks, ultimately
accelerating biomedical discovery.

# Supported Datasets

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

Additionally, dummy datasets are provided to mimic real data structure for development and testing purposes.

Each dataset is accessible through a standardized class interface that mimics the pytorch
lightning [@falcon2019pytorchlightning]
philosophy and includes methods for downloading, preparing, and accessing processed components (images, masks, features and metadata). These datasets follow consistent naming conventions and data schemas, making them immediately
usable for downstream tasks.

# Example Usage

```python
from ai4bmr_datasets import Jackson2020
from pathlib import Path

dataset = Jackson2020(base_dir=Path("<PATH>"))
dataset.prepare_data()
dataset.setup(image_version="published", mask_version="published")

print(dataset.sample_ids)  # list of sample IDs
print(dataset.images)  # list of images
print(dataset.masks)  # list of masks

dataset.setup(image_version="published", mask_version="published",
              feature_version='published', load_intensity=True,
              metadata_version='published', load_metadata=True,
              )
print(dataset.intensity.shape)  # cell x marker matrix
print(dataset.metadata.shape)  # cell x annotation matrix
```

# Conclusion

`SpatialOmicsNet` lowers the technical barrier to working with spatial proteomics data by providing unified, open access
to several published datasets and processing routines. Its modular design and standardized outputs make it a practical
tool for researchers developing computational methods in spatial biology. We welcome contributions and extensions from
the community and envision this package as a foundation for reproducible spatial proteomics analysis.

# References


