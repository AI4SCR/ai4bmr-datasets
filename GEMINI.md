# GEMINI.md

## Project Overview

This project, **SpatialProteomicsNet**, is an open-source Python package designed to provide a unified interface for accessing, preparing, and loading spatial proteomics datasets, such as Imaging Mass Cytometry (IMC) and Multiplexed Ion Beam Imaging (MIBI). The goal is to simplify the use of these datasets in machine learning workflows.

The package provides a consistent data model for various datasets, allowing researchers to work with different data sources using the same code. It handles the download and preprocessing of data and provides a lazy-loading mechanism for images and masks to manage memory efficiently.

The project is structured as a Python package with the source code in the `src/ai4bmr_datasets` directory. It includes modules for different datasets, data models, and utility functions.

## Building and Running

### Installation

The package can be installed using pip directly from the GitHub repository:

```bash
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git
```

The project dependencies are listed in the `pyproject.toml` file and include libraries like `pydantic`, `torch`, `numpy`, `pandas`, and `scikit-image`.

### Running Tests

The project uses `pytest` for testing. The tests are located in the `tests` directory. To run the tests, you can use the following command:

```bash
pytest
```

## Development Conventions

The project follows standard Python development conventions. The `pyproject.toml` file indicates the use of the following tools for code formatting and quality:

*   **Black:** for code formatting.
*   **isort:** for sorting imports.
*   **mypy:** for static type checking.
*   **flake8:** for linting.

The project also uses `pre-commit` to run these checks automatically before each commit. The configuration for `pre-commit` can be found in the `.pre-commit-config.yaml` file.

The contribution guidelines in the `README.md` file suggest forking the repository, creating a feature branch, and submitting a pull request.
