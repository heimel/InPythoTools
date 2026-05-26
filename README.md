# InPythoTools #

InPythoTools are a python-port of some InVivoTools functions.

## Installation

The tools can be run from a conda environment with Python 3.11 and PyQt6.
Create and activate the environment with:

```bash
conda create -n pyqt6_env python=3.11 pyqt6 -y -c conda-forge
conda activate pyqt6_env
```

Install the remaining dependencies with:

```bash
conda install -y -c conda-forge pandas scipy matplotlib statsmodels pytest openpyxl nptdms pyqtgraph opencv spyder-kernels pyyaml
```

When using Spyder, start Spyder from an environment that can connect to this
kernel, or select the `pyqt6_env` interpreter/kernel in Spyder after installing
`spyder-kernels`.

Maintainer: Alexander Heimel
