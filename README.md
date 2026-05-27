# InPythoTools

InPythoTools contains reusable Python tools that support Python ports of
Heimel-lab analysis code. It is the Python counterpart to general-purpose
MATLAB tools from InVivoTools.

NoviTrack-specific analysis code has moved to the `NoviTrack` repository in the
`novitrack` package. The reusable `inpythotools` package remains here.

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

## Python path

Python needs to know where this repository is before `inpythotools` can be
imported.

For one Python session:

```python
import sys
sys.path.append(r"C:\Users\alexa\Documents\Porting NoviTrack\InPythoTools")

from inpythotools import browse_database
```

For one PowerShell or conda terminal session:

```powershell
conda activate pyqt6_env
$env:PYTHONPATH = "C:\Users\alexa\Documents\Porting NoviTrack\InPythoTools;$env:PYTHONPATH"
python
```

To make this persistent for the conda environment:

```powershell
conda activate pyqt6_env
conda env config vars set PYTHONPATH="C:\Users\alexa\Documents\Porting NoviTrack\InPythoTools"
conda deactivate
conda activate pyqt6_env
```

Test the path with:

```powershell
python -c "from inpythotools import browse_database; print(browse_database)"
```

## Usage

```python
from inpythotools import browse_database
from inpythotools import load_mat_database

db = load_mat_database("database.mat")
browse_database(db)
```

## Local parameter overrides

Local machine-specific analysis settings are stored in a user configuration
folder, not in the repository. Create or open the file with:

```python
from inpythotools import edit_local_config

edit_local_config()
```

To choose a specific editor, for example VS Code:

```python
edit_local_config(editor="code")
```

If the `code` command is not available in your Python session, pass the full
path to the editor instead:

```python
edit_local_config(editor=r"C:\Users\alexa\AppData\Local\Programs\Microsoft VS Code\Code.exe")
```

To print the path without opening an editor:

```python
from inpythotools import local_config_path

print(local_config_path())
```

Maintainer: Alexander Heimel
