"""
Utilities for loading MATLAB struct arrays into pandas DataFrames.

The main function is `load_mat_database`, which reads a MATLAB .mat file,
converts a MATLAB array of structs into a list of Python dictionaries, and then
wraps it in a pandas DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import mat_struct


def _todict(obj: mat_struct) -> dict[str, Any]:
    """Recursively convert a scipy MATLAB mat_struct object to a dictionary."""
    out: dict[str, Any] = {}
    for field in obj._fieldnames:
        out[field] = _convert_mat_value(getattr(obj, field))
    return out


def _convert_mat_value(value: Any) -> Any:
    """
    Recursively convert values loaded from scipy.io.loadmat.

    Converts:
    - mat_struct objects -> dict
    - object arrays containing structs -> list / nested list
    - scalar numpy arrays -> scalar values where appropriate
    - bytes -> str

    Numeric arrays are preserved as numpy arrays unless they are scalar.
    """
    if isinstance(value, mat_struct):
        return _todict(value)

    if isinstance(value, bytes):
        return value.decode()

    if isinstance(value, np.ndarray):
        # Empty arrays are preserved.
        if value.size == 0:
            return value

        # Scalar arrays become scalars.
        if value.size == 1:
            return _convert_mat_value(value.item())

        # Object arrays often contain MATLAB structs/cells.
        if value.dtype == object:
            converted = [_convert_mat_value(v) for v in value.flat]
            if value.ndim == 1:
                return converted
            return np.array(converted, dtype=object).reshape(value.shape).tolist()

        # Character arrays can sometimes appear as arrays of single characters.
        if value.dtype.kind in {"U", "S"}:
            squeezed = np.squeeze(value)
            if squeezed.ndim == 0:
                return str(squeezed.item())
            if squeezed.ndim == 1:
                return "".join(map(str, squeezed.tolist()))

        # Keep real numeric arrays as numpy arrays.
        return value

    # Convert numpy scalar values to Python scalars.
    if isinstance(value, np.generic):
        return value.item()

    return value


def _struct_array_to_records(value: Any) -> list[dict[str, Any]]:
    """Convert a MATLAB struct array-like value into a list of dictionaries."""
    value = np.squeeze(value)

    if isinstance(value, mat_struct):
        return [_todict(value)]

    if isinstance(value, np.ndarray):
        records = []
        for item in value.flat:
            converted = _convert_mat_value(item)
            if not isinstance(converted, dict):
                raise TypeError(
                    "The selected MATLAB variable does not appear to be a struct array."
                )
            records.append(converted)
        return records

    converted = _convert_mat_value(value)
    if isinstance(converted, dict):
        return [converted]

    raise TypeError("The selected MATLAB variable does not appear to be a struct array.")

def load_mat_database(filename: str | Path) -> pd.DataFrame:
    """Load MATLAB database and convert to dataframe"""
    db = loadmat_as_dataframe(filename, variable_name = 'db')
    return db

def loadmat_as_dataframe(
    filename: str | Path,
    variable_name: Optional[str] = 'None',
    *,
    squeeze_me: bool = True,
    struct_as_record: bool = False,
) -> pd.DataFrame:
    """
    Load a MATLAB struct array from a .mat file and return it as a DataFrame.

    Parameters
    ----------
    filename:
        Path to the .mat file.
    variable_name:
        Name of the MATLAB variable containing the struct array. If omitted,
        the function tries to find a single non-internal variable in the file.
    squeeze_me:
        Passed to scipy.io.loadmat. The default removes singleton dimensions.
    struct_as_record:
        Passed to scipy.io.loadmat. Must usually be False to get mat_struct
        objects that can be converted recursively.

    Returns
    -------
    pandas.DataFrame
        One row per MATLAB struct-array element and one column per struct field.

    Examples
    --------
    >>> df = loadmat_as_dataframe("database.mat", "db")
    >>> df = loadmat_as_dataframe("database.mat")  # if the .mat file has one variable
    >>> df.loc[df["subject"] == "1236733", "measures"]
    """
    filename = Path(filename)

    mat = loadmat(
        filename,
        squeeze_me=squeeze_me,
        struct_as_record=struct_as_record,
    )

    user_vars = {k: v for k, v in mat.items() if not k.startswith("__")}

    if variable_name is None:
        if len(user_vars) != 1:
            names = ", ".join(user_vars.keys())
            raise ValueError(
                "Please specify `variable_name`; the .mat file contains "
                f"multiple variables: {names}"
            )
        variable_name = next(iter(user_vars))

    if variable_name not in user_vars:
        names = ", ".join(user_vars.keys())
        raise KeyError(
            f"Variable {variable_name!r} was not found in {filename}. "
            f"Available variables: {names}"
        )

    records = _struct_array_to_records(user_vars[variable_name])
    
    db = pd.DataFrame.from_records(records)

    # make sessionid first column, but not set it as index
    if "sessionid" in db.columns:
        cols = ["sessionid"] + [c for c in db.columns if c != "sessionid"]
        db = db[cols]
    
    return db 
