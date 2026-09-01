"""
Utilities for loading MATLAB struct arrays into pandas DataFrames.

The main function is `load_mat_database`, which reads a MATLAB .mat file,
converts a MATLAB array of structs into a list of Python dictionaries, and then
wraps it in a pandas DataFrame.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import shutil
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.io.matlab import mat_struct


_MATLAB_FIELD_RE = re.compile(r"^[A-Za-z]\w*$")


def _modification_timestamp_ns(filename: Path) -> int:
    """Return the file's modification timestamp in nanoseconds."""
    return filename.stat().st_mtime_ns


def _backup_existing_mat_database(filename: Path) -> Path | None:
    """Move an existing database aside unless its backup is already current."""
    if not filename.is_file():
        return None

    backup = Path(f"{filename}_copy")
    if (
        backup.is_file()
        and _modification_timestamp_ns(backup) == _modification_timestamp_ns(filename)
    ):
        return backup

    os.replace(filename, backup)
    return backup


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


def _is_matlab_opaque_table(value: Any) -> bool:
    """Return True for MATLAB table objects that SciPy loaded as MCOS placeholders."""
    return (
        isinstance(value, tuple)
        and len(value) >= 3
        and value[1] == b"MCOS"
        and value[2] == b"table"
    )


def _events_from_markers(markers: Any) -> list[dict[str, Any]] | None:
    """Recreate a NoviTrack events table from marker structs when possible."""
    if not isinstance(markers, list):
        return None

    events: list[dict[str, Any]] = []
    for marker in markers:
        if not isinstance(marker, Mapping):
            return None
        if "time" not in marker or "marker" not in marker:
            return None
        events.append({"time": marker["time"], "event": marker["marker"]})
    return events


def _repair_known_matlab_tables(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace unsupported MATLAB table placeholders with useful Python records.

    MATLAB tables are stored as class objects. SciPy cannot reconstruct those
    objects, so it returns an opaque MCOS tuple instead. For NoviTrack
    databases, ``record.measures.events`` is derived from
    ``record.measures.markers`` as columns ``time`` and ``event``. Rebuilding it
    here prevents a save/load round trip from preserving the unusable MCOS
    placeholder.
    """
    for record in records:
        measures = record.get("measures")
        if not isinstance(measures, dict):
            continue

        if _is_matlab_opaque_table(measures.get("events")):
            events = _events_from_markers(measures.get("markers"))
            if events is not None:
                measures["events"] = events

    return records

def load_mat_database(filename: str | Path) -> pd.DataFrame:
    """Load a MATLAB test database named ``db`` and convert it to a DataFrame."""
    print(f"Loading {filename}")
    db = loadmat_as_dataframe(filename, variable_name="db")
    if "comment" in db.columns:
        db["comment"] = db["comment"].map(
            lambda value: "" if isinstance(value, np.ndarray) and value.size == 0 else value
        )
    return db

def loadmat_as_dataframe(
    filename: str | Path,
    variable_name: Optional[str] = None,
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

    records = _repair_known_matlab_tables(_struct_array_to_records(user_vars[variable_name]))
    
    db = pd.DataFrame.from_records(records)

    # make sessionid first column, but not set it as index
    if "sessionid" in db.columns:
        cols = ["sessionid"] + [c for c in db.columns if c != "sessionid"]
        db = db[cols]
    
    return db 


def _validate_matlab_field_name(name: str) -> None:
    """Raise a clear error if a DataFrame column cannot be a MATLAB field."""
    if not _MATLAB_FIELD_RE.match(name):
        raise ValueError(
            f"{name!r} is not a valid MATLAB struct field name. "
            "Use a name that starts with a letter and contains only letters, "
            "numbers, and underscores."
        )


def _is_missing_scalar(value: Any) -> bool:
    """Return True for scalar pandas/numpy missing values, but not arrays."""
    if isinstance(value, (np.ndarray, list, tuple, Mapping)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _records_to_mat_struct_array(records: list[Mapping[str, Any]]) -> np.ndarray:
    """Convert a list of dictionaries into a MATLAB-compatible struct array."""
    if not records:
        return np.empty((0, 0), dtype=object)
    if all(not record for record in records):
        # NumPy represents a struct array with no fields as dtype([]), which
        # scipy.io.savemat cannot write. An object array round-trips these as a
        # MATLAB cell array containing the same number of empty structs.
        return np.asarray([{} for _ in records], dtype=object)

    field_names = list(records[0].keys())
    for field_name in field_names:
        _validate_matlab_field_name(str(field_name))

    struct_array = np.empty((1, len(records)), dtype=[(str(name), object) for name in field_names])

    for index, record in enumerate(records):
        for field_name in field_names:
            struct_array[0, index][str(field_name)] = _convert_python_value_to_mat(record.get(field_name))

    return struct_array


def _convert_python_value_to_mat(value: Any) -> Any:
    """Recursively convert Python values into values suitable for ``savemat``."""
    if _is_missing_scalar(value):
        return np.array([])

    if isinstance(value, pd.Series):
        value = value.to_dict()

    if isinstance(value, pd.DataFrame):
        return _records_to_mat_struct_array(value.to_dict(orient="records"))

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            field_name = str(key)
            _validate_matlab_field_name(field_name)
            out[field_name] = _convert_python_value_to_mat(item)
        return out

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        values = list(value)
        if values and all(isinstance(item, Mapping) for item in values):
            return _records_to_mat_struct_array(values)
        converted = [_convert_python_value_to_mat(item) for item in values]
        try:
            return np.asarray(converted)
        except ValueError:
            return np.asarray(converted, dtype=object)

    return value


def save_mat_database(
    db: pd.DataFrame,
    filename: str | Path,
    variable_name: str = "db",
    *,
    do_compression: bool = True,
) -> None:
    """Save a session DataFrame as a MATLAB struct-array database.

    Each row becomes one MATLAB struct in ``variable_name``. Columns become
    struct fields, including nested dictionaries such as ``measures``. Before
    overwriting an existing file, its current contents are moved to a sibling
    file whose name ends in ``_copy``. If saving creates a broken file before
    failing, the backup is copied back while the save exception is re-raised.
    """
    if not isinstance(db, pd.DataFrame):
        raise TypeError("save_mat_database expects a pandas DataFrame.")

    _validate_matlab_field_name(variable_name)
    records = db.to_dict(orient="records")
    mat_db = _records_to_mat_struct_array(records)

    filename = Path(filename)
    backup = _backup_existing_mat_database(filename)
    try:
        savemat(
            filename,
            {variable_name: mat_db},
            do_compression=do_compression,
            long_field_names=True,
        )
    except Exception:
        if backup is not None and backup.is_file() and filename.exists():
            try:
                shutil.copy2(backup, filename)
            except OSError:
                # Do not hide the original save error from the caller.
                pass
        raise
