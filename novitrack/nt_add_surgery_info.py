"""Add surgery/fiber metadata to NoviTrack records."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import pandas as pd

from inpythotools.logmsg import logmsg
from .nt_session_path import nt_session_path


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_string(value: Any) -> str:
    if _is_empty(value):
        return ""
    return str(value)


def _row_value(row: pd.Series, name: str, default: Any = "") -> Any:
    return row[name] if name in row.index else default


def _fiber_info(row: pd.Series, fiber: str) -> dict[str, str] | None:
    location = _row_value(row, f"{fiber}_location", "")
    if _is_empty(location):
        return None
    return {
        "hemisphere": _as_string(_row_value(row, f"{fiber}_hemisphere", "")),
        "location": _as_string(location),
        "green_sensor": _as_string(_row_value(row, f"{fiber}_green", "")),
        "red_sensor": _as_string(_row_value(row, f"{fiber}_red", "")),
    }


def nt_add_surgery_info(record: Any, params: Any | None = None) -> dict[str, Any]:
    """Add surgery sheet metadata to ``record["measures"]`` when available."""
    out = deepcopy(dict(record))
    out.setdefault("measures", {})

    session_folder, exists = nt_session_path(out, params)
    if not exists:
        logmsg(f"Cannot find session folder for {_get(out, 'sessionid', 'record')}")
        return out

    filename = session_folder.parent.parent / "Surgery" / "Surgery_sites.xlsx"
    if not filename.exists():
        logmsg(f"Cannot find surgery log for {_get(out, 'sessionid', 'record')}")
        return out

    try:
        surgery_table = pd.read_excel(filename, sheet_name="Sheet1")
    except Exception as exc:  # noqa: BLE001 - optional metadata should not stop analysis.
        logmsg(f"Could not read surgery log {filename}: {exc}")
        return out

    subject = str(_get(out, "subject", ""))
    subject_values = surgery_table.get("subject")
    if subject_values is None:
        logmsg(f"Could not find subject column in {filename}")
        return out

    subject_strings = subject_values.astype(str)
    matches = surgery_table[(subject_strings == f"#{subject}") | (subject_strings == f"#0{subject}")]
    if matches.empty:
        logmsg(f"Could not find mouse #{subject} in Surgery_sites.xlsx")
        return out
    if len(matches) > 1:
        logmsg(f"More than one mouse #{subject} in Surgery_sites.xlsx")
        return out

    row = matches.iloc[0]
    measures = dict(_get(out, "measures", {}))
    measures["strain"] = _as_string(_row_value(row, "strain", ""))
    measures["surgery_comment"] = _as_string(_row_value(row, "comment", ""))

    fiber_info: dict[str, dict[str, str]] = {}
    for fiber in ("fiber1", "fiber2"):
        info = _fiber_info(row, fiber)
        if info is not None:
            fiber_info[fiber] = info
    if fiber_info:
        measures["fiber_info"] = fiber_info

    out["measures"] = measures
    return out


__all__ = ["nt_add_surgery_info"]
