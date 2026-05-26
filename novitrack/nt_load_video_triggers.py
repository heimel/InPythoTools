"""Load Raspberry Pi video trigger logs for NoviTrack movies."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inpythotools.logmsg import logmsg
from .nt_load_parameters import nt_load_parameters
from .nt_session_path import nt_session_path


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _movie_stem(record: Any, session_path: Path, camera_name: str) -> Path:
    sessionid = str(_get(record, "sessionid", ""))
    condition = str(_get(record, "condition", ""))
    stimulus = str(_get(record, "stimulus", ""))

    candidates = [
        session_path / f"{sessionid}_{condition}_{stimulus}_{camera_name}",
        session_path / f"{sessionid}_{stimulus}_{camera_name}",
        session_path / f"{sessionid}_{camera_name}",
    ]
    for candidate in candidates:
        if list(candidate.parent.glob(candidate.name + ".*")):
            return candidate
    return candidates[-1]


def _parse_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def _read_trigger_csv(filename: str | Path) -> np.ndarray:
    rows: list[tuple[bool, float]] = []
    with Path(filename).open(newline="") as csvfile:
        for row in csv.reader(csvfile):
            values = [item.strip() for item in row if item.strip()]
            if not values:
                continue

            if len(values) == 1:
                value = _parse_float(values[0])
                if value is not None:
                    rows.append((False, value))
                continue

            time_seconds = _parse_float(values[-1])
            if time_seconds is None:
                continue
            rows.append((True, time_seconds))

    if any(is_multicolumn for is_multicolumn, _value in rows):
        return np.asarray([[np.nan, np.nan, value] for _is_multicolumn, value in rows], dtype=float)
    return np.asarray([[value] for _is_multicolumn, value in rows], dtype=float)


def nt_load_video_triggers(
    record: Any,
    camera_name: str,
    framerate: float = 30.0,
    *,
    params: Any | None = None,
    session_path: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return camera trigger times and event rows.

    The trigger format mirrors MATLAB ``nt_load_video_triggers.m``. Old one
    column files are frame numbers, newer files contain timestamps in column 3
    after a header/start row.
    """
    if params is None:
        params = nt_load_parameters(record)
    if session_path is None:
        folder, _ = nt_session_path(record, params)
    else:
        folder = Path(session_path)

    stem = _movie_stem(record, folder, camera_name)
    trigger_filename = stem.with_name(stem.name + "_triggers.csv")
    if not trigger_filename.exists():
        logmsg(f"Cannot find trigger file {trigger_filename}. Setting trigger after first frame.")
        triggers = np.array([1.0 / float(framerate)], dtype=float)
        events = pd.DataFrame({"time": triggers, "code": ["trigger1"], "duration": [0.001]})
        return triggers, events

    data = _read_trigger_csv(trigger_filename)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[1] == 1:
        triggers = data[:, 0] / float(framerate)
        times = triggers
    else:
        times = data[:, 2]
        triggers = data[1:, 2] if data.shape[0] > 1 else np.array([], dtype=float)

    if triggers.size == 0:
        logmsg("No video triggers found. Adding trigger at 0:00.")
        triggers = np.array([0.0], dtype=float)

    events = pd.DataFrame(
        {
            "time": times,
            "code": ["start"] + ["trigger1"] * max(0, len(times) - 1),
            "duration": np.full(len(times), 0.001),
        }
    )
    return np.asarray(triggers, dtype=float).reshape(-1), events


__all__ = ["nt_load_video_triggers"]
