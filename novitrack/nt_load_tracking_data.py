"""Load precomputed NoviTrack tracking data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import loadmat

from inpythotools.mat_database import _convert_mat_value
from inpythotools.logmsg import logmsg
from .nt_load_neurotar_data import nt_load_neurotar_data
from .nt_session_path import nt_session_path


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        if isinstance(value, (list, tuple)):
            parts = [_as_array(item) for item in value]
            parts = [part for part in parts if part.size]
            if parts:
                return np.concatenate(parts)
        return np.array([], dtype=float)


def _median_filter_omitnan(x: np.ndarray, width: int) -> np.ndarray:
    """Median filter roughly matching MATLAB ``medfilt1(...,'omitnan')``."""
    x = _as_array(x)
    if width <= 1 or x.size == 0:
        return x

    if width % 2 == 0:
        width += 1
    return signal.medfilt(x, kernel_size=width)


def _ensure_field(nt_data: dict[str, Any], field: str, value: Any) -> None:
    if field not in nt_data or np.asarray(nt_data[field]).size == 0:
        nt_data[field] = value


def _complete_tracking_fields(nt_data: dict[str, Any], params: Any) -> dict[str, Any]:
    """Add derived/default fields expected by downstream NoviTrack analysis."""
    time = _as_array(nt_data.get("Time", []))
    if time.size == 0:
        return nt_data

    n = time.size
    nan_vec = np.full(n, np.nan)
    filter_width = int(_get(params, "nt_pose_temporal_filter_width", 20))

    for x_name, y_name in (("X", "Y"), ("CoM_X", "CoM_Y"), ("tailbase_X", "tailbase_Y")):
        if x_name in nt_data:
            nt_data[x_name] = _median_filter_omitnan(_as_array(nt_data[x_name]), filter_width)
            nt_data[y_name] = _median_filter_omitnan(_as_array(nt_data[y_name]), filter_width)

    if "Speed" not in nt_data:
        if "CoM_X" in nt_data:
            dt = float(np.nanmean(np.diff(time)))
            overhead_mm_per_pixel = 0.5
            speed = np.full(n, np.nan)
            speed[:-1] = (
                np.sqrt(np.diff(_as_array(nt_data["CoM_X"])) ** 2 + np.diff(_as_array(nt_data["CoM_Y"])) ** 2)
                / dt
                * overhead_mm_per_pixel
                / 1000
            )
            nt_data["Speed"] = speed
        else:
            nt_data["Speed"] = nan_vec.copy()

    _ensure_field(nt_data, "X", nan_vec.copy())
    _ensure_field(nt_data, "Y", nan_vec.copy())
    _ensure_field(nt_data, "Coordinates", _get(params, "OVERHEAD", 4))
    _ensure_field(nt_data, "CoM_X", nan_vec.copy())
    _ensure_field(nt_data, "CoM_Y", nan_vec.copy())
    _ensure_field(nt_data, "tailbase_X", nan_vec.copy())
    _ensure_field(nt_data, "tailbase_Y", nan_vec.copy())

    if "alpha" not in nt_data:
        alpha = nan_vec.copy()
        if np.any(~np.isnan(nt_data["X"])) and np.any(~np.isnan(nt_data["CoM_X"])):
            vx = _as_array(nt_data["X"]) - _as_array(nt_data["CoM_X"])
            vy = _as_array(nt_data["Y"]) - _as_array(nt_data["CoM_Y"])
            alpha = np.angle(vy + 1j * vx) / np.pi * 180
        nt_data["alpha"] = alpha

    _ensure_field(nt_data, "Forward_speed", nan_vec.copy())

    if "Angular_velocity" not in nt_data:
        angular_velocity = nan_vec.copy()
        alpha = _as_array(nt_data["alpha"])
        if np.any(~np.isnan(alpha)):
            dt = float(np.nanmean(np.diff(time)))
            angular_velocity[1:] = np.angle(np.exp(1j * np.diff(alpha) / 180 * np.pi)) / dt
            angular_velocity = _median_filter_omitnan(angular_velocity, filter_width)
        nt_data["Angular_velocity"] = angular_velocity

    if "Abs_angular_velocity" not in nt_data:
        nt_data["Abs_angular_velocity"] = np.abs(_as_array(nt_data["Angular_velocity"]))

    if "Distance_to_center" not in nt_data:
        nt_data["Distance_to_center"] = np.sqrt(_as_array(nt_data["CoM_X"]) ** 2 + _as_array(nt_data["CoM_Y"]) ** 2)

    _ensure_field(nt_data, "Object_distance", nan_vec.copy())
    return nt_data


def nt_load_tracking_data(
    record: Any,
    params: Any,
    *,
    recompute: bool | None = None,
    session_path: str | Path | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Load tracking data into the NoviTrack format."""
    if recompute is None:
        recompute = bool(_get(params, "nt_recompute_tracking_data", False))

    if session_path is None:
        folder, exists = nt_session_path(record, params)
    else:
        folder = Path(session_path)
        exists = folder.is_dir()

    if not exists:
        logmsg(f"Folder {folder} does not exist.")
        return {}, _as_array(_get(_get(record, "measures", {}), "trigger_times", []))

    filename = folder / "nt_tracking_data.mat"
    if filename.exists() and not recompute:
        mat = loadmat(filename, squeeze_me=True, struct_as_record=False)
        nt_data = _convert_mat_value(mat["nt_data"])
        nt_data = _complete_tracking_fields(nt_data, params)
        trigger_times = _as_array(_get(_get(record, "measures", {}), "trigger_times", []))
        return nt_data, trigger_times

    nt_data, _ = nt_load_neurotar_data(record, params)
    if nt_data:
        logmsg("Not yet reading in all triggers. Assuming one trigger broadcast by Neurotar at time 0.")
        return nt_data, np.array([0.0])

    if recompute:
        logmsg("Non-Neurotar tracking-data recompute branches are not ported yet. Loading precomputed data if present.")

    if filename.exists():
        mat = loadmat(filename, squeeze_me=True, struct_as_record=False)
        nt_data = _convert_mat_value(mat["nt_data"])
        nt_data = _complete_tracking_fields(nt_data, params)
        trigger_times = _as_array(_get(_get(record, "measures", {}), "trigger_times", []))
        return nt_data, trigger_times

    logmsg(f"Precomputed tracking data not found: {filename}")
    return {}, _as_array(_get(_get(record, "measures", {}), "trigger_times", []))
