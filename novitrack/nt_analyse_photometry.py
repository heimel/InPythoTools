"""Analyze loaded NoviTrack photometry data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from inpythotools.logmsg import logmsg
from .nt_load_photometry import nt_load_photometry
from .nt_preprocess_photometry import nt_preprocess_photometry


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _interp_linear(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    return np.interp(x_new, x[valid], y[valid], left=np.nan, right=np.nan)


def compute_maps(
    nt_data: Mapping[str, Any],
    photometry: Mapping[str, Any],
    measures: dict[str, Any],
    params: Any,
) -> dict[str, Any]:
    """Compute spatial photometry maps."""
    if not bool(_get(params, "nt_compute_maps", False)):
        return measures

    com_x = _as_array(_get(nt_data, "CoM_X"))
    if com_x.size == 0 or np.all(np.isnan(com_x)):
        return measures

    nt_time = _as_array(_get(nt_data, "Time"))
    com_y = _as_array(_get(nt_data, "CoM_Y"))
    period = _as_array(_get(measures, "period_of_interest"))
    ind = np.where((nt_time >= period[0]) & (nt_time <= period[1]))[0]
    if ind.size == 0:
        return measures

    time = nt_time[ind]
    n_x = int(np.ceil(np.sqrt(float(_get(params, "nt_map_bins", 100)))))
    n_y = n_x
    range_x = np.array([np.nanmin(com_x[ind]), np.nanmax(com_x[ind])])
    range_y = np.array([np.nanmin(com_y[ind]), np.nanmax(com_y[ind])])
    resolution = min(np.diff(range_x)[0] / n_x, np.diff(range_y)[0] / n_y)
    if np.isnan(np.diff(range_x)[0]) or np.isnan(np.diff(range_y)[0]) or resolution <= 0:
        return measures

    n_x = int(np.ceil(np.diff(range_x)[0] / resolution))
    n_y = int(np.ceil(np.diff(range_y)[0] / resolution))

    x = np.ceil((com_x[ind] - range_x[0]) / resolution).astype(float)
    y = np.ceil((com_y[ind] - range_y[0]) / resolution).astype(float)
    x[x == 0] = 1
    y[y == 0] = 1

    counts = np.zeros((n_x, n_y))
    for xi, yi in zip(x, y):
        if np.isfinite(xi) and np.isfinite(yi):
            counts[int(xi) - 1, int(yi) - 1] += 1

    maps = measures.setdefault("maps", {})
    maps["counts"] = counts
    for channel in _get(measures, "channels", []):
        channel_name = _get(channel, "channel")
        maps.setdefault(channel_name, {})
        for light in _get(channel, "lights", []):
            light_type = _get(light, "type")
            ph = _interp_linear(
                _as_array(photometry[channel_name][light_type]["time"]),
                _as_array(photometry[channel_name][light_type]["signal"]),
                time,
            )
            ph_map = np.full((n_x, n_y), np.nan)
            for xi, yi, value in zip(x, y, ph):
                if not (np.isfinite(xi) and np.isfinite(yi)):
                    continue
                ix = int(xi) - 1
                iy = int(yi) - 1
                ph_map[ix, iy] = value if np.isnan(ph_map[ix, iy]) else ph_map[ix, iy] + value
            maps[channel_name][light_type] = ph_map / counts

    return measures


def compute_correlations(
    nt_data: Mapping[str, Any],
    photometry: Mapping[str, Any],
    measures: dict[str, Any],
) -> dict[str, Any]:
    """Compute photometry correlations with behavior variables."""
    variables = ("Speed", "Angular_velocity", "Abs_angular_velocity", "Distance_to_center")
    nt_time = _as_array(_get(nt_data, "Time"))
    if nt_time.size < 2:
        return measures

    nt_sample_rate = 1 / np.nanmedian(np.diff(nt_time))
    period = _as_array(_get(measures, "period_of_interest"))
    correlations: dict[str, Any] = {}

    for variable in variables:
        motion = _as_array(_get(nt_data, variable))
        if motion.size == 0:
            continue
        for channel in _get(measures, "channels", []):
            channel_name = _get(channel, "channel")
            photometry_sample_rate = float(_get(channel, "sample_rate"))
            resample_motion = photometry_sample_rate < nt_sample_rate
            for light in _get(channel, "lights", []):
                light_type = _get(light, "type")
                ph_time = _as_array(photometry[channel_name][light_type]["time"])
                ph_signal = _as_array(photometry[channel_name][light_type]["signal"])
                if resample_motion:
                    t = ph_time
                    mask = (t > period[0]) & (t < period[1])
                    x = ph_signal[mask]
                    y = _interp_linear(nt_time, motion, t[mask])
                else:
                    t = nt_time
                    mask = (t > period[0]) & (t < period[1])
                    x = motion[mask]
                    y = _interp_linear(ph_time, ph_signal, t[mask])
                valid = np.isfinite(x) & np.isfinite(y)
                if np.sum(valid) < 3:
                    continue
                cc, p_value = stats.pearsonr(x[valid], y[valid])
                if p_value < 0.10:
                    logmsg("Found some correlation")
                    correlations.setdefault(channel_name, {}).setdefault(light_type, {})[variable] = float(cc)

    measures["correlation"] = correlations
    return measures


def nt_analyse_photometry(
    record: Mapping[str, Any],
    nt_data: Mapping[str, Any] | None,
    params: Any,
    *,
    photometry_folder: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load, preprocess, and analyze photometry for one record."""
    measures = dict(_get(record, "measures", {}))
    if not measures:
        logmsg("No data in measures. Track first.")
        return dict(record), {}, {}

    photometry, measures = nt_load_photometry(
        record, params, photometry_folder=photometry_folder
    )
    out_record = dict(record)
    if not photometry:
        out_record["measures"] = measures
        return out_record, {}, {}

    photometry, measures = nt_preprocess_photometry(photometry, measures, params)
    if nt_data:
        measures = compute_maps(nt_data, photometry, measures, params)
        measures = compute_correlations(nt_data, photometry, measures)

    out_record["measures"] = measures
    return out_record, photometry, {}
