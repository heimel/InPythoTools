"""Preprocess NoviTrack fiber-photometry signals.

Python translation of ``nt_preprocess_photometry.m``.

The expected data shape mirrors the MATLAB structure:

``photometry[channel][light_type]["time"]``
``photometry[channel][light_type]["signal"]``

``measures["channels"]`` is a list of channel dictionaries with ``channel``,
``lights`` and ``sample_rate`` fields.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Get a field from a dict-like or object-like value."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set(obj: Any, name: str, value: Any) -> None:
    """Set a field on a dict-like or object-like value."""
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _param(params: Any, name: str, default: Any = None) -> Any:
    return _get(params, name, default)


def _period_mask(time: np.ndarray, measures: Any) -> np.ndarray:
    period = _as_array(_get(measures, "period_of_interest", [-np.inf, np.inf]))
    return (time >= period[0]) & (time <= period[1])


def _marker_pretime_mask(time: np.ndarray, measures: Any, pretime: float) -> np.ndarray:
    mask = np.zeros(time.shape, dtype=bool)
    markers = _get(measures, "markers", [])
    for marker in markers:
        marker_time = float(_get(marker, "time", np.nan))
        if np.isfinite(marker_time):
            mask |= (time > marker_time - pretime) & (time < marker_time)
    return mask


def _interp_spline_extrap(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    keep = np.concatenate(([True], np.diff(x) > 0))
    x = x[keep]
    y = y[keep]

    if x.size < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    if x.size < 4:
        return np.interp(x_new, x, y, left=y[0], right=y[-1])

    return CubicSpline(x, y, extrapolate=True)(x_new)


def median_filter_truncate(x: np.ndarray, window: int) -> np.ndarray:
    """Median filter similar to MATLAB ``medfilt1(..., 'truncate')``."""
    if window < 3:
        return x

    half = window // 2
    out = np.empty_like(x, dtype=float)
    for index in range(x.size):
        start = max(0, index - half)
        stop = min(x.size, index + half + 1)
        out[index] = np.nanmedian(x[start:stop])
    return out


def filter_photometry(x: Any, fs: float, params: Any) -> np.ndarray:
    """Filter one photometry trace."""
    x = _as_array(x)

    low_pass = float(_param(params, "nt_photometry_low_pass", np.inf))
    high_pass = float(_param(params, "nt_photometry_high_pass", 0))
    order = int(_param(params, "nt_photometry_butterworth_order", 2))

    median_window = round(float(_param(params, "nt_photometry_median_filter_window", 0)) * fs)
    if median_window >= 3 and median_window % 2 == 0:
        median_window += 1
    if median_window >= 3:
        x = median_filter_truncate(x, median_window)

    if high_pass > 0:
        b_high, a_high = signal.butter(order, high_pass / (fs / 2), btype="high")
        x = signal.filtfilt(b_high, a_high, x)

    if low_pass > 0 and low_pass < fs / 2:
        b_low, a_low = signal.butter(order, low_pass / (fs / 2), btype="low")
        x = signal.filtfilt(b_low, a_low, x)

    return x


def nt_preprocess_photometry(
    photometry: Mapping[str, Any],
    measures: Mapping[str, Any],
    params: Any,
    *,
    copy: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply isosbestic correction, filtering, and time offset.

    Parameters
    ----------
    photometry:
        Nested photometry dictionary, usually loaded from ``nt_photometry.mat``
        or produced by a future Python ``nt_load_photometry``.
    measures:
        Session measures dictionary containing ``channels``, ``markers`` and
        ``period_of_interest``.
    params:
        Dict-like parameter object, for example from ``nt_load_parameters``.
    copy:
        If true, deep-copy inputs before modifying them.
    """
    if copy:
        photometry = deepcopy(dict(photometry))
        measures = deepcopy(dict(measures))
    else:
        photometry = dict(photometry)
        measures = dict(measures)

    channels = _get(measures, "channels", [])

    if bool(_param(params, "nt_photometry_isosbestic_correction", False)):
        for channel_index, channel in enumerate(channels):
            channel_name = _get(channel, "channel")
            lights = _get(channel, "lights", [])
            isos_lights = [
                light for light in lights if "isosbestic" in str(_get(light, "type", "")).lower()
            ]
            signal_lights = [
                light for light in lights if light not in isos_lights
            ]

            if not isos_lights or not signal_lights:
                continue

            isos_type = _get(isos_lights[0], "type")
            iso_time = _as_array(photometry[channel_name][isos_type]["time"])
            iso_signal = _as_array(photometry[channel_name][isos_type]["signal"])

            for light in signal_lights:
                light_type = _get(light, "type")
                time = _as_array(photometry[channel_name][light_type]["time"])
                f_signal = _as_array(photometry[channel_name][light_type]["signal"])
                f_iso = _interp_spline_extrap(iso_time, iso_signal, time)

                mask = _period_mask(time, measures)
                if bool(_param(params, "nt_only_use_pretime_for_isosbestic_correction", False)):
                    mask = _marker_pretime_mask(
                        time, measures, float(_param(params, "nt_pretime", 10))
                    )

                x_fit = np.column_stack([f_iso[mask], np.ones(np.sum(mask))])
                y_fit = f_signal[mask]
                valid = np.isfinite(x_fit).all(axis=1) & np.isfinite(y_fit)
                if np.sum(valid) < 2:
                    fit = np.array([np.nan, np.nan])
                    artifact = np.zeros_like(f_signal)
                else:
                    fit, *_ = np.linalg.lstsq(x_fit[valid], y_fit[valid], rcond=None)
                    artifact = np.column_stack([f_iso, np.ones(f_iso.size)]) @ fit

                _set(channel, "fit_isos", fit)
                channels[channel_index] = channel
                photometry[channel_name][light_type]["signal"] = f_signal - artifact

    _set(measures, "photometry_isosbestic_correction", bool(_param(params, "nt_photometry_isosbestic_correction", False)))

    for channel in channels:
        channel_name = _get(channel, "channel")
        fs = float(_get(channel, "sample_rate"))
        for light in _get(channel, "lights", []):
            light_type = _get(light, "type")
            photometry[channel_name][light_type]["signal"] = filter_photometry(
                photometry[channel_name][light_type]["signal"], fs, params
            )

    time_offset = float(_param(params, "nt_photometry_time_offset", 0))
    if time_offset:
        for channel in channels:
            channel_name = _get(channel, "channel")
            for light in _get(channel, "lights", []):
                light_type = _get(light, "type")
                photometry[channel_name][light_type]["time"] = (
                    _as_array(photometry[channel_name][light_type]["time"]) - time_offset
                )

    _set(measures, "channels", channels)
    return photometry, measures
