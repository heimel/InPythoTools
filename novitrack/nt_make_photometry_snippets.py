"""Cut peri-event snippets from preprocessed photometry traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from inpythotools.logmsg import logmsg
from .nt_get_events import nt_get_events


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _interp_linear_extrap(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Linear interpolation with extrapolation, matching MATLAB interp1/extrap."""
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    keep = np.concatenate(([True], np.diff(x) > 0))
    x = x[keep]
    y = y[keep]

    if x.size < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    out = np.interp(x_new, x, y)
    left = x_new < x[0]
    right = x_new > x[-1]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        out[left] = y[0] + slope * (x_new[left] - x[0])
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out[right] = y[-1] + slope * (x_new[right] - x[-1])
    return out


def nt_make_photometry_snippets(
    photometry: Mapping[str, Any],
    measures: Any,
    params: Any,
) -> dict[str, Any]:
    """Cut photometry snippets around all events.

    Returns a dictionary with MATLAB-like fields ``data``, ``baseline_std`` and
    ``unit``. Each ``data[channel_type]`` array has shape
    ``n_events x n_bins_per_snippet``.
    """
    markers = _get(measures, "markers", None)
    if markers is None or len(markers) == 0:
        return {}

    events = nt_get_events(measures, params)
    n_events = len(events)

    t_bins = _as_array(_get(measures, "snippets_tbins"))
    n_bins_per_snippet = t_bins.size
    mask_pre = t_bins < 0

    snippets: dict[str, Any] = {"data": {}, "baseline_std": {}, "unit": {}}

    pretime = float(_get(params, "nt_pretime", 10))
    posttime = float(_get(params, "nt_posttime", 20))
    bin_width = float(_get(params, "nt_photometry_bin_width", 0.1))
    subtract_baseline = bool(_get(params, "nt_photometry_subtract_baseline", True))
    zscore = bool(_get(params, "nt_photometry_zscoring", True))

    for channel in _get(measures, "channels", []):
        channel_name = _get(channel, "channel")
        for light in _get(channel, "lights", []):
            light_type = _get(light, "type")
            field = f"{channel_name}_{light_type}"
            data = np.full((n_events, n_bins_per_snippet), np.nan)
            snippets["unit"][field] = "raw"

            time = _as_array(photometry[channel_name][light_type]["time"])
            trace = _as_array(photometry[channel_name][light_type]["signal"])

            for event_index, event in events.iterrows():
                event_time = float(event["time"])
                mask = (time > event_time - pretime - bin_width) & (
                    time < event_time + posttime + bin_width
                )
                if np.sum(mask) < 3:
                    logmsg(f"No photometry data points for event at {event_time:.2g} s.")
                    snippet = np.full_like(t_bins, np.nan)
                else:
                    snippet = _interp_linear_extrap(time[mask], trace[mask], event_time + t_bins)

                if subtract_baseline:
                    snippet = snippet - np.nanmean(snippet[mask_pre])
                data[event_index, :] = snippet

            with np.errstate(invalid="ignore"):
                baseline_per_event = np.nanstd(data[:, mask_pre], axis=1, ddof=1)
                baseline_std = float(np.nanmedian(baseline_per_event))
            snippets["baseline_std"][field] = baseline_std

            if zscore:
                data = data / baseline_std
                snippets["unit"][field] = "zscore"

            snippets["data"][field] = data

    return snippets
