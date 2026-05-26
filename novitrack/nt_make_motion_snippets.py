"""Cut peri-event snippets from NoviTrack motion traces."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from inpythotools.logmsg import logmsg
from .nt_get_events import nt_get_events
from .nt_make_photometry_snippets import _interp_linear_extrap


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return len(value) == 0
    try:
        return len(value) == 0
    except TypeError:
        return False


def nt_make_motion_snippets(
    nt_data: Any,
    measures: Any,
    snippets: Mapping[str, Any] | None,
    params: Any,
) -> dict[str, Any]:
    """Cut motion snippets around all events.

    This mirrors ``nt_make_motion_snippets.m`` and appends motion observables
    to an existing snippets dictionary when one is supplied.
    """
    if snippets is None or _is_empty(snippets):
        out: dict[str, Any] = {"data": {}, "unit": {}}
    else:
        out = deepcopy(dict(snippets))
        out.setdefault("data", {})
        out.setdefault("unit", {})

    markers = _get(measures, "markers", None)
    if markers is None or len(markers) == 0 or _is_empty(nt_data):
        return out

    events = nt_get_events(measures, params)
    t_bins = _as_array(_get(measures, "snippets_tbins"))
    time = _as_array(_get(nt_data, "Time"))

    pretime = float(_get(params, "nt_pretime", 10))
    posttime = float(_get(params, "nt_posttime", 20))
    bin_width = float(_get(params, "nt_photometry_bin_width", 0.1))

    for observable in ("Speed", "Abs_angular_velocity", "Distance_to_center"):
        values = _as_array(_get(nt_data, observable, np.array([])))
        if values.size == 0 or np.all(np.isnan(values)):
            continue

        data = np.full((len(events), t_bins.size), np.nan)
        for event_index, event in events.iterrows():
            event_time = float(event["time"])
            mask = (time > event_time - pretime - bin_width) & (
                time < event_time + posttime + bin_width
            )
            if np.any(mask):
                data[event_index, :] = _interp_linear_extrap(time[mask], values[mask], event_time + t_bins)
            else:
                logmsg(f"No samples for event at {event_time}")

        out["data"][observable] = data
        out["unit"][observable] = "a.u."

    return out
