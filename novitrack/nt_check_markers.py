"""Check NoviTrack marker start/stop consistency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from inpythotools.logmsg import logmsg


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _record_label(record: Any) -> str:
    return str(_get(record, "sessionid", _get(record, "subject", "record")))


def nt_check_markers(record: Any, params: Any | None = None, verbose: bool = True) -> bool:
    """Return True when marker starts/stops are self-consistent."""
    if params is None:
        from .nt_load_parameters import nt_load_parameters

        params = nt_load_parameters(record)

    measures = _get(record, "measures", {})
    markers = _get(measures, "markers", None)
    if markers is None or len(markers) == 0:
        return True

    stimulus_present = False
    stim_markers = set(str(marker) for marker in _get(params, "nt_stim_markers", []))
    stop_marker = str(_get(params, "nt_stop_marker", "t"))
    msg = ""

    for marker in markers:
        marker_name = str(_get(marker, "marker", ""))
        marker_time = float(_get(marker, "time", np.nan))
        if marker_name == stop_marker:
            if not stimulus_present:
                msg = f"Stimulus stopped before starting at {marker_time:.2g} s"
                break
            stimulus_present = False
        elif marker_name in stim_markers:
            if stimulus_present:
                msg = f"Stimulus started twice at {marker_time:.2g} s"
                break
            stimulus_present = True

    if msg:
        if verbose:
            logmsg(f"{msg} in {_record_label(record)}")
        return False
    return True


__all__ = ["nt_check_markers"]
