"""Create NoviTrack event tables from marker annotations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def nt_get_events(measures: Any, params: Any | None = None) -> pd.DataFrame:
    """Create an events DataFrame from ``measures["markers"]``.

    This mirrors ``nt_get_events.m``. Events are derived on demand so saved
    databases do not need to store MATLAB table objects.
    """
    markers = _get(measures, "markers", None)
    if markers is None or len(markers) == 0:
        return pd.DataFrame({"time": pd.Series(dtype=float), "event": pd.Series(dtype=str)})

    events = pd.DataFrame(
        {
            "time": [float(_get(marker, "time")) for marker in markers],
            "event": [str(_get(marker, "marker")) for marker in markers],
        }
    )
    events["event"] = events["event"].replace({"0": "opto_off", "1": "opto_on"})

    pretime = float(_get(params, "nt_pretime", 10))

    if bool(_get(params, "use_clean_baseline", False)):
        index = 0
        while index < len(events):
            row = events.iloc[index]
            remove = (
                (events["time"] > row["time"])
                & (events["time"] < row["time"] + pretime)
                & (events["event"] == row["event"])
            )
            events = events.loc[~remove].reset_index(drop=True)
            index += 1

    if bool(_get(params, "use_ultraclean_baseline", False)):
        index = 0
        while index < len(events):
            row = events.iloc[index]
            remove = (events["time"] > row["time"]) & (events["time"] < row["time"] + pretime)
            events = events.loc[~remove].reset_index(drop=True)
            index += 1

    return events
