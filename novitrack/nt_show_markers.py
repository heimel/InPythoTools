"""Draw NoviTrack marker annotations on matplotlib axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.axes
import numpy as np
import pandas as pd


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _marker_table(params: Any) -> pd.DataFrame:
    markers = _get(params, "markers", pd.DataFrame())
    if isinstance(markers, pd.DataFrame):
        return markers
    return pd.DataFrame(markers)


def _marker_definition(params: Any, marker: str) -> Mapping[str, Any] | None:
    marker_table = _marker_table(params)
    if marker_table.empty or "marker" not in marker_table:
        return None

    match = marker_table[marker_table["marker"].astype(str) == marker[0]]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _as_color(value: Any) -> Any:
    if value is None:
        return "black"
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(float(v) for v in value)
    return value


def nt_show_markers(
    markers: Any,
    ax: matplotlib.axes.Axes,
    params: Any,
    bounds: Sequence[float] | None = None,
    yl: Sequence[float] | None = None,
) -> None:
    """Show marker times on a matplotlib timeline.

    This mirrors MATLAB ``nt_show_markers.m``: marker colors come from
    ``params.markers``, and behavior markers can be hidden with
    ``params.nt_show_behavior_markers``.
    """
    if not bool(_get(params, "nt_show_markers", True)):
        return
    if markers is None or len(markers) == 0:
        return

    if bounds is None:
        bounds = ax.get_xlim()
    if yl is None:
        yl = ax.get_ylim()

    for artist in list(ax.lines):
        if artist.get_gid() == "Marker":
            artist.remove()

    min_time = float(bounds[0])
    max_time = float(bounds[1])
    y = [float(yl[0]), float(yl[1])]

    for marker in markers:
        marker_name = str(_get(marker, "marker", ""))
        if not marker_name:
            continue
        marker_time = float(_get(marker, "time", np.nan))
        if not np.isfinite(marker_time) or marker_time < min_time or marker_time > max_time:
            continue

        marker_def = _marker_definition(params, marker_name)
        if marker_def is None:
            color = "black"
        else:
            if bool(_get(marker_def, "behavior", False)) and not bool(_get(params, "nt_show_behavior_markers", True)):
                continue
            color = _as_color(_get(marker_def, "color", "black"))

        (line,) = ax.plot(
            [marker_time, marker_time],
            y,
            color=color,
            linewidth=1,
            solid_capstyle="butt",
        )
        line.set_gid("Marker")


__all__ = ["nt_show_markers"]
