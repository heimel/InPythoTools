"""Compute and optionally plot NoviTrack ethograms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inpythotools.logmsg import logmsg
from .nt_load_parameters import nt_load_parameters
from .nt_show_markers import nt_show_markers


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


def _record_label(record: Any) -> str:
    return str(_get(record, "sessionid", _get(record, "subject", "record")))


def nt_get_ethogram(
    record: Any,
    show: bool = True,
    params: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, plt.Axes | None]:
    """Return ``ethogram, t, motifs, ax`` for a record's behavior markers."""
    measures = _get(record, "measures", {})
    markers = _get(measures, "markers", None)
    if markers is None or len(markers) == 0:
        logmsg(f"No markers found in record {_record_label(record)}")
        return np.array([]), np.array([]), pd.DataFrame(), None

    if params is None:
        params = nt_load_parameters(record)

    marker_definitions = _marker_table(params)
    if marker_definitions.empty or "behavior" not in marker_definitions:
        return np.array([]), np.array([]), pd.DataFrame(), None

    motifs = marker_definitions[marker_definitions["behavior"].astype(bool)].reset_index(drop=True).copy()
    if motifs.empty:
        return np.array([]), np.array([]), motifs, None

    motif_list = motifs["marker"].astype(str).tolist()
    dt = 0.1
    min_time = float(_get(measures, "min_time", np.floor(min(_get(m, "time") for m in markers) / 60) * 60))
    max_time = float(_get(measures, "max_time", np.ceil(max(_get(m, "time") for m in markers) / 60) * 60))
    n_samples = int(np.ceil((max_time - min_time) / dt))
    ethogram = np.zeros((n_samples, len(motifs)))

    current_motif: int | None = None
    start_index: int | None = None
    for marker in markers:
        marker_name = str(_get(marker, "marker", ""))
        if marker_name and marker_name[0] in motif_list:
            if current_motif is not None and start_index is not None:
                stop_index = min(int(np.ceil((float(_get(marker, "time")) - min_time + 0.0001) / dt)), n_samples)
                ethogram[start_index:stop_index, current_motif] = current_motif + 1
            current_motif = motif_list.index(marker_name[0])
            start_index = max(int(np.ceil((float(_get(marker, "time")) - min_time + 0.0001) / dt)) - 1, 0)

    if current_motif is not None and start_index is not None:
        ethogram[start_index:, current_motif] = current_motif + 1

    t = (np.arange(n_samples) + 0.5) * dt + min_time
    durations = np.count_nonzero(ethogram, axis=0) * dt
    motifs["total_duration"] = durations
    motifs["n"] = [int(np.sum(np.diff(ethogram[:, i] > 0) > 0)) for i in range(len(motifs))]
    motifs["mean_duration"] = motifs.apply(
        lambda row: row["total_duration"] / row["n"] if row["n"] else np.nan,
        axis=1,
    )

    ax = None
    if show and np.any(ethogram):
        fig, ax = plt.subplots(figsize=(11, 3), num="Ethogram")
        fig.set_label("ethogram")
        ax.imshow(
            ethogram.T,
            aspect="auto",
            interpolation="nearest",
            extent=[t[0], t[-1], len(motifs) + 0.5, 0.5],
        )
        ax.set_yticks(np.arange(1, len(motifs) + 1))
        ax.set_yticklabels([str(value).capitalize() for value in motifs["description"]])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Ethogram - {_record_label(record)}")
        marker_params = dict(params)
        marker_params["nt_show_behavior_markers"] = False
        nt_show_markers(markers, ax, marker_params)

    return ethogram, t, motifs, ax


__all__ = ["nt_get_ethogram"]
