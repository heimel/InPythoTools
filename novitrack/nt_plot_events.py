"""Plot NoviTrack per-event analysis results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from .nt_get_events import nt_get_events


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
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return len(value) == 0
    except TypeError:
        return False


def _marker_table(params: Any) -> pd.DataFrame:
    markers = _get(params, "markers", pd.DataFrame())
    if isinstance(markers, pd.DataFrame):
        return markers
    return pd.DataFrame(markers)


def _event_description(params: Any, event_type: str) -> str:
    markers = _marker_table(params)
    if markers.empty or "marker" not in markers:
        return event_type
    match = markers[markers["marker"].astype(str) == event_type[0]]
    if match.empty:
        return event_type
    return f"{match.iloc[0].get('description', event_type)} {event_type}"


def nt_plot_events(
    record: Mapping[str, Any],
    params: Any,
    snippets: Mapping[str, Any] | None = None,
) -> list[plt.Figure]:
    """Plot per-event heatmaps and mean traces for all event observables."""
    measures = _get(record, "measures", {})
    event_measures = _get(measures, "event", {})
    if _is_empty(event_measures):
        return []

    events = nt_get_events(measures, params)
    t = _as_array(_get(measures, "snippets_tbins"))
    figures: list[plt.Figure] = []
    snippets_data = _get(snippets, "data", {}) if snippets else {}
    snippet_units = _get(snippets, "unit", {}) if snippets else {}

    for event_type, event in event_measures.items():
        observables = list(event.keys())
        n_cols = min(3, len(observables))
        n_rows = int(np.ceil(len(observables) / n_cols))
        fig = plt.figure(figsize=(4.5 * n_cols, 4.2 * n_rows), num=str(event_type), constrained_layout=True)
        fig.set_label(f"event_{event_type}")
        fig.suptitle(_event_description(params, str(event_type)))
        event_indices = events.index[events["event"] == str(event_type)].to_numpy()
        grid = GridSpec(n_rows, n_cols, figure=fig)

        for index, observable in enumerate(observables):
            row = index // n_cols
            col = index % n_cols
            subgrid = grid[row, col].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.05)
            heat_ax = fig.add_subplot(subgrid[0])
            trace_ax = fig.add_subplot(subgrid[1], sharex=heat_ax)
            result = event[observable]
            if observable in snippets_data and event_indices.size:
                heat_ax.imshow(
                    np.asarray(snippets_data[observable])[event_indices, :],
                    aspect="auto",
                    interpolation="nearest",
                    extent=[t[0], t[-1], event_indices.size + 0.5, 0.5],
                )
                heat_ax.set_ylim(event_indices.size + 0.5, 0.5)
                heat_ax.set_yticks(np.unique([1, event_indices.size]))
            else:
                heat_ax.text(0.5, 0.5, "No snippets", ha="center", va="center", transform=heat_ax.transAxes)
                heat_ax.set_yticks([])
            heat_ax.set_title(f"{observable}, n = {result.get('n', '')}")
            heat_ax.set_ylabel("Trial")
            heat_ax.tick_params(axis="x", labelbottom=False)
            heat_ax.spines[["top", "right"]].set_visible(False)

            y = _as_array(result["snippet_mean"])
            sem = _as_array(result.get("snippet_sem", np.zeros_like(y)))
            trace_ax.plot(t, y, color="black", linewidth=1.5)
            trace_ax.fill_between(t, y - 1.97 * sem, y + 1.97 * sem, color="black", alpha=0.18, linewidth=0)
            trace_ax.axhline(0, color="0.4", linewidth=0.8)
            trace_ax.axvline(0, color="0.4", linewidth=0.8)
            trace_ax.set_xlabel("Time (s)")
            trace_ax.set_ylabel(snippet_units.get(observable, result.get("unit", "")))
            trace_ax.spines[["top", "right"]].set_visible(False)

        for index in range(len(observables), n_rows * n_cols):
            blank_ax = fig.add_subplot(grid[index // n_cols, index % n_cols])
            blank_ax.axis("off")
        figures.append(fig)
    return figures


__all__ = ["nt_plot_events"]
