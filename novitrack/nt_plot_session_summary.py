"""Plot NoviTrack object-independent session summary measures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _record_label(record: Mapping[str, Any]) -> str:
    return str(_get(record, "sessionid", _get(record, "subject", "record")))


def nt_plot_session_summary(record: Mapping[str, Any]) -> plt.Figure | None:
    """Plot simple session-level running/backward-motion summary bars."""
    measures = _get(record, "measures", {})
    required = (
        "session_fraction_running_forward",
        "session_start_running_forward_per_min",
        "session_fraction_moving_backward",
        "session_start_moving_backward_per_min",
    )
    if not all(key in measures for key in required):
        return None

    values = [
        float(measures["session_fraction_running_forward"]) * 100,
        float(measures["session_start_running_forward_per_min"]),
        float(measures["session_fraction_moving_backward"]) * 100,
        float(measures["session_start_moving_backward_per_min"]),
    ]
    labels = [
        "Running forward\n(% time)",
        "Running forward\n(#/min)",
        "Moving backward\n(% time)",
        "Moving backward\n(#/min)",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), num="Session summary")
    fig.set_label("session_summary")
    fig.suptitle(_record_label(record))
    ylims = [(0, 40), (0, 70), (0, 5), (0, 20)]
    for ax, value, label, ylim in zip(axes, values, labels, ylims):
        ax.bar([0], [value], color="0.25", width=0.5)
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


__all__ = ["nt_plot_session_summary"]
