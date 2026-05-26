"""Plot NoviTrack photometry analysis results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .nt_get_events import nt_get_events
from .nt_show_markers import nt_show_markers


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _record_label(record: Mapping[str, Any]) -> str:
    return str(_get(record, "sessionid", _get(record, "subject", "record")))


def _zscore(values: np.ndarray) -> np.ndarray:
    values = _as_array(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return np.full_like(values, np.nan, dtype=float)
    return (values - np.nanmean(values)) / std


def _channel_label(channel: Mapping[str, Any]) -> str:
    location = str(_get(channel, "location", "") or "")
    sensor = str(_get(channel, "green_sensor", "") or "")
    label = f"{location} - {sensor}".strip(" -")
    return label or str(_get(channel, "channel", "Channel"))


def nt_plot_channel_correlation(
    record: Mapping[str, Any],
    photometry: Mapping[str, Any],
    measures: Mapping[str, Any],
) -> plt.Figure | None:
    """Plot pairwise green-channel photometry correlations."""
    channels = list(_get(measures, "channels", []))
    if len(channels) <= 1:
        return None

    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for i, channel1 in enumerate(channels):
        for channel2 in channels[:i]:
            channel1_name = _get(channel1, "channel")
            channel2_name = _get(channel2, "channel")
            if (
                channel1_name in photometry
                and channel2_name in photometry
                and "green" in photometry[channel1_name]
                and "green" in photometry[channel2_name]
            ):
                pairs.append((channel1, channel2))

    if not pairs:
        return None

    n_side = max(1, len(channels) - 1)
    fig, axes = plt.subplots(n_side, n_side, figsize=(4 * n_side, 4 * n_side), squeeze=False, num="Channel correlation")
    fig.set_label("channel_correlation")

    for ax in axes.ravel():
        ax.axis("off")

    phi = np.linspace(0, 2 * np.pi, 100)
    for count, (channel1, channel2) in enumerate(pairs):
        ax = axes.ravel()[count]
        ax.axis("on")

        channel1_name = _get(channel1, "channel")
        channel2_name = _get(channel2, "channel")
        x = _zscore(photometry[channel1_name]["green"]["signal"])
        y = _zscore(photometry[channel2_name]["green"]["signal"])
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        valid = np.isfinite(x) & np.isfinite(y)

        ax.plot(x[valid], y[valid], ".", markersize=2, color="black", alpha=0.45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.plot(2 * np.sin(phi), 2 * np.cos(phi), color="black", linewidth=0.8)

        if np.any(valid):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xy_min = min(xmin, ymin)
            xy_max = max(xmax, ymax)
            ax.plot([xy_min, xy_max], [xy_min, xy_max], color="0.55", linewidth=0.8)
            ax.set_xlim(xy_min, xy_max)
            ax.set_ylim(xy_min, xy_max)
            r = float(np.corrcoef(x[valid], y[valid])[0, 1]) if np.sum(valid) > 1 else np.nan
        else:
            r = np.nan

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{_channel_label(channel1)} (z)")
        ax.set_ylabel(f"{_channel_label(channel2)} (z)")
        ax.set_title(f"{_record_label(record)} - r = {r:.3g}")
        ax.spines[["top", "right"]].set_visible(False)

    return fig


def nt_plot_photometry(
    record: Mapping[str, Any],
    photometry: Mapping[str, Any],
    snippets: Mapping[str, Any] | None,
    params: Any,
) -> list[plt.Figure]:
    """Plot full photometry traces, snippets, and channel correlations."""
    measures = _get(record, "measures", {})
    if not photometry or "channels" not in measures:
        return []

    events = nt_get_events(measures, params)
    figures: list[plt.Figure] = []
    period = _as_array(_get(measures, "period_of_interest", [-np.inf, np.inf]))
    t_bins = _as_array(_get(measures, "snippets_tbins", []))

    for channel in measures["channels"]:
        channel_name = _get(channel, "channel")
        lights = _get(channel, "lights", [])
        heat_lights = [
            light for light in lights if not (_get(measures, "photometry_isosbestic_correction", False) and _get(light, "type") == "isosbestic")
        ]
        n_rows = 1 + len(heat_lights)
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3 + 2.5 * len(heat_lights)), squeeze=False, num=channel_name)
        fig.set_label(f"photometry_{channel_name}")

        ax = axes[0, 0]
        for light in lights:
            light_type = _get(light, "type")
            time = _as_array(photometry[channel_name][light_type]["time"])
            signal = _as_array(photometry[channel_name][light_type]["signal"])
            mask = (time > period[0]) & (time < period[1])
            ax.plot(time[mask], signal[mask], linewidth=0.8, label=light_type)
        ax.axvline(period[0], color="black", linewidth=0.8)
        ax.axvline(period[1], color="black", linewidth=0.8)
        nt_show_markers(_get(measures, "markers", []), ax, params, bounds=period)
        ax.set_ylabel("Fluorescence (a.u.)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
        ax.set_title(f"{_record_label(record)} - {channel_name}")

        if snippets and heat_lights and len(events):
            sorted_indices = events.sort_values("event").index.to_numpy()
            for row, light in enumerate(heat_lights, start=1):
                light_type = _get(light, "type")
                field = f"{channel_name}_{light_type}"
                if field not in snippets.get("data", {}):
                    continue
                heat_ax = axes[row, 0]
                image = heat_ax.imshow(
                    np.asarray(snippets["data"][field])[sorted_indices, :],
                    aspect="auto",
                    interpolation="nearest",
                    extent=[t_bins[0], t_bins[-1], len(sorted_indices) + 0.5, 0.5],
                )
                heat_ax.set_title(light_type)
                heat_ax.set_xlabel("Time (s)")
                heat_ax.set_ylabel("Event (sorted by type)")
                fig.colorbar(image, ax=heat_ax, fraction=0.025, pad=0.02)
        figures.append(fig)

    correlation_figure = nt_plot_channel_correlation(record, photometry, measures)
    if correlation_figure is not None:
        figures.append(correlation_figure)

    return figures


__all__ = ["nt_plot_photometry", "nt_plot_channel_correlation"]
