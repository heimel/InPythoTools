"""Plot NoviTrack analysis results with matplotlib."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logmsg import logmsg
from nt_get_events import nt_get_events


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


def _save_or_show(figures: list[plt.Figure], output_dir: str | Path | None, show: bool) -> list[plt.Figure]:
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for index, figure in enumerate(figures, start=1):
            title = figure.get_label() or f"figure_{index:02d}"
            safe_title = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title).strip("_")
            figure.savefig(folder / f"{index:02d}_{safe_title}.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return figures


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


def plot_ethogram(record: Mapping[str, Any], params: Any) -> plt.Figure | None:
    """Plot a simple ethogram derived from behavior markers."""
    measures = _get(record, "measures", {})
    markers = _get(measures, "markers", [])
    if not markers:
        logmsg(f"No markers found in record {_record_label(record)}")
        return None

    marker_definitions = _marker_table(params)
    if marker_definitions.empty or "behavior" not in marker_definitions:
        return None

    motifs = marker_definitions[marker_definitions["behavior"].astype(bool)].reset_index(drop=True)
    if motifs.empty:
        return None

    motif_list = motifs["marker"].astype(str).tolist()
    dt = 0.1
    min_time = float(_get(measures, "min_time", np.floor(min(_get(m, "time") for m in markers) / 60) * 60))
    max_time = float(_get(measures, "max_time", np.ceil(max(_get(m, "time") for m in markers) / 60) * 60))
    n_samples = int(np.ceil((max_time - min_time) / dt))
    ethogram = np.zeros((n_samples, len(motifs)))

    current_motif: int | None = None
    start_index: int | None = None
    for marker in markers:
        marker_name = str(_get(marker, "marker"))
        if marker_name and marker_name[0] in motif_list:
            if current_motif is not None and start_index is not None:
                stop_index = min(int(np.ceil((float(_get(marker, "time")) - min_time + 0.0001) / dt)), n_samples)
                ethogram[start_index:stop_index, current_motif] = current_motif + 1
            current_motif = motif_list.index(marker_name[0])
            start_index = max(int(np.ceil((float(_get(marker, "time")) - min_time + 0.0001) / dt)) - 1, 0)

    if current_motif is not None and start_index is not None:
        ethogram[start_index:, current_motif] = current_motif + 1

    if not np.any(ethogram):
        return None

    t = (np.arange(n_samples) + 0.5) * dt + min_time
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
    return fig


def plot_session_summary(record: Mapping[str, Any]) -> plt.Figure | None:
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


def plot_maps(record: Mapping[str, Any]) -> plt.Figure | None:
    measures = _get(record, "measures", {})
    maps = _get(measures, "maps", {})
    if not maps:
        return None

    panels = [("Presence", maps.get("counts"))]
    for channel in _get(measures, "channels", []):
        channel_name = _get(channel, "channel")
        for light in _get(channel, "lights", []):
            light_type = _get(light, "type")
            value = _get(_get(maps, channel_name, {}), light_type, None)
            if value is not None:
                panels.append((f"{channel_name} - {light_type}", value))

    n_cols = min(3, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False, num="Maps")
    fig.set_label("maps")
    for ax, (title, data) in zip(axes.ravel(), panels):
        image = ax.imshow(np.asarray(data).T, origin="lower", aspect="equal")
        ax.invert_xaxis()
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    return fig


def plot_event_results(
    record: Mapping[str, Any],
    params: Any,
    snippets: Mapping[str, Any] | None = None,
) -> list[plt.Figure]:
    measures = _get(record, "measures", {})
    event_measures = _get(measures, "event", {})
    if not event_measures:
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False, num=str(event_type))
        fig.set_label(f"event_{event_type}")
        fig.suptitle(_event_description(params, str(event_type)))
        event_indices = events.index[events["event"] == str(event_type)].to_numpy()

        for ax, observable in zip(axes.ravel(), observables):
            result = event[observable]
            if observable in snippets_data and event_indices.size:
                inset = ax.inset_axes([0.0, 0.58, 1.0, 0.36])
                inset.imshow(
                    np.asarray(snippets_data[observable])[event_indices, :],
                    aspect="auto",
                    interpolation="nearest",
                    extent=[t[0], t[-1], event_indices.size + 0.5, 0.5],
                )
                inset.set_ylabel("Trial")
                inset.set_xticks([])

            y = _as_array(result["snippet_mean"])
            sem = _as_array(result.get("snippet_sem", np.zeros_like(y)))
            ax.plot(t, y, color="black", linewidth=1.5)
            ax.fill_between(t, y - 1.97 * sem, y + 1.97 * sem, color="black", alpha=0.18, linewidth=0)
            ax.axhline(0, color="0.4", linewidth=0.8)
            ax.axvline(0, color="0.4", linewidth=0.8)
            ax.set_title(f"{observable}, n = {result.get('n', '')}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(snippet_units.get(observable, result.get("unit", "")))
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axes.ravel()[len(observables) :]:
            ax.axis("off")
        figures.append(fig)
    return figures


def plot_photometry_results(
    record: Mapping[str, Any],
    photometry: Mapping[str, Any],
    snippets: Mapping[str, Any] | None,
    params: Any,
) -> list[plt.Figure]:
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
        for _, event in events.iterrows():
            ax.axvline(float(event["time"]), color="0.75", linewidth=0.4)
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

    return figures


def results_nttestrecord(
    record: Mapping[str, Any],
    params: Any,
    *,
    photometry: Mapping[str, Any] | None = None,
    snippets: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Create result figures for one analyzed NoviTrack record."""
    measures = _get(record, "measures", {})
    if not measures:
        logmsg("No measures. Run analysis first.")
        return []

    figures: list[plt.Figure] = []
    for figure in (
        plot_ethogram(record, params),
        plot_maps(record),
        plot_session_summary(record),
    ):
        if figure is not None:
            figures.append(figure)

    if photometry is not None:
        figures.extend(plot_photometry_results(record, photometry, snippets, params))

    figures.extend(plot_event_results(record, params, snippets))

    logmsg("Generated result figures.")
    return _save_or_show(figures, output_dir, show)
