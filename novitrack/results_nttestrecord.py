"""Top-level NoviTrack result plotting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from scipy.io import loadmat

from inpythotools.mat_database import _convert_mat_value
from inpythotools.logmsg import logmsg
from .nt_analyse_photometry import nt_analyse_photometry
from .nt_get_ethogram import nt_get_ethogram
from .nt_load_parameters import nt_load_parameters
from .nt_load_tracking_data import nt_load_tracking_data
from .nt_photometry_folder import nt_photometry_folder
from .nt_plot_events import nt_plot_events
from .nt_plot_maps import nt_plot_maps
from .nt_plot_photometry import nt_plot_photometry
from .nt_plot_session_summary import nt_plot_session_summary
from .nt_session_path import nt_session_path


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _load_mat_field(filename: Path, field_name: str) -> Any:
    mat = loadmat(str(filename), squeeze_me=True, struct_as_record=False)
    if field_name not in mat:
        return None
    return _convert_mat_value(mat[field_name])


def _load_snippets_from_disk(record: Mapping[str, Any], params: Any) -> Mapping[str, Any] | None:
    try:
        folder, exists = nt_session_path(record, params)
    except OSError as exc:
        logmsg(f"Could not resolve session path for snippets: {exc}")
        return None
    if not exists:
        return None

    filename = folder / "nt_snippets.mat"
    if not filename.exists():
        return None
    snippets = _load_mat_field(filename, "snippets")
    return snippets if isinstance(snippets, Mapping) else None


def _resolve_snippets(
    record: Mapping[str, Any],
    params: Any,
    snippets: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if snippets is not None:
        return snippets
    record_snippets = _get(record, "snippets", None)
    if isinstance(record_snippets, Mapping) and record_snippets:
        return record_snippets
    return _load_snippets_from_disk(record, params)


def _load_photometry_from_disk(record: Mapping[str, Any], params: Any) -> Mapping[str, Any] | None:
    folder, found = nt_photometry_folder(record, params)
    if not found or folder is None:
        return None
    filename = folder / "nt_photometry.mat"
    if not filename.exists():
        return None
    photometry = _load_mat_field(filename, "photometry")
    return photometry if isinstance(photometry, Mapping) else None


def _resolve_photometry(
    record: Mapping[str, Any],
    params: Any,
    photometry: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    out_record = dict(record)
    if photometry is not None:
        return photometry, out_record

    photometry = _load_photometry_from_disk(record, params)
    if photometry is not None:
        return photometry, out_record

    nt_data, _ = nt_load_tracking_data(record, params)
    analysed_record, photometry_dict, _ = nt_analyse_photometry(record, nt_data, params)
    if photometry_dict:
        out_record = dict(analysed_record)
        return photometry_dict, out_record
    return None, out_record


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


def results_nttestrecord(
    record: Mapping[str, Any],
    params: Any | None = None,
    *,
    photometry: Mapping[str, Any] | None = None,
    snippets: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Create result figures for one analyzed NoviTrack record."""
    if params is None:
        params = nt_load_parameters(record)

    snippets = _resolve_snippets(record, params, snippets)
    photometry, record = _resolve_photometry(record, params, photometry)
    measures = _get(record, "measures", {})
    if not measures:
        logmsg("No measures. Run analysis first.")
        return []

    figures: list[plt.Figure] = []
    _ethogram, _t, _motifs, ethogram_ax = nt_get_ethogram(record, show=True, params=params)
    if ethogram_ax is not None:
        figures.append(ethogram_ax.figure)

    for figure in (
        nt_plot_maps(record),
        nt_plot_session_summary(record),
    ):
        if figure is not None:
            figures.append(figure)

    if photometry is not None:
        figures.extend(nt_plot_photometry(record, photometry, snippets, params))

    figures.extend(nt_plot_events(record, params, snippets))

    logmsg("Generated result figures.")
    return _save_or_show(figures, output_dir, show)


__all__ = ["results_nttestrecord"]
