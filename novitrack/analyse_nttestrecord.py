"""Top-level NoviTrack session analysis.

This is a first Python port of the non-GUI, already-tracked path through
``analyse_nttestrecord.m``. Neurotar object analysis and tracking recompute
branches are intentionally left for later.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from inpythotools.logmsg import logmsg
from .nt_add_surgery_info import nt_add_surgery_info
from .nt_analyse_photometry import nt_analyse_photometry
from .nt_check_markers import nt_check_markers
from .nt_compute_event_measures import nt_compute_event_measures
from .nt_compute_locations import nt_compute_locations
from .nt_load_parameters import nt_load_parameters
from .nt_load_tracking_data import nt_load_tracking_data
from .nt_make_motion_snippets import nt_make_motion_snippets
from .nt_make_photometry_snippets import nt_make_photometry_snippets
from .nt_session_path import nt_session_path


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


def _session_measures(measures: dict[str, Any], nt_data: Mapping[str, Any], params: Any) -> dict[str, Any]:
    time = _as_array(_get(nt_data, "Time"))
    mask = time > 0
    n_samples = int(np.sum(mask))
    if n_samples == 0:
        return measures

    speed = _as_array(_get(nt_data, "Speed"))
    forward_speed = _as_array(_get(nt_data, "Forward_speed"))
    angular_velocity = _as_array(_get(nt_data, "Angular_velocity"))

    def nanmean(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    def nanstd(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        return float(np.std(values, ddof=1)) if values.size > 1 else np.nan

    def nanmax(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        return float(np.max(values)) if values.size else np.nan

    measures["session_speed_mean"] = nanmean(speed[mask])
    measures["session_speed_std"] = nanstd(speed[mask])
    measures["session_speed_max"] = nanmax(speed[mask])
    measures["session_forward_speed_mean"] = nanmean(forward_speed[mask])
    measures["session_forward_speed_std"] = nanstd(forward_speed[mask])
    measures["session_forward_speed_max"] = nanmax(forward_speed[mask])
    measures["session_angular_velocity_mean"] = nanmean(angular_velocity[mask])
    measures["session_angular_velocity_std"] = nanstd(angular_velocity[mask])
    measures["session_angular_velocity_max"] = nanmax(angular_velocity[mask])

    min_approach_speed = float(_get(params, "nt_min_approach_speed"))
    min_retreat_speed = float(_get(params, "nt_min_retreat_speed"))
    forward_masked = forward_speed[mask]

    measures["session_fraction_running_forward"] = float(np.sum(forward_masked > min_approach_speed) / n_samples)
    measures["session_count_start_running_forward"] = int(np.sum(np.diff(forward_masked > min_approach_speed) > 0))
    measures["session_start_running_forward_per_min"] = float(
        measures["session_count_start_running_forward"] / time[-1] * 60
    )
    measures["session_fraction_moving_backward"] = float(np.sum(forward_masked < min_retreat_speed) / n_samples)
    measures["session_count_start_moving_backward"] = int(np.sum(np.diff(forward_masked < min_retreat_speed) > 0))
    measures["session_start_moving_backward_per_min"] = float(
        measures["session_count_start_moving_backward"] / time[-1] * 60
    )

    return measures


def analyse_nttestrecord(
    record: Mapping[str, Any],
    *,
    params: Any | None = None,
    yaml_file: str | Path | None = None,
    session_path: str | Path | None = None,
    photometry_folder: str | Path | None = None,
    save_snippets: bool | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Analyze one NoviTrack record and return an updated record dictionary."""
    out = deepcopy(dict(record))
    out.setdefault("measures", {})

    if params is None:
        params = nt_load_parameters(out, yaml_file=yaml_file)

    if save_snippets is None:
        save_snippets = bool(_get(params, "nt_save_snippets", True))

    if _get(params, "nt_seed", None):
        np.random.seed(int(_get(params, "nt_seed")))

    out = nt_add_surgery_info(out, params)

    nt_data, trigger_times = nt_load_tracking_data(out, params, recompute=False, session_path=session_path)
    if not nt_data:
        logmsg(f"Could not find any position data for {_record_label(out)}")

    if verbose:
        logmsg(f"Analyzing {_record_label(out)}")

    measures = dict(_get(out, "measures", {}))
    measures["event"] = {}
    measures.pop("events", None)
    if trigger_times.size:
        measures["trigger_times"] = trigger_times

    if "markers" not in measures or len(measures.get("markers", [])) == 0:
        measures["markers"] = []

    out["measures"] = measures
    if not nt_check_markers(out, params, verbose=verbose):
        return out

    bin_width = float(_get(params, "nt_photometry_bin_width"))
    pretime = float(_get(params, "nt_pretime"))
    posttime = float(_get(params, "nt_posttime"))
    measures["snippets_tbins"] = np.arange(
        -pretime + bin_width / 2,
        posttime - bin_width / 2 + bin_width / 10,
        bin_width,
    )
    out["measures"] = measures

    out, photometry, _ = nt_analyse_photometry(
        out, nt_data, params, photometry_folder=photometry_folder
    )
    measures = dict(out["measures"])

    snippets: dict[str, Any] = {}
    if photometry:
        snippets = nt_make_photometry_snippets(photometry, measures, params)

    snippets = nt_make_motion_snippets(nt_data, measures, snippets, params)
    if save_snippets:
        if session_path is None:
            folder, exists = nt_session_path(out, params)
        else:
            folder = Path(session_path)
            exists = folder.is_dir()
        if exists:
            filename = folder / "nt_snippets.mat"
            try:
                savemat(str(filename), {"snippets": snippets}, do_compression=True, long_field_names=True)
            except (OSError, TypeError, ValueError) as exc:
                logmsg(f"Could not save snippets to {filename}: {exc}")

    measures = nt_compute_event_measures(snippets, measures, params)

    if nt_data:
        measures = _session_measures(measures, nt_data, params)
        out["measures"] = measures
        if not bool(_get(params, "neurotar", False)):
            out = nt_compute_locations(out, nt_data, params)
        else:
            logmsg("Compute locations is not yet implemented for neurotar.")
    else:
        out["measures"] = measures

    if bool(_get(params, "neurotar", False)):
        logmsg("Neurotar object analysis and shuffling are not ported yet.")

    # Keep snippets available to callers without storing them inside measures.
    out["snippets"] = snippets
    return out
