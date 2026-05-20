"""
Load Neurotar/NoviTrack analysis parameters from ``nt_default_parameters.yaml``.

Pythonic translation of ``nt_load_parameters.m``.

The function returns an ``AttrDict``: it behaves like a normal dictionary, but
also allows attribute access, e.g. ``params.networkpathbase`` as well as
``params["networkpathbase"]``.

Expected YAML layout
--------------------
The MATLAB version expects sections such as ``nt_behaviors``, ``nt_indices``,
``nt_rates``, ``nt_marker_sets``, ``camera_sets``, ``arenas`` and
``tracking_presets``. This Python version preserves that convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class AttrDict(dict):
    """Dictionary with recursive attribute access.

    This is convenient during a MATLAB-to-Python transition because MATLAB code
    often uses ``params.field`` syntax. You can still use it as an ordinary dict.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def _as_attrdict(value: Any) -> Any:
    """Recursively convert dictionaries to ``AttrDict``."""
    if isinstance(value, Mapping):
        return AttrDict({k: _as_attrdict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_as_attrdict(v) for v in value]
    return value


@dataclass
class Record:
    """Minimal record object used when no record is supplied."""

    setup: str = "neurotar"
    stimulus: str = "none"
    date: str = field(default_factory=lambda: date.today().isoformat())


def _get(record: Any, field_name: str, default: Any = None) -> Any:
    """Get a field from a dict-like or object-like record."""
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _has(record: Any, field_name: str) -> bool:
    if record is None:
        return False
    if isinstance(record, Mapping):
        return field_name in record
    return hasattr(record, field_name)


def _merge_params(base: AttrDict, extra: Mapping[str, Any] | None) -> AttrDict:
    """MATLAB ``catstruct`` equivalent: update top-level fields in ``base``."""
    if extra:
        for key, value in extra.items():
            base[key] = _as_attrdict(value)
    return base


def _date_before(record_date: str | date | datetime | None, cutoff: str) -> bool:
    if record_date is None:
        return False
    if isinstance(record_date, datetime):
        d = record_date.date()
    elif isinstance(record_date, date):
        d = record_date
    else:
        d = datetime.strptime(str(record_date), "%Y-%m-%d").date()
    return d < datetime.strptime(cutoff, "%Y-%m-%d").date()


def _table_from_name_mapping(mapping: Mapping[str, Any], name_column: str) -> pd.DataFrame:
    """Convert YAML entries of the form ``name: [description, color]`` to a table."""
    rows: list[dict[str, Any]] = []
    for name, value in mapping.items():
        rows.append(
            {
                name_column: name,
                "description": value[0] if len(value) > 0 else None,
                "color": value[1] if len(value) > 1 else None,
            }
        )
    return pd.DataFrame(rows)


def _marker_table(marker_set: Mapping[str, Any]) -> pd.DataFrame:
    """Convert one marker set from YAML to a pandas table."""
    rows: list[dict[str, Any]] = []
    for marker_id, value in marker_set.items():
        rows.append(
            {
                "marker_id": marker_id,
                "marker": value[0] if len(value) > 0 else None,
                "description": value[1] if len(value) > 1 else None,
                "color": value[2] if len(value) > 2 else None,
                "behavior": bool(value[3]) if len(value) > 3 else False,
                "linked": bool(value[4]) if len(value) > 4 else False,
            }
        )
    return pd.DataFrame(rows)


def _normalise_vector_fields(params: AttrDict, field_names: tuple[str, ...]) -> None:
    """Convert small vector fields to plain lists for predictable downstream use."""
    for name in field_names:
        if name in params and params[name] is not None:
            value = params[name]
            if not isinstance(value, list):
                params[name] = list(value)


def _choose_marker_set(record: Any) -> str:
    stimulus = str(_get(record, "stimulus", "")).lower()
    condition = str(_get(record, "condition", "")).lower()
    setup = str(_get(record, "setup", "")).lower()

    if stimulus == "firstandnovelobject":
        return "default"
    if stimulus == "looming_stimulus":
        return "looming"

    if condition == "looming_stimulus":
        return "looming"
    if condition == "social_behavior":
        return "social_behavior"

    if setup == "neurotar":
        return "prey_capture"
    if setup == "elevated_plus_maze":
        return "elevated_plus_maze"
    return "default"


def nt_load_parameters(
    record: Any | None = None,
    *,
    yaml_file: str | Path | None = None,
    apply_local_overrides: bool = True,
) -> AttrDict:
    """Load and adapt Neurotar/NoviTrack parameters for one record.

    Parameters
    ----------
    record:
        Dict-like or object-like record with fields such as ``setup``, ``date``,
        ``stimulus``, ``condition`` and optionally ``measures``. If omitted, a
        default Neurotar record for today's date is used.
    yaml_file:
        Path to ``nt_default_parameters.yaml``. By default, the function looks
        next to this Python file.
    apply_local_overrides:
        If true, tries to import ``processparams_local.processparams_local`` and
        apply it as the last step, mimicking the MATLAB function.

    Returns
    -------
    AttrDict
        Parameter dictionary with additional convenient tables:
        ``nt_behaviors``, ``nt_indices``, ``nt_rates`` and ``markers`` are
        pandas DataFrames.
    """
    if record is None:
        record = Record()

    yaml_path = Path(yaml_file) if yaml_file is not None else Path(__file__).resolve().parent / "nt_default_parameters.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cannot find config file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fid:
        raw = yaml.safe_load(fid) or {}
    params = _as_attrdict(raw)

    # Convert MATLAB struct arrays to DataFrames. Keep the original names so the
    # calling code does not have to change much.
    if "nt_behaviors" in params:
        params.nt_behaviors = _table_from_name_mapping(params.nt_behaviors, "behavior")
    if "nt_indices" in params:
        params.nt_indices = _table_from_name_mapping(params.nt_indices, "index")
    if "nt_rates" in params:
        params.nt_rates = _table_from_name_mapping(params.nt_rates, "rate")

    if "nt_marker_sets" in params:
        params.nt_marker_sets = AttrDict(
            {name: _marker_table(marker_set) for name, marker_set in params.nt_marker_sets.items()}
        )

    setup = str(_get(record, "setup", "default") or "default").lower()
    params.neurotar = setup == "neurotar"

    # Camera configuration and arena configuration.
    camera_sets = params.get("camera_sets", {})
    _merge_params(params, camera_sets.get(setup, camera_sets.get("default", {})))

    arenas = params.get("arenas", {})
    _merge_params(params, arenas.get(setup, arenas.get("default", {})))

    _normalise_vector_fields(params, ("overhead_neurotar_headring", "overhead_neurotar_center"))

    # Historical Neurotar overrides.
    record_date = _get(record, "date", None)
    if setup == "neurotar":
        if _date_before(record_date, "2023-11-03"):
            params.neurotar_snout_distance_mm = 45
        if _date_before(record_date, "2023-06-21"):
            params.neurotar_snout_distance_mm = 0

    # Record-specific overrides from record.measures.
    measures = _get(record, "measures", None)
    measure_fields = (
        "overhead_neurotar_headring",
        "overhead_neurotar_center",
        "overhead_arena_center",
        "overhead_camera_distortion",
        "overhead_camera_shear",
        "overhead_camera_height",
        "overhead_camera_width",
        "overhead_camera_angle",
        "picamera_time_multiplier",
    )
    if measures:
        for name in measure_fields:
            if _has(measures, name):
                params[name] = _get(measures, name)

    # Select marker set and derive stimulus markers.
    marker_set_name = _choose_marker_set(record)
    if "nt_marker_sets" not in params or marker_set_name not in params.nt_marker_sets:
        raise KeyError(f"Marker set {marker_set_name!r} not found in nt_marker_sets")

    params.marker_set = marker_set_name
    params.markers = params.nt_marker_sets[marker_set_name].copy()

    stop_marker = params.get("nt_stop_marker", None)
    linked_non_behavior = params.markers[~params.markers["behavior"] & params.markers["linked"]]
    params.nt_stim_markers = [m for m in linked_non_behavior["marker"].tolist() if m != stop_marker]

    # Tracking preset.
    tracking_presets = params.get("tracking_presets", {})
    if setup in tracking_presets:
        _merge_params(params, tracking_presets[setup])

    # Optional user-local override, mirroring processparams_local.m.
    if apply_local_overrides:
        try:
            from processparams_local import processparams_local  # type: ignore
        except ImportError:
            pass
        else:
            params = processparams_local(params)

    return params


if __name__ == "__main__":
    # Small smoke test. Requires nt_default_parameters.yaml next to this file.
    p = nt_load_parameters()
    print(f"Loaded {len(p)} top-level parameter fields")
    print(f"Selected marker set: {p.marker_set}")
