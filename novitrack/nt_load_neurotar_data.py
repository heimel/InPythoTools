"""Load Neurotar tracking data into the NoviTrack data structure."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from inpythotools.mat_database import _convert_mat_value
from inpythotools.logmsg import logmsg


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _smoothen(x: np.ndarray, sigma: float | int | None = 1) -> np.ndarray:
    """Gaussian smoothing matching MATLAB ``smoothen`` for 1D arrays."""
    x = _as_array(x)
    if x.size == 0:
        return x
    if sigma is None or np.isnan(float(sigma)) or float(sigma) == 0:
        return x

    sigma = float(sigma)
    cutoff = int(np.ceil(3 * sigma))
    if cutoff == 0:
        return x

    bins = np.arange(-cutoff, cutoff + 1, dtype=float)
    if sigma > 0:
        kernel = np.exp(-(bins**2) / (2 * sigma**2))
    else:
        kernel = np.ones_like(bins)
    kernel = kernel / np.sum(kernel)
    return np.convolve(x, kernel, mode="same")


def _thresholdlinear(x: np.ndarray) -> np.ndarray:
    out = _as_array(x).copy()
    out[out < 0] = 0
    return out


def _load_npz_cache(filename: Path) -> dict[str, Any]:
    logmsg(f"Loading neurotar data {filename}")
    with np.load(filename, allow_pickle=False) as cache:
        return {field: cache[field] for field in cache.files}


def _save_npz_cache(filename: Path, neurotar_data: Mapping[str, Any]) -> None:
    arrays = {}
    for field, value in neurotar_data.items():
        arr = np.asarray(value)
        if arr.dtype == object:
            continue
        arrays[field] = arr
    np.savez_compressed(filename, **arrays)


def _load_tdms_data(filename: Path) -> dict[str, Any]:
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError(
            "Reading Neurotar TDMS files requires the conda-forge package "
            "`nptdms` in the gui_pyqt environment."
        ) from exc

    logmsg(f"Loading neurotar data {filename}")
    tdms_file = TdmsFile.read(filename)
    try:
        group = tdms_file["Pp_Data"]
    except KeyError as exc:
        groups = ", ".join(group.name for group in tdms_file.groups())
        raise KeyError(f"TDMS file {filename} has no Pp_Data group. Available groups: {groups}") from exc

    neurotar_data: dict[str, Any] = {}
    for channel in group.channels():
        neurotar_data[channel.name] = np.asarray(channel[:])
    neurotar_data = _pad_tdms_channels(neurotar_data)
    return neurotar_data


def _pad_tdms_channels(neurotar_data: dict[str, Any]) -> dict[str, Any]:
    lengths = [len(value) for value in neurotar_data.values() if np.asarray(value).ndim > 0]
    if not lengths:
        return neurotar_data
    n_samples = max(lengths)

    for field, value in list(neurotar_data.items()):
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] == n_samples:
            continue
        if arr.shape[0] > n_samples:
            neurotar_data[field] = arr[:n_samples]
            continue

        if np.issubdtype(arr.dtype, np.number):
            padded = np.full(n_samples, np.nan, dtype=float)
            padded[: arr.shape[0]] = arr.astype(float)
        elif np.issubdtype(arr.dtype, np.datetime64):
            padded = np.full(n_samples, np.datetime64("NaT"), dtype=arr.dtype)
            padded[: arr.shape[0]] = arr
        else:
            padded = np.empty(n_samples, dtype=object)
            padded[:] = None
            padded[: arr.shape[0]] = arr
        neurotar_data[field] = padded

    return neurotar_data


def _find_neurotar_folder(record: Any, params: Any) -> Path | None:
    base = Path(str(_get(params, "networkpathbase"))) / str(_get(record, "project")) / "Data_collection" / "Neurotar"
    date = str(_get(record, "date", ""))
    subject = str(_get(record, "subject", ""))
    sessnr = _get(record, "sessnr", 1)
    prefix = f"Track_[{date}"
    suffix = f"]_{subject}_session{int(sessnr)}"

    matches = (
        sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith(prefix) and path.name.endswith(suffix))
        if base.exists()
        else []
    )
    if not matches and subject == "exampleVideo":
        base = base / "exampleVideos"
        matches = (
            sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith(prefix) and path.name.endswith(suffix))
            if base.exists()
            else []
        )

    if not matches:
        return None
    if len(matches) > 1:
        logmsg(f"Cannot decide which Neurotar data to use. Multiple folders matching {prefix}*{suffix}.")
        return None
    return matches[0]


def _trim_after_end_marker(neurotar_data: dict[str, Any], record: Any) -> dict[str, Any]:
    measures = _get(record, "measures", {})
    markers = _get(measures, "markers", [])
    if not isinstance(markers, list):
        return neurotar_data

    end_times = [
        float(_get(marker, "time"))
        for marker in markers
        if str(_get(marker, "marker", "")) == "e" and _get(marker, "time", None) is not None
    ]
    if not end_times:
        return neurotar_data

    end_time = end_times[0]
    time = _as_array(neurotar_data.get("Time", []))
    if time.size == 0:
        since_start = _as_array(neurotar_data.get("Since_track_start", []))
        ttl = _as_array(neurotar_data.get("TTL_outputs", []))
        trigger_frames = np.flatnonzero(ttl != 0)
        if since_start.size and trigger_frames.size:
            time = since_start - since_start[trigger_frames[0]]
    if time.size == 0:
        return neurotar_data

    keep_count = int(np.searchsorted(time, end_time, side="left"))
    logmsg(f"Removing all data after end of session marker at {end_time:.2g} s.")
    for field, value in list(neurotar_data.items()):
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == time.size:
            neurotar_data[field] = arr[:keep_count]
    return neurotar_data


def nt_load_neurotar_data(record: Any, params: Any | None = None) -> tuple[dict[str, Any], Path | None]:
    """Load Neurotar data from Python cache, MATLAB cache, or TDMS."""
    if params is None:
        from .nt_load_parameters import nt_load_parameters

        params = nt_load_parameters(record)

    folder = _find_neurotar_folder(record, params)
    if folder is None:
        return {}, None

    close_bracket = folder.name.find("]")
    stem = folder.name[: close_bracket + 1] if close_bracket >= 0 else folder.name
    filename = folder / stem
    npz_filename = filename.with_suffix(".npz")
    mat_filename = filename.with_suffix(".mat")
    tdms_filename = filename.with_suffix(".tdms")

    source_filename = None
    if npz_filename.exists():
        neurotar_data = _load_npz_cache(npz_filename)
        source_filename = npz_filename
    elif mat_filename.exists():
        logmsg(f"Loading neurotar data {mat_filename}")
        mat = loadmat(mat_filename, squeeze_me=True, struct_as_record=False)
        if "neurotar_data" not in mat:
            logmsg(f"No neurotar_data variable found in {mat_filename}")
            return {}, mat_filename

        neurotar_data = _convert_mat_value(mat["neurotar_data"])
        if not isinstance(neurotar_data, dict):
            logmsg(f"Unsupported neurotar_data format in {mat_filename}")
            return {}, mat_filename
        source_filename = mat_filename
    elif tdms_filename.exists():
        try:
            neurotar_data = _load_tdms_data(tdms_filename)
        except (ImportError, KeyError, OSError, ValueError) as exc:
            logmsg(str(exc))
            return {}, tdms_filename
        try:
            _save_npz_cache(npz_filename, neurotar_data)
            logmsg(f"Saved neurotar Python cache {npz_filename}")
        except OSError as exc:
            logmsg(f"Could not save neurotar Python cache {npz_filename}: {exc}")
        source_filename = tdms_filename
    else:
        logmsg(f"Cannot find Neurotar data {mat_filename}, {npz_filename}, or {tdms_filename}")
        return {}, None

    neurotar_data = _trim_after_end_marker(neurotar_data, record)

    since_start = _as_array(neurotar_data.get("Since_track_start", []))
    ttl_outputs = _as_array(neurotar_data.get("TTL_outputs", []))
    if since_start.size == 0:
        logmsg(f"Neurotar data has no Since_track_start in {source_filename}")
        return {}, source_filename

    trigger_frames = np.flatnonzero(ttl_outputs != 0)
    if trigger_frames.size:
        trigger_frame = int(trigger_frames[0])
    else:
        logmsg("No Neurotar TTL output trigger found. Using first sample as time zero.")
        trigger_frame = 0
    neurotar_data["Time"] = since_start - since_start[trigger_frame]

    speed = _as_array(neurotar_data.get("Speed", []))
    if speed.size:
        neurotar_data["Speed"] = speed / 1000

    alpha = _as_array(neurotar_data.get("alpha", []))
    x = _as_array(neurotar_data.get("X", []))
    y = _as_array(neurotar_data.get("Y", []))
    time = _as_array(neurotar_data["Time"])
    if alpha.size == time.size and x.size == time.size and y.size == time.size:
        dt = np.concatenate(([1.0], np.diff(time)))
        dt[dt == 0] = np.nan

        rx = -np.sin(alpha / 180 * np.pi)
        ry = np.cos(alpha / 180 * np.pi)
        dx = np.concatenate(([0.0], np.diff(x)))
        dy = np.concatenate(([0.0], np.diff(y)))
        forward_speed = (rx * dx + ry * dy) / dt
        neurotar_data["Forward_speed"] = _smoothen(forward_speed, _get(params, "nt_temporal_filter_width", 5))

        complex_alpha = np.exp(1j * alpha / 180 * np.pi)
        d_alpha = complex_alpha[1:] / complex_alpha[:-1]
        angular_velocity = np.concatenate(([0.0], np.angle(d_alpha) / np.pi * 180)) / dt
        neurotar_data["Angular_velocity"] = _smoothen(angular_velocity, _get(params, "nt_temporal_filter_width", 5))

    radius = _as_array(neurotar_data.get("R", []))
    if radius.size == time.size:
        neurotar_data["Distance_to_wall"] = _thresholdlinear(float(_get(params, "arena_radius_mm")) - radius)

    n_samples = time.size
    nan_vec = np.full(n_samples, np.nan)
    neurotar_data.setdefault("Object_distance", nan_vec.copy())
    neurotar_data["CoM_X"] = nan_vec.copy()
    neurotar_data["CoM_Y"] = nan_vec.copy()
    neurotar_data["tailbase_X"] = nan_vec.copy()
    neurotar_data["tailbase_Y"] = nan_vec.copy()
    neurotar_data["Coordinates"] = _get(params, "ARENA", 1)

    return neurotar_data, source_filename


__all__ = ["nt_load_neurotar_data"]
