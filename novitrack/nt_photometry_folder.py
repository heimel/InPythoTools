"""Resolve NoviTrack/RWD photometry folders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inpythotools.logmsg import logmsg


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def nt_photometry_folder(record: Any, params: Any | None = None) -> tuple[Path | None, bool]:
    """Return the folder containing RWD photometry data, if found."""
    if params is None:
        from .nt_load_parameters import nt_load_parameters

        params = nt_load_parameters(record)

    network_path = _get(params, "networkpathbase")
    project = _get(record, "project")
    dataset = _get(record, "dataset")
    subject = _get(record, "subject")
    sessionid = _get(record, "sessionid")

    if not all([network_path, project, dataset, subject, sessionid]):
        return None, False

    folder = Path(str(network_path)) / str(project) / "Data_collection" / str(dataset) / str(subject) / str(sessionid)
    if not folder.exists():
        date = str(_get(record, "date", "")).replace("-", "_")
        condition = str(_get(record, "condition", ""))
        folder = Path(str(network_path)) / str(project) / "Data_collection" / str(dataset) / f"{date}_{subject}_{condition}"

    if not (folder / "Fluorescence-unaligned.csv").exists():
        dated_folders = sorted(path for path in folder.glob("20*") if path.is_dir()) if folder.exists() else []
        if not dated_folders:
            logmsg(f"Cannot find photometry data for {_get(record, 'sessionid', 'record')}")
            return None, False
        if len(dated_folders) > 1:
            logmsg(f"Not sure which folder to pick for photometry data. Picking first folder for {_get(record, 'sessionid', 'record')}.")
        folder = dated_folders[0]

    if not (folder / "Fluorescence-unaligned.csv").exists():
        logmsg(f"Cannot find photometry data for {_get(record, 'sessionid', 'record')}")
        return None, False
    return folder, True


__all__ = ["nt_photometry_folder"]
