"""Load NoviTrack/RWD fiber-photometry data.

Python translation of the data-loading part of ``nt_load_photometry.m``.
The preprocessing step lives in ``nt_preprocess_photometry.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from inpythotools.logmsg import logmsg
from .nt_change_times import nt_change_times
from .nt_load_parameters import nt_load_parameters
from .nt_photometry_folder import nt_photometry_folder


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _as_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def parse_channels(comment: str | None) -> dict[str, str]:
    """Parse channel-to-fiber mappings from a record comment."""
    if not comment:
        return {}

    result: dict[str, str] = {}
    pattern = re.compile(r"channel(\d+)\s*=\s*([^,]+)", flags=re.IGNORECASE)
    for channel_number, fiber in pattern.findall(str(comment)):
        value = fiber.strip().lower()
        if value.endswith("."):
            value = value[:-1]
        result[f"Channel{channel_number.strip()}"] = value
    return result


def _convert_event_rwd(
    event_rwd: pd.DataFrame,
    prev_state: Sequence[float] | None = None,
    slack_time: float = 0.02,
) -> pd.DataFrame:
    input_names = ["Input1", "Input2", "Input3", "Input4", "Input5"]
    trigger_names = ["Trigger1", "Trigger2", "Trigger3", "Trigger4", "Trigger5"]
    if prev_state is None:
        prev_state = [1, 1, np.nan, np.nan, np.nan]

    rows: list[dict[str, Any]] = []
    for input_name, trigger_name, initial_state in zip(input_names, trigger_names, prev_state):
        input_events = event_rwd.loc[event_rwd["Name"] == input_name, ["TimeStamp", "State"]]
        times = input_events["TimeStamp"].to_numpy(dtype=float)
        states = input_events["State"].to_numpy(dtype=float)
        state = float(initial_state)

        index = 0
        while index < len(states):
            if not np.isnan(state) and states[index] == state:
                index += 1
                continue

            start_time = float(times[index])
            changed_state = float(states[index])
            previous_state = 1 - changed_state if np.isnan(state) else state

            return_candidates = np.flatnonzero(states[index + 1 :] == previous_state)
            if return_candidates.size == 0:
                duration = 0.0
                state = changed_state
                index += 1
                code = input_name
            else:
                return_index = index + 1 + int(return_candidates[0])
                duration = float(times[return_index] - start_time)
                state = previous_state
                index = return_index + 1
                if duration <= slack_time:
                    code = trigger_name
                    duration = 0.0
                else:
                    code = input_name

            rows.append({"time": start_time, "code": code, "duration": duration})

    return pd.DataFrame(rows, columns=["time", "code", "duration"]).sort_values("time").reset_index(drop=True)


def nt_load_rwd_triggers(
    photometry_folder: str | Path,
    params: Any | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load RWD trigger timestamps from ``Events.csv``."""
    events_file = Path(photometry_folder) / "Events.csv"
    if not events_file.exists():
        return np.array([], dtype=float), pd.DataFrame()

    events = pd.read_csv(events_file)
    events["TimeStamp"] = events["TimeStamp"] / 1000
    prev_state = _get(params, "rwd_initial_input_state", [1, 1, np.nan, np.nan, np.nan])
    slack_time = float(_get(params, "rwd_slack_time", 0.02))
    converted_events = _convert_event_rwd(events, prev_state, slack_time)
    triggers = converted_events.loc[converted_events["code"].astype(str) == "Trigger1", "time"].to_numpy()

    if triggers.size == 0:
        logmsg("No triggers found on Input1")
        if not converted_events.empty:
            logmsg("But there are events on other inputs")

    return triggers, converted_events


def _marker_for_marker_id(params: Any, marker_id: str, default_marker: str) -> str:
    markers = _get(params, "markers", None)
    if isinstance(markers, pd.DataFrame) and "marker_id" in markers:
        matches = markers.loc[markers["marker_id"] == marker_id, "marker"]
        if not matches.empty:
            return str(matches.iloc[0])
    logmsg(f"Cannot find marker motif {marker_id}. Using marker {default_marker}.")
    return default_marker


def _has_markers(measures: Mapping[str, Any]) -> bool:
    markers = _get(measures, "markers", None)
    return markers is not None and len(markers) > 0


def _import_rwd_markers(
    measures: dict[str, Any],
    rwd_events: pd.DataFrame,
    rwd_triggers: np.ndarray,
    trigger_times: np.ndarray,
    params: Any,
) -> None:
    if rwd_events.empty:
        return

    events = rwd_events.copy()
    events["time"], _, multiplier = nt_change_times(events["time"].to_numpy(), rwd_triggers, trigger_times)
    events["duration"] = events["duration"].to_numpy(dtype=float) * multiplier

    markers = list(_get(measures, "markers", []) or [])
    opto_on_marker = _marker_for_marker_id(params, "opto_on", "1")
    opto_off_marker = _marker_for_marker_id(params, "opto_off", "0")

    for _, event in events.loc[events["code"] == "Input3"].iterrows():
        time = float(event["time"])
        duration = float(event["duration"])
        markers.append({"time": time, "marker": opto_on_marker})
        markers.append({"time": time + duration, "marker": opto_off_marker})

    stim_events = events.loc[events["code"] != "Input3"].copy()
    unique_codes = {code: index + 1 for index, code in enumerate(sorted(stim_events["code"].astype(str).unique()))}
    for _, event in stim_events.iterrows():
        code = str(event["code"])
        markers.append({"time": float(event["time"]), "marker": f"o{unique_codes[code]}"})

    markers.sort(key=lambda marker: float(_get(marker, "time")))
    measures["markers"] = markers


def _default_channel_mapping(channel_names: Sequence[str]) -> dict[str, str]:
    return {channel: f"fiber{channel[7:]}" for channel in channel_names}


def _light_type(wavelength: Any) -> str:
    wavelength = int(wavelength)
    if wavelength == 410:
        return "isosbestic"
    if wavelength == 470:
        return "green"
    if wavelength == 560:
        return "red"
    return "unknown"


def _fiber_info(measures: Mapping[str, Any], fiber: str) -> dict[str, Any]:
    info = _get(_get(measures, "fiber_info", {}), fiber, None)
    if isinstance(info, Mapping):
        return {
            "hemisphere": _get(info, "hemisphere", ""),
            "location": _get(info, "location", ""),
            "green_sensor": _get(info, "green_sensor", "green_sensor"),
            "red_sensor": _get(info, "red_sensor", "red_sensor"),
        }
    return {
        "hemisphere": "",
        "location": "",
        "green_sensor": "green_sensor",
        "red_sensor": "red_sensor",
    }


def nt_load_photometry(
    record: Any,
    params: Any | None = None,
    *,
    photometry_folder: str | Path | None = None,
    copy: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load raw photometry data and update the session measures dictionary."""
    if params is None:
        params = nt_load_parameters(record)

    measures = deepcopy(_get(record, "measures", {})) if copy else dict(_get(record, "measures", {}))
    photometry: dict[str, Any] = {}

    if "channels" in measures:
        measures.pop("channels")

    if photometry_folder is None:
        folder, found = nt_photometry_folder(record, params)
        if not found or folder is None:
            return photometry, measures
    else:
        folder = Path(photometry_folder)
        if not (folder / "Fluorescence-unaligned.csv").exists():
            return photometry, measures

    fluorescence = pd.read_csv(folder / "Fluorescence-unaligned.csv")
    wavelengths = sorted(fluorescence["Lights"].dropna().unique())
    channel_names = [name for name in fluorescence.columns if name not in {"TimeStamp", "Lights"}]

    channel_mapping = parse_channels(_get(record, "comment", ""))
    if not channel_mapping:
        logmsg("No channels specified in comment. Assuming channelX = fiberX.")
        channel_mapping = _default_channel_mapping(channel_names)

    channels: list[dict[str, Any]] = []
    for channel_name in channel_names:
        fiber = channel_mapping.get(channel_name, _default_channel_mapping([channel_name])[channel_name])
        info = _fiber_info(measures, fiber)
        lights = [{"wavelength": int(wavelength), "type": _light_type(wavelength), "median": np.nan} for wavelength in wavelengths]
        channels.append(
            {
                "channel": channel_name,
                "hemisphere": info["hemisphere"],
                "location": info["location"],
                "green_sensor": info["green_sensor"],
                "red_sensor": info["red_sensor"],
                "lights": lights,
                "fit_isos": [],
                "sample_rate": [],
            }
        )

    fluorescence["TimeStamp"] = fluorescence["TimeStamp"] / 1000

    triggers_fp, rwd_events = nt_load_rwd_triggers(folder, params)
    if triggers_fp.size == 0:
        logmsg("No recorded RWD triggers. Assuming 0.")
        triggers_fp = np.array([0.0])

    trigger_times = _as_array(_get(measures, "trigger_times", None))
    if trigger_times.size == 0:
        logmsg("No record trigger_times found. Using RWD trigger times.")
        trigger_times = triggers_fp.copy()
        measures["trigger_times"] = trigger_times

    fluorescence_time, _, _ = nt_change_times(fluorescence["TimeStamp"].to_numpy(), triggers_fp, trigger_times)
    fluorescence["time"] = fluorescence_time

    if not _has_markers(measures):
        _import_rwd_markers(measures, rwd_events, triggers_fp, trigger_times, params)

    markers = _get(measures, "markers", [])
    if markers:
        marker_times = np.array([float(_get(marker, "time")) for marker in markers], dtype=float)
        pretime = float(_get(params, "nt_pretime", 10))
        posttime = float(_get(params, "nt_posttime", 20))
        measures["period_of_interest"] = np.array(
            [
                max(float(fluorescence["time"].iloc[0]), np.nanmin(marker_times) - pretime),
                min(float(fluorescence["time"].iloc[-1]), np.nanmax(marker_times) + posttime),
            ]
        )
    else:
        duration = float(fluorescence["time"].iloc[-1] - fluorescence["time"].iloc[0])
        measures["period_of_interest"] = np.array(
            [
                float(fluorescence["time"].iloc[0] + duration * 0.05),
                float(fluorescence["time"].iloc[-1] - duration * 0.05),
            ]
        )

    for channel in channels:
        channel_name = channel["channel"]
        photometry[channel_name] = {}
        last_type = None
        for light in channel["lights"]:
            light_type = light["type"]
            last_type = light_type
            mask = fluorescence["Lights"] == light["wavelength"]
            time = fluorescence.loc[mask, "time"].to_numpy()
            trace = fluorescence.loc[mask, channel_name].to_numpy()
            photometry[channel_name][light_type] = {"time": time, "signal": trace}

            period = _as_array(measures["period_of_interest"])
            period_mask = (time > period[0]) & (time < period[1])
            light["median"] = float(np.nanmedian(trace[period_mask])) if np.any(period_mask) else np.nan

        if last_type is not None:
            time = photometry[channel_name][last_type]["time"]
            channel["sample_rate"] = float(1 / np.nanmedian(np.diff(time)))

    measures["channels"] = channels
    return photometry, measures
