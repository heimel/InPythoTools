"""Convert timestamps between NoviTrack time reference frames."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from inpythotools.logmsg import logmsg


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return MATLAB-like corrcoef(x, y)[0, 1], tolerating short vectors."""
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _prefer_tail_alignment(longer: np.ndarray, shorter: np.ndarray) -> bool:
    """Choose whether a longer trigger vector should be tail-aligned."""
    n_long = longer.size
    n_short = shorter.size
    cc_missing_first = _correlation(longer[n_long - n_short :], shorter)
    cc_missing_last = _correlation(longer[:n_short], shorter)

    if np.isnan(cc_missing_first) and np.isnan(cc_missing_last):
        return False
    if np.isnan(cc_missing_last):
        return True
    if np.isnan(cc_missing_first):
        return False
    return cc_missing_first > cc_missing_last


def nt_change_times(
    from_times: Any,
    triggers_from: Any,
    triggers_to: Any,
    multiplier_from: float | None = None,
    multiplier_to: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Change timestamps from one reference frame to another.

    This mirrors MATLAB ``nt_change_times.m``:

    ``to = multiplier * from_times + offset``

    Trigger arrays are internally flattened so row/column orientation does not
    affect the alignment, matching the recent MATLAB-side normalization.
    """
    from_array = np.asarray(from_times, dtype=float)
    original_shape = from_array.shape
    flat_from = from_array.reshape(-1)
    triggers_from_vec = _as_vector(triggers_from)
    triggers_to_vec = _as_vector(triggers_to)

    n_triggers_from = triggers_from_vec.size
    n_triggers_to = triggers_to_vec.size

    if n_triggers_from == 0 or n_triggers_to == 0:
        raise ValueError("nt_change_times requires at least one trigger in both reference frames.")

    if n_triggers_from == 1 and n_triggers_to > 1:
        logmsg("Detected too many triggers TO. Using only the first! May be wrong trigger. If so edit trigger log.")
        triggers_to_vec = triggers_to_vec[:1]
        n_triggers_to = 1
    if n_triggers_to == 1 and n_triggers_from > 1:
        logmsg("Detected too many triggers FROM. Using only the first! May be wrong trigger. If so edit trigger log.")
        triggers_from_vec = triggers_from_vec[:1]
        n_triggers_from = 1

    if n_triggers_from > n_triggers_to:
        if _prefer_tail_alignment(triggers_from_vec, triggers_to_vec):
            triggers_from_vec = triggers_from_vec[n_triggers_from - n_triggers_to :]
            logmsg("Missed first FROM triggers in TO reference")
        else:
            triggers_from_vec = triggers_from_vec[:n_triggers_to]
            logmsg("Missed last FROM triggers in TO reference")
        n_triggers_from = triggers_from_vec.size

    if n_triggers_from < n_triggers_to:
        if _prefer_tail_alignment(triggers_to_vec, triggers_from_vec):
            triggers_to_vec = triggers_to_vec[n_triggers_to - n_triggers_from :]
            logmsg("Missed first TO triggers in FROM reference")
        else:
            triggers_to_vec = triggers_to_vec[:n_triggers_from]
            logmsg("Missed last TO triggers in FROM reference")
        n_triggers_to = triggers_to_vec.size

    if n_triggers_from == 1:
        if multiplier_from is None or multiplier_to is None:
            logmsg(
                "Only single matching trigger and no multipliers given. "
                "Assuming them to be 1. This is inaccurate for large times."
            )
            multiplier_from = 1.0
            multiplier_to = 1.0
        triggers_from_vec = np.array(
            [triggers_from_vec[0], triggers_from_vec[0] + 1000 * float(multiplier_from)],
            dtype=float,
        )
        triggers_to_vec = np.array(
            [triggers_to_vec[0], triggers_to_vec[0] + 1000 * float(multiplier_to)],
            dtype=float,
        )

    cc = _correlation(triggers_from_vec, triggers_to_vec)
    if not np.isnan(cc) and cc < 0.999:
        logmsg(
            f"Only correlation of {cc:.3g} between TO and FROM triggers. "
            "This suggest missing triggers and inaccurate time change."
        )

    x = np.column_stack((np.ones(triggers_from_vec.size), triggers_from_vec))
    offset, multiplier = np.linalg.lstsq(x, triggers_to_vec, rcond=None)[0]
    offset = float(offset)
    multiplier = float(multiplier)

    if abs(multiplier - 1) > 0.01:
        logmsg("Clocks are more than 1% different. There is a likely mismatch of triggers.")

    changed = (flat_from * multiplier + offset).reshape(original_shape)
    return changed, offset, multiplier


def nt_change_video_to_neurotar_times(video_t: Any, trigger_times: Any, params: Any) -> np.ndarray:
    """Deprecated MATLAB-compatible wrapper for video-to-neurotar time conversion."""
    logmsg("DEPRECATED: CHANGE CODE TO USE nt_change_times")
    trigger_times_vec = _as_vector(trigger_times)
    if trigger_times_vec.size == 0:
        raise ValueError("trigger_times must contain at least one value.")
    return (np.asarray(video_t, dtype=float) - trigger_times_vec[0]) / float(_get(params, "picamera_time_multiplier"))


def nt_change_neurotar_to_video_times(neurotar_t: Any, trigger_times: Any, params: Any) -> np.ndarray:
    """Deprecated MATLAB-compatible wrapper for neurotar-to-video time conversion."""
    trigger_times_vec = _as_vector(trigger_times)
    if trigger_times_vec.size == 0:
        raise ValueError("trigger_times must contain at least one value.")
    return float(_get(params, "picamera_time_multiplier")) * np.asarray(neurotar_t, dtype=float) + trigger_times_vec[0]


__all__ = [
    "nt_change_times",
    "nt_change_video_to_neurotar_times",
    "nt_change_neurotar_to_video_times",
]
