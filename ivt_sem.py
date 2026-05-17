"""Small statistics helpers ported from InVivoTools."""

from __future__ import annotations

from typing import Any

import numpy as np


def ivt_sem(x: Any, axis: int | None = None) -> np.ndarray | float:
    """Return standard error of the mean, matching ``ivt_sem.m``.

    When ``axis`` is omitted, NaNs are ignored in both the standard deviation
    and sample count. When ``axis`` is supplied, this mirrors the MATLAB helper:
    ``std(x,0,dim) / sqrt(size(x,dim))``.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.array([])

    if axis is None:
        return np.nanstd(arr, ddof=1) / np.sqrt(np.sum(~np.isnan(arr)))

    return np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])
