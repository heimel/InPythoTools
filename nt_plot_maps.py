"""Plot NoviTrack spatial maps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def nt_plot_maps(record: Mapping[str, Any]) -> plt.Figure | None:
    """Plot spatial occupancy and photometry maps for one record."""
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


__all__ = ["nt_plot_maps"]
