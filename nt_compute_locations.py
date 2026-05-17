"""Compute location-derived NoviTrack measures."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _cart2pol(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.arctan2(y, x), np.sqrt(x**2 + y**2)


def _pol2cart(theta: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return radius * np.cos(theta), radius * np.sin(theta)


def nt_change_overhead_to_camera_coordinates(
    overhead_x: Any, overhead_y: Any, params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Convert overhead image coordinates in pixels to camera-centered coordinates."""
    x = _as_array(overhead_x) - float(_get(params, "overhead_camera_width")) / 2 + _as_array(
        _get(params, "overhead_camera_image_offset", [0, 0])
    )[0]
    y = _as_array(overhead_y) - float(_get(params, "overhead_camera_height")) / 2 + _as_array(
        _get(params, "overhead_camera_image_offset", [0, 0])
    )[1]

    distort = _as_array(_get(params, "overhead_camera_distortion"))
    method = str(_get(params, "overhead_camera_distortion_method", "normal"))

    if method == "normal":
        theta, overhead_r = _cart2pol(x, y)
        return _pol2cart(theta, overhead_r / distort[0])

    if method == "fisheye_log":
        x = x * _as_array(_get(params, "overhead_camera_shear", [1, 1]))[0]
        y = y * _as_array(_get(params, "overhead_camera_shear", [1, 1]))[1]
        theta, overhead_r = _cart2pol(x, y)
        return _pol2cart(theta, (np.exp(overhead_r * distort[0]) - 1) / distort[0])

    if method == "fisheye_equidistant":
        theta, overhead_r = _cart2pol(x, y)
        camera_phi = overhead_r / distort[1]
        return _pol2cart(theta, distort[0] * np.tan(camera_phi))

    if method == "fisheye_orthographic":
        theta, overhead_r = _cart2pol(x, y)
        camera_r = distort[0] * np.tan(np.arcsin(np.minimum(overhead_r / distort[1], 1)))
        camera_x, camera_y = _pol2cart(theta, camera_r)
        outside = overhead_r > distort[1]
        camera_x[outside] = np.nan
        camera_y[outside] = np.nan
        return camera_x, camera_y

    raise ValueError(f"Unknown overhead_camera_distortion_method: {method}")


def nt_change_camera_to_arena_coordinates(
    camera_x: Any, camera_y: Any, params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-centered coordinates to arena coordinates."""
    if bool(_get(params, "neurotar", False)):
        raise NotImplementedError("Neurotar camera-to-arena coordinates are not ported yet.")

    center = _as_array(_get(params, "overhead_arena_center"))
    center_x, center_y = nt_change_overhead_to_camera_coordinates(center[0], center[1], params)

    x = _as_array(camera_x) - center_x[0]
    y = _as_array(camera_y) - center_y[0]

    alpha = -float(_get(params, "overhead_camera_angle", 0))
    arena_x = np.cos(alpha) * x + np.sin(alpha) * y
    arena_y = -np.sin(alpha) * x + np.cos(alpha) * y
    return arena_x, arena_y


def nt_change_overhead_to_arena_coordinates(
    overhead_x: Any, overhead_y: Any, params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Convert overhead coordinates to arena coordinates."""
    if bool(_get(params, "neurotar", False)):
        raise NotImplementedError("Neurotar overhead-to-arena coordinates are not ported yet.")
    camera_x, camera_y = nt_change_overhead_to_camera_coordinates(overhead_x, overhead_y, params)
    return nt_change_camera_to_arena_coordinates(camera_x, camera_y, params)


def nt_arena_walls(params: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return arena wall coordinates."""
    shape = str(_get(params, "arena_shape"))
    if shape == "circular":
        theta = np.arange(0, 2 * np.pi + np.pi / 15, np.pi / 15)
        return float(_get(params, "arena_radius_mm")) * np.sin(theta), float(_get(params, "arena_radius_mm")) * np.cos(theta)

    if shape == "square":
        half_width = float(_get(params, "arena_diameter_mm")) / 2
        return np.array([-1, 1, 1, -1, -1]) * half_width, np.array([-1, -1, 1, 1, -1]) * half_width

    if shape == "plus":
        how = 50 / 2
        hcw = 63 / 2
        ol = hcw + 297
        cl = how + 300
        return (
            np.array([hcw, ol, np.nan, ol, hcw, hcw, -hcw, -hcw, -ol, np.nan, -ol, -hcw, -hcw, hcw, hcw]),
            np.array([-how, -how, np.nan, how, how, cl, cl, how, how, np.nan, -how, -how, -cl, -cl, -how]),
        )

    raise ValueError(f"Unknown arena_shape: {shape}")


def _inpolygon(x: np.ndarray, y: np.ndarray, polygon_x: np.ndarray, polygon_y: np.ndarray) -> np.ndarray:
    valid = ~(np.isnan(polygon_x) | np.isnan(polygon_y))
    polygon = np.column_stack([polygon_x[valid], polygon_y[valid]])
    points = np.column_stack([x, y])
    finite = np.isfinite(points).all(axis=1)
    inside = np.zeros(x.shape, dtype=bool)
    if polygon.size:
        inside[finite] = MplPath(polygon).contains_points(points[finite])
    return inside


def nt_compute_locations(
    record: Mapping[str, Any],
    nt_data: Mapping[str, Any],
    params: Any,
    *,
    copy: bool = True,
) -> dict[str, Any]:
    """Compute center and arena occupancy fractions."""
    out = deepcopy(dict(record)) if copy else dict(record)
    measures = deepcopy(dict(_get(out, "measures", {})))

    arena_x, arena_y = nt_change_overhead_to_arena_coordinates(nt_data["CoM_X"], nt_data["CoM_Y"], params)
    arena_walls_x, arena_walls_y = nt_arena_walls(params)

    center_scale = (float(_get(params, "arena_radius_mm")) - float(_get(params, "nt_max_distance_to_wall"))) / float(
        _get(params, "arena_radius_mm")
    )
    center_x = center_scale * arena_walls_x
    center_y = center_scale * arena_walls_y

    in_center = _inpolygon(arena_x, arena_y, center_x, center_y)
    in_arena = _inpolygon(arena_x, arena_y, arena_walls_x, arena_walls_y)

    measures["frac_in_center"] = float(np.sum(in_center) / len(in_center)) if len(in_center) else np.nan
    measures["frac_out_off_arena"] = float(np.sum(~in_arena) / len(in_arena)) if len(in_arena) else np.nan

    out["measures"] = measures
    return out
