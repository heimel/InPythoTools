"""PyQt6/PyQtGraph behavior tracking GUI for NoviTrack records."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtGui import QAction, QCloseEvent, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover - dependency checked at runtime
    pg = None
    _PYQTGRAPH_IMPORT_ERROR = exc
else:
    _PYQTGRAPH_IMPORT_ERROR = None

from inpythotools.logmsg import logmsg
from .nt_change_times import nt_change_times
from .nt_load_parameters import nt_load_parameters
from .nt_load_tracking_data import nt_load_tracking_data
from .nt_open_videos import OpenCVVideoReader, VideoInfo, nt_open_videos


_OPEN_WINDOWS: list["NTTrackBehaviorWindow"] = []
_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_record_field(record: Any, name: str, value: Any) -> None:
    if isinstance(record, MutableMapping):
        record[name] = value
    elif isinstance(record, pd.Series):
        record.at[name] = value
    else:
        setattr(record, name, value)


def _as_array(value: Any, default: Sequence[float] | None = None) -> np.ndarray:
    if value is None:
        return np.asarray(default if default is not None else [], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray(default if default is not None else [], dtype=float)


def _ensure_measures(record: Any, params: Any) -> dict[str, Any]:
    measures = _get(record, "measures", None)
    if not isinstance(measures, dict):
        measures = {}
    measures.pop("events", None)
    measures.setdefault("markers", [])
    measures.setdefault("object_positions", np.empty((0, 5)))
    if bool(_get(params, "neurotar", False)):
        measures.setdefault("overhead_neurotar_headring", _get(params, "overhead_neurotar_headring", [np.nan, np.nan]))
        measures.setdefault("overhead_neurotar_center", _get(params, "overhead_neurotar_center", [np.nan, np.nan]))
    else:
        measures.setdefault("overhead_arena_center", _get(params, "overhead_arena_center", [np.nan, np.nan]))
    _set_record_field(record, "measures", measures)
    return measures


def _markers_as_records(markers: Any) -> list[dict[str, Any]]:
    if markers is None:
        return []
    if isinstance(markers, pd.DataFrame):
        return markers.to_dict(orient="records")
    if isinstance(markers, Mapping):
        return [dict(markers)]
    records: list[dict[str, Any]] = []
    try:
        iterator = list(markers)
    except TypeError:
        return records
    for marker in iterator:
        if isinstance(marker, Mapping):
            records.append(dict(marker))
        else:
            records.append({"time": _get(marker, "time", np.nan), "marker": _get(marker, "marker", "")})
    return sorted(records, key=lambda item: float(item.get("time", np.nan)))


def _marker_definition(params: Any, marker_key: str) -> Mapping[str, Any] | None:
    table = _get(params, "markers", pd.DataFrame())
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None
    hits = table[table["marker"].astype(str) == str(marker_key)[0]]
    if hits.empty:
        return None
    return hits.iloc[0].to_dict()


def _qt_color(value: Any) -> tuple[int, int, int]:
    arr = _as_array(value, [0.0, 0.0, 0.0])
    arr = np.clip(arr[:3], 0.0, 1.0)
    return tuple(int(round(v * 255)) for v in arr)


def _record_title(record: Any) -> str:
    sessionid = _get(record, "sessionid", "unknown session")
    subject = _get(record, "subject", "")
    return f"{sessionid} {subject}".strip()


class NTTrackBehaviorWindow(QMainWindow):
    """First usable PyQt6 port of MATLAB ``nt_track_behavior``."""

    def __init__(self, record: Any, parent: QWidget | None = None) -> None:
        if pg is None:
            raise ImportError(
                "nt_track_behavior needs pyqtgraph. Install it in the GUI environment, "
                "for example: conda install -n gui_pyqt -c conda-forge pyqtgraph"
            ) from _PYQTGRAPH_IMPORT_ERROR
        super().__init__(parent)

        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
        self.record = record
        self.params = nt_load_parameters(record)
        self.measures = _ensure_measures(record, self.params)
        self.changed = False
        self.playing = True
        self.playback_speed = 1.0
        self.master_time = 0.0
        self._last_tick = time.perf_counter()
        self._fps_filtered = 0.0
        self._closed = False
        self._video_to_master: dict[int, tuple[float, float]] = {}
        self._master_to_video: dict[int, tuple[float, float]] = {}

        self.readers, self.video_info, self.active_cameras = nt_open_videos(self.record, self.params)
        if not self.active_cameras:
            raise FileNotFoundError("No NoviTrack movies were found for this record.")

        self.nt_data, trigger_times = nt_load_tracking_data(self.record, self.params, recompute=False)
        if not self.nt_data:
            raise FileNotFoundError("No Neurotar/tracking data were found for this record.")
        self.measures["trigger_times"] = _as_array(trigger_times, [0.0])

        self._prepare_time_alignment()
        self._prepare_tracking_arrays()
        self._build_ui()
        self._refresh_marker_items()
        self._seek(0.0, force=True)

        fps = max(1.0, min(float(self.video_info[self.active_cameras[0]].framerate), 60.0))
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(1, int(round(1000 / fps))))

    def _prepare_time_alignment(self) -> None:
        trigger_times = _as_array(self.measures.get("trigger_times"), [0.0])
        if trigger_times.size == 0:
            trigger_times = np.array([0.0], dtype=float)
            self.measures["trigger_times"] = trigger_times
        max_time = 0.0
        min_time = 0.0
        for camera_index in self.active_cameras:
            info = self.video_info[camera_index]
            assert info is not None
            video_triggers = _as_array(info.trigger_times, [0.0])
            if video_triggers.size == 0:
                video_triggers = np.array([0.0], dtype=float)
            _, offset, multiplier = nt_change_times(0.0, video_triggers, trigger_times)
            self._video_to_master[camera_index] = (offset, multiplier)
            _, offset, multiplier = nt_change_times(0.0, trigger_times, video_triggers)
            self._master_to_video[camera_index] = (offset, multiplier)
            video_end, _, _ = nt_change_times(info.duration, video_triggers, trigger_times)
            video_start, _, _ = nt_change_times(0.0, video_triggers, trigger_times)
            max_time = max(max_time, float(np.asarray(video_end)))
            min_time = min(min_time, float(np.asarray(video_start)))

        time_values = _as_array(self.nt_data.get("Time"), [0.0])
        finite = time_values[np.isfinite(time_values)]
        if finite.size:
            min_time = min(min_time, float(np.nanmin(finite)))
            max_time = max(max_time, float(np.nanmax(finite)))
        self.min_time = min_time
        self.max_time = max_time
        self.measures["min_time"] = min_time
        self.measures["max_time"] = max_time

    def _prepare_tracking_arrays(self) -> None:
        self.time_values = _as_array(self.nt_data.get("Time"), [0.0])
        self.x_values = _as_array(self.nt_data.get("X"), np.full_like(self.time_values, np.nan))
        self.y_values = _as_array(self.nt_data.get("Y"), np.full_like(self.time_values, np.nan))
        self.alpha_values = _as_array(self.nt_data.get("alpha"), np.full_like(self.time_values, np.nan))
        self.com_x_values = _as_array(self.nt_data.get("CoM_X"), np.full_like(self.time_values, np.nan))
        self.com_y_values = _as_array(self.nt_data.get("CoM_Y"), np.full_like(self.time_values, np.nan))
        self.tail_x_values = _as_array(self.nt_data.get("tailbase_X"), np.full_like(self.time_values, np.nan))
        self.tail_y_values = _as_array(self.nt_data.get("tailbase_Y"), np.full_like(self.time_values, np.nan))
        self.speed_values = _as_array(
            self.nt_data.get("Forward_speed" if bool(_get(self.params, "nt_forward_speed_in_speed_trace", True)) else "Speed"),
            np.full_like(self.time_values, np.nan),
        )
        self.rotation_values = _as_array(self.nt_data.get("Angular_velocity"), np.full_like(self.time_values, np.nan))
        self.distance_values = _as_array(self.nt_data.get("Object_distance"), np.full_like(self.time_values, np.nan))

    def _build_ui(self) -> None:
        self.setWindowTitle(f"Tracking - {_record_title(self.record)}")
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.setCentralWidget(root)

        toolbar = QToolBar("Tracking", self)
        self.addToolBar(toolbar)
        for text, shortcut, slot in (
            ("Prev marker", "Shift+P", self.previous_marker),
            ("Play", "Space", self.toggle_play),
            ("Next marker", "Shift+N", self.next_marker),
            ("Frame -", "Left", self.backward_frame),
            ("Frame +", "Right", self.forward_frame),
            ("-", "-", self.speed_decrease),
            ("1x", "=", self.speed_original),
            ("+", "+", self.speed_increase),
            ("Add marker", "Shift+M", self.add_marker_dialog),
            ("Delete next", "Del", self.delete_next_marker),
            ("Go to", "Shift+G", self.goto_dialog),
            ("Help", "Shift+H", self.show_help),
            ("Stop", "Shift+Q", self.close),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            toolbar.addAction(action)

        status = QHBoxLayout()
        layout.addLayout(status)
        self.state_label = QLabel("Playing")
        self.time_label = QLabel("0.00")
        self.fps_label = QLabel("")
        self.speed_label = QLabel("1x")
        self.message_label = QLabel("Ready")
        self.message_label.setMinimumWidth(320)
        for label, widget in (
            ("State:", self.state_label),
            ("Time:", self.time_label),
            ("FPS:", self.fps_label),
            ("Speed:", self.speed_label),
        ):
            status.addWidget(QLabel(label))
            status.addWidget(widget)
        status.addWidget(QLabel("Status:"))
        status.addWidget(self.message_label, 1)
        status.addStretch(1)

        video_row = QHBoxLayout()
        layout.addLayout(video_row, stretch=5)
        self.video_views: dict[int, pg.PlotWidget] = {}
        self.video_images: dict[int, pg.ImageItem] = {}
        self.video_info_by_camera: dict[int, VideoInfo] = {}
        self.overhead_mouse_item: pg.PlotDataItem | None = None
        for camera_index in self.active_cameras:
            info = self.video_info[camera_index]
            assert info is not None
            plot = pg.PlotWidget(title=info.camera_name)
            plot.setAspectLocked(True)
            plot.setMouseEnabled(x=False, y=False)
            plot.getViewBox().setDefaultPadding(0)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            self.video_views[camera_index] = plot
            self.video_images[camera_index] = image
            self.video_info_by_camera[camera_index] = info
            video_row.addWidget(plot)

        overhead_index = int(_get(self.params, "nt_overhead_camera", 1)) - 1
        if overhead_index in self.video_views:
            self.overhead_mouse_item = pg.PlotDataItem(
                pen=pg.mkPen((0, 255, 0), width=2),
                symbol="o" if bool(_get(self.params, "nt_show_mouse_keypoints", True)) else None,
                symbolBrush=(0, 255, 0),
                symbolSize=5,
            )
            self.video_views[overhead_index].addItem(self.overhead_mouse_item)

        self.timeline = pg.PlotWidget()
        self.timeline.setBackground("w")
        self.timeline.setMouseEnabled(y=False)
        self.timeline.setYRange(0, float(_get(self.params, "nt_track_timeline_max_speed", 0.375)))
        self.timeline.setXRange(self.min_time, self.max_time, padding=0)
        self.timeline.plot(self.time_values, np.nan_to_num(np.abs(self.speed_values), nan=0.0), pen=pg.mkPen((140, 140, 140)))
        self.timeline_cursor = pg.InfiniteLine(
            self.master_time,
            angle=90,
            pen=pg.mkPen((230, 40, 40), width=3),
            movable=True,
        )
        self.timeline_cursor.setZValue(1000)
        self.timeline_cursor.sigPositionChangeFinished.connect(lambda item: self._seek(float(item.value()), force=True))
        self.timeline.addItem(self.timeline_cursor)
        self.timeline.scene().sigMouseClicked.connect(self._timeline_clicked)
        layout.addWidget(self.timeline, stretch=1)

        traces = QHBoxLayout()
        layout.addLayout(traces, stretch=2)
        self.speed_plot = self._make_trace_plot("Speed", self.speed_values, (-0.25, 0.25))
        traces.addWidget(self.speed_plot)
        self.rotation_plot = self._make_trace_plot("Rotation", self.rotation_values, (-360, 360))
        traces.addWidget(self.rotation_plot)
        self.distance_plot = self._make_trace_plot("Distance", self.distance_values, (0, 300))
        traces.addWidget(self.distance_plot)

        self.resize(1200, 820)
        QTimer.singleShot(0, self._fit_video_views_to_height)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_video_views_to_height)

    def _fit_video_views_to_height(self) -> None:
        """Fit the full movie frame with vertical size as the limiting scale."""
        for camera_index, plot in self.video_views.items():
            info = self.video_info_by_camera[camera_index]
            view_box = plot.getViewBox()
            rect = view_box.geometry()
            view_height = max(float(rect.height()), 1.0)
            view_width = max(float(rect.width()), 1.0)
            view_aspect = view_width / view_height

            x_center = info.width / 2.0
            x_span = max(float(info.width), float(info.height) * view_aspect)
            x0 = x_center - x_span / 2.0
            x1 = x_center + x_span / 2.0
            view_box.setRange(xRange=(x0, x1), yRange=(float(info.height), 0.0), padding=0)

    def _timeline_clicked(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        view_box = self.timeline.getViewBox()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        position = view_box.mapSceneToView(event.scenePos())
        self.playing = False
        self.state_label.setText("Paused")
        self._seek(float(position.x()), force=True)
        self._report_status(f"Jumped to {self.master_time:.2f} s")
        event.accept()

    def _report_status(self, message: str) -> None:
        self.message_label.setText(message)

    def _make_trace_plot(self, title: str, values: np.ndarray, y_range: tuple[float, float]) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.setBackground("w")
        plot.plot(self.time_values, values, pen=pg.mkPen("k"))
        plot.setYRange(*y_range)
        plot.setXRange(-3, 3, padding=0)
        cursor = pg.InfiniteLine(0, angle=90, pen=pg.mkPen((230, 40, 40), width=2))
        cursor.setZValue(1000)
        plot.addItem(cursor)
        setattr(self, f"_{title.lower()}_cursor", cursor)
        return plot

    def _refresh_marker_items(self) -> None:
        markers = _markers_as_records(self.measures.get("markers"))
        self.measures["markers"] = markers
        for item in list(self.timeline.items()):
            if getattr(item, "_nt_marker", False):
                self.timeline.removeItem(item)
        for marker in markers:
            marker_text = str(marker.get("marker", ""))
            marker_time = float(marker.get("time", np.nan))
            if not np.isfinite(marker_time):
                continue
            definition = _marker_definition(self.params, marker_text)
            if definition is not None and bool(definition.get("behavior", False)) and not bool(
                _get(self.params, "nt_show_behavior_markers", True)
            ):
                continue
            color = _qt_color(definition.get("color", [0, 0, 0]) if definition else [0, 0, 0])
            line = pg.InfiniteLine(marker_time, angle=90, pen=pg.mkPen(color, width=1))
            line._nt_marker = True
            line.setZValue(100)
            self.timeline.addItem(line)

    def _tick(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_tick
        self._last_tick = now
        if self.playing:
            self.master_time += elapsed * self.playback_speed
            if self.master_time > self.max_time:
                self.master_time = self.min_time
        self._seek(self.master_time)
        if elapsed > 0:
            fps = 1.0 / elapsed
            self._fps_filtered = fps if self._fps_filtered == 0 else 0.9 * self._fps_filtered + 0.1 * fps
            self.fps_label.setText(f"{self._fps_filtered:.0f}")

    def _seek(self, master_time: float, *, force: bool = False) -> None:
        self.master_time = max(self.min_time, min(float(master_time), self.max_time))
        camera_master_times = []
        for camera_index in self.active_cameras:
            reader = self.readers[camera_index]
            info = self.video_info[camera_index]
            if reader is None or info is None:
                continue
            offset, multiplier = self._master_to_video[camera_index]
            video_time = self.master_time * multiplier + offset
            if force or self.playing:
                frame = reader.read_at_time(video_time)
                if frame is not None:
                    self.video_images[camera_index].setImage(frame, autoLevels=False)
            offset, multiplier = self._video_to_master[camera_index]
            camera_master_times.append(video_time * multiplier + offset)
        if camera_master_times:
            self.master_time = float(np.nanmean(camera_master_times))
        self._update_overlays()
        self._update_trace_ranges()
        self.time_label.setText(f"{self.master_time:.2f}")
        self.timeline_cursor.setValue(self.master_time)

    def _current_index(self) -> int | None:
        if self.time_values.size == 0:
            return None
        index = int(np.searchsorted(self.time_values, self.master_time, side="right") - 1)
        if index < 0 or index >= self.time_values.size:
            return None
        return index

    def _update_overlays(self) -> None:
        if self.overhead_mouse_item is None or not bool(_get(self.params, "nt_show_overhead_mouse", True)):
            return
        index = self._current_index()
        if index is None:
            self.overhead_mouse_item.setData([], [])
            return
        x = np.asarray([self.x_values[index], self.com_x_values[index], self.tail_x_values[index]], dtype=float)
        y = np.asarray([self.y_values[index], self.com_y_values[index], self.tail_y_values[index]], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            self.overhead_mouse_item.setData([], [])
            return
        self.overhead_mouse_item.setData(x[finite], y[finite])

    def _update_trace_ranges(self) -> None:
        half_window = float(_get(self.params, "nt_mouse_trace_window", 3.0))
        x0 = self.master_time - half_window
        x1 = self.master_time + half_window
        for plot in (self.speed_plot, self.rotation_plot, self.distance_plot):
            plot.setXRange(x0, x1, padding=0)
        self._speed_cursor.setValue(self.master_time)
        self._rotation_cursor.setValue(self.master_time)
        self._distance_cursor.setValue(self.master_time)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.state_label.setText("Playing" if self.playing else "Paused")
        self._last_tick = time.perf_counter()
        self._report_status("Playing" if self.playing else "Paused")

    def backward_frame(self) -> None:
        self.playing = False
        self.state_label.setText("Paused")
        self._seek(self.master_time - 1.0 / self._base_fps(), force=True)
        self._report_status(f"Stepped back to {self.master_time:.2f} s")

    def forward_frame(self) -> None:
        self.playing = False
        self.state_label.setText("Paused")
        self._seek(self.master_time + 1.0 / self._base_fps(), force=True)
        self._report_status(f"Stepped forward to {self.master_time:.2f} s")

    def _base_fps(self) -> float:
        info = self.video_info[self.active_cameras[0]]
        return float(info.framerate if info is not None else 30.0)

    def previous_marker(self) -> None:
        marker_times = [float(m["time"]) for m in _markers_as_records(self.measures.get("markers")) if float(m["time"]) < self.master_time - 0.04]
        if marker_times:
            self.playing = False
            self.state_label.setText("Paused")
            self._seek(max(marker_times), force=True)
            self._report_status(f"Jumped to previous marker at {self.master_time:.2f} s")
        else:
            self._report_status("No previous marker")

    def next_marker(self) -> None:
        marker_times = [float(m["time"]) for m in _markers_as_records(self.measures.get("markers")) if float(m["time"]) > self.master_time + 0.04]
        if marker_times:
            self.playing = False
            self.state_label.setText("Paused")
            self._seek(min(marker_times), force=True)
            self._report_status(f"Jumped to next marker at {self.master_time:.2f} s")
        else:
            self._report_status("No next marker")

    def goto_dialog(self) -> None:
        value, ok = QInputDialog.getDouble(self, "Go to", "Second:", self.master_time, self.min_time, self.max_time, 2)
        if ok:
            self.playing = False
            self.state_label.setText("Paused")
            self._seek(value, force=True)
            self._report_status(f"Jumped to {self.master_time:.2f} s")

    def add_marker_dialog(self) -> None:
        keys = [str(row["marker"]) for _, row in _get(self.params, "markers", pd.DataFrame()).iterrows()]
        key, ok = QInputDialog.getItem(self, "Add marker", "Marker:", keys, 0, False)
        if ok and key:
            self.add_marker(key)

    def add_marker(self, marker_key: str) -> None:
        definition = _marker_definition(self.params, marker_key)
        if definition is None:
            self._report_status(f"Unknown marker key {marker_key!r}")
            QMessageBox.warning(self, "Unknown marker", f"Marker {marker_key!r} is not in params.markers.")
            return
        marker_text = marker_key[0]
        if bool(definition.get("linked", False)):
            stim_id = 1 if bool(_get(self.params, "neurotar", False)) else 0
            if not bool(_get(self.params, "neurotar", False)):
                stim_id, ok = QInputDialog.getInt(self, "Stimulus", "Stimulus id:", 1, 1, 9)
                if not ok:
                    self._report_status("Marker insertion cancelled")
                    return
            marker_text = f"{marker_text}{stim_id}"

        markers = _markers_as_records(self.measures.get("markers"))
        if any(abs(float(m.get("time", np.nan)) - self.master_time) < 1e-9 and str(m.get("marker")) == marker_text for m in markers):
            logmsg(f"Marker {marker_text} already present at t = {self.master_time:g}.")
            self._report_status(f"Marker {marker_text} already present at {self.master_time:.2f} s")
            return
        markers.append({"time": float(self.master_time), "marker": marker_text})
        self.measures["markers"] = sorted(markers, key=lambda item: float(item["time"]))
        if marker_key[0] == str(_get(self.params, "nt_stop_marker", "t")):
            positions = np.asarray(self.measures.get("object_positions", np.empty((0, 5))), dtype=float).reshape(-1, 5)
            stim_id = int(marker_text[1:]) if len(marker_text) > 1 and marker_text[1:].isdigit() else 1
            positions = np.vstack([positions, [self.master_time, np.nan, np.nan, float(_get(self.params, "ARENA", 1)), stim_id]])
            self.measures["object_positions"] = positions[np.argsort(positions[:, 0])]
        self.changed = True
        self._refresh_marker_items()
        _set_record_field(self.record, "measures", self.measures)
        self._report_status(f"Added marker {marker_text} at {self.master_time:.2f} s")

    def delete_next_marker(self) -> None:
        markers = _markers_as_records(self.measures.get("markers"))
        later = [(i, marker) for i, marker in enumerate(markers) if float(marker.get("time", np.nan)) > self.master_time]
        if not later:
            self._report_status("No next marker to delete")
            return
        index, marker = later[0]
        answer = QMessageBox.question(self, "Delete marker", f"Delete marker {marker.get('marker')} at {float(marker.get('time')):.2f} s?")
        if answer != QMessageBox.StandardButton.Yes:
            self._report_status("Marker deletion cancelled")
            return
        del markers[index]
        self.measures["markers"] = markers
        self.changed = True
        self._refresh_marker_items()
        _set_record_field(self.record, "measures", self.measures)
        self._report_status(f"Deleted marker {marker.get('marker')} at {float(marker.get('time')):.2f} s")

    def speed_increase(self) -> None:
        index = min(_SPEEDS.index(self.playback_speed) + 1, len(_SPEEDS) - 1)
        self.playback_speed = _SPEEDS[index]
        self.speed_label.setText(f"{self.playback_speed:g}x")
        self._report_status(f"Playback speed {self.playback_speed:g}x")

    def speed_decrease(self) -> None:
        index = max(_SPEEDS.index(self.playback_speed) - 1, 0)
        self.playback_speed = _SPEEDS[index]
        self.speed_label.setText(f"{self.playback_speed:g}x")
        self._report_status(f"Playback speed {self.playback_speed:g}x")

    def speed_original(self) -> None:
        self.playback_speed = 1.0
        self.speed_label.setText("1x")
        self._report_status("Playback speed 1x")

    def show_help(self) -> None:
        marker_lines = []
        table = _get(self.params, "markers", pd.DataFrame())
        if isinstance(table, pd.DataFrame):
            marker_lines = [f"{row.marker}: {row.description}" for row in table.itertuples()]
        QMessageBox.information(
            self,
            "Tracking help",
            "\n".join(
                [
                    "Space: play/pause",
                    "Left/Right: frame step",
                    "Shift+P/Shift+N: previous/next marker",
                    "+/-: playback speed",
                    "Shift+M: add marker",
                    "Shift+G: go to time",
                    "Delete: delete next marker",
                    "Shift+H: show help",
                    "Shift+Q/Esc: stop tracking",
                    "",
                    *marker_lines,
                ]
            ),
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        modifiers = event.modifiers()
        if key == Qt.Key.Key_M and modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.add_marker_dialog()
        elif key == Qt.Key.Key_Delete:
            self.delete_next_marker()
        elif key == Qt.Key.Key_G and modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.goto_dialog()
        elif key == Qt.Key.Key_Q and modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.close()
        elif key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_Left and modifiers == Qt.KeyboardModifier.AltModifier:
            self._seek(self.master_time - 5.0, force=True)
        elif key == Qt.Key.Key_Right and modifiers == Qt.KeyboardModifier.AltModifier:
            self._seek(self.master_time + 5.0, force=True)
        elif event.text():
            marker_key = event.text()
            if _marker_definition(self.params, marker_key) is not None:
                self.add_marker(marker_key)
            else:
                self._report_unmapped_key(event)
                super().keyPressEvent(event)
        else:
            self._report_unmapped_key(event)
            super().keyPressEvent(event)

    def _report_unmapped_key(self, event: QKeyEvent) -> None:
        text = event.text()
        key_name = text if text else f"key code {int(event.key())}"
        self._report_status(f"Unmapped key: {key_name}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closed = True
        self.timer.stop()
        _set_record_field(self.record, "measures", self.measures)
        for reader in self.readers:
            if reader is not None:
                reader.close()
        if self in _OPEN_WINDOWS:
            _OPEN_WINDOWS.remove(self)
        super().closeEvent(event)


def nt_track_behavior(record: Any, *, block: bool | None = None) -> Any:
    """Open the behavior tracking GUI for one NoviTrack record.

    If ``block`` is true, this function returns ``(record, changed)`` after the
    window closes. If ``block`` is false, it returns the live window object. The
    default is non-blocking when a QApplication already exists, which is friendlier
    inside Spyder, and blocking when this function creates the application.
    """
    app = QApplication.instance()
    created_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    if block is None:
        block = created_app

    window = NTTrackBehaviorWindow(record)
    _OPEN_WINDOWS.append(window)
    window.show()

    if not block:
        return window

    loop = QEventLoop()
    window.destroyed.connect(loop.quit)
    if created_app:
        app.exec()
    else:
        loop.exec()
    return record, window.changed


def track_record(record: Any) -> Any:
    """Database-browser friendly wrapper that returns an updated record."""
    updated_record, _changed = nt_track_behavior(record, block=True)
    return updated_record


__all__ = ["NTTrackBehaviorWindow", "nt_track_behavior", "track_record"]


if __name__ == "__main__":
    from inpythotools.mat_database import load_mat_database

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python nt_track_behavior.py database.mat [row_index]")
    db = load_mat_database(Path(sys.argv[1]))
    row = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nt_track_behavior(db.iloc[row], block=True)
