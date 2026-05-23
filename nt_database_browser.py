"""Small PyQt6 browser for NoviTrack pandas databases.

The browser shows one database record at a time and exposes record-level
actions, such as analysis and result plotting, as ordinary Python callbacks.
It is intentionally lightweight so it can be started comfortably from Spyder.
"""

from __future__ import annotations

import ast
import re
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from .analyse_nttestrecord import analyse_nttestrecord
    from .load_mat_database import load_mat_database, save_mat_database
    from .nt_load_parameters import nt_load_parameters
    from .results_nttestrecord import results_nttestrecord
except ImportError:  # pragma: no cover - supports Spyder sessions run from this folder
    from analyse_nttestrecord import analyse_nttestrecord
    from load_mat_database import load_mat_database, save_mat_database
    from nt_load_parameters import nt_load_parameters
    from results_nttestrecord import results_nttestrecord


RecordAction = Callable[[pd.Series], Any]
_OPEN_WINDOWS: list["NTDatabaseBrowser"] = []
_LAST_WINDOW: "NTDatabaseBrowser | None" = None
_SIMPLE_COMPARISON_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(==|!=)\s*([^\s'\"]+)\s*$")


def track_behavior_record(record: pd.Series) -> Any:
    """Launch the behavior tracker lazily so normal database browsing stays light."""
    try:
        from .nt_track_behavior import track_record
    except ImportError:  # pragma: no cover - supports Spyder sessions run from this folder
        from nt_track_behavior import track_record

    return track_record(record)


def _is_missing(value: Any) -> bool:
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    return repr(value)


def _parse_edited_value(text: str, original: Any) -> Any:
    text = text.strip()
    if text == "":
        return np.nan
    if isinstance(original, str):
        return text
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _normalize_action_result(result: Any) -> Mapping[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, pd.Series):
        return result.to_dict()
    if isinstance(result, Mapping):
        return result
    return None


def _filter_database(db: pd.DataFrame, expression: str) -> pd.DataFrame:
    filtered = db.query(expression, engine="python")
    if not filtered.empty:
        return filtered

    match = _SIMPLE_COMPARISON_RE.match(expression)
    if match is None:
        return filtered

    column, operator, value = match.groups()
    if column not in db.columns:
        return filtered

    values = db[column].astype(str)
    mask = values == value
    if operator == "!=":
        mask = ~mask
    return db.loc[mask]


def _load_gui_params(yaml_file: str | Path | None = None) -> tuple[int | None, int | None]:
    try:
        params = nt_load_parameters(yaml_file=yaml_file)
    except Exception:
        return None, None

    font_size = _as_int(params.get("fontsize", None))
    spacing = _as_int(params.get("nt_database_browser_spacing", None))
    return font_size, spacing


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class NTDatabaseBrowser(QMainWindow):
    """Browse and act on one row at a time from a NoviTrack DataFrame."""

    def __init__(
        self,
        db: pd.DataFrame | None = None,
        *,
        filename: str | Path | None = None,
        actions: Mapping[str, RecordAction] | None = None,
        font_size: int | None = None,
        spacing: int | None = None,
        yaml_file: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.db = db.copy() if db is not None else pd.DataFrame()
        self.filename = Path(filename) if filename is not None else None
        self.filtered_index: list[Any] = list(self.db.index)
        self.position = 0
        self._updating_table = False
        self.dirty = False
        self.save_button: QPushButton | None = None

        yaml_font_size, yaml_spacing = _load_gui_params(yaml_file)
        self.font_size = font_size if font_size is not None else yaml_font_size
        self.spacing = spacing if spacing is not None else (yaml_spacing if yaml_spacing is not None else 2)

        if actions is None:
            actions = {
                "Analyze": analyse_nttestrecord,
                "Results": results_nttestrecord,
                "Track": track_behavior_record,
            }
        self.actions = dict(actions)

        self._build_ui()
        self._refresh_view()
        self._update_window_state()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(self.spacing, self.spacing, self.spacing, self.spacing)
        layout.setSpacing(self.spacing)

        if self.font_size is not None:
            font = QFont()
            font.setPointSize(self.font_size)
            self.setFont(font)
        control_height = max(22, (self.font_size or 8) + 14)
        self.setStyleSheet(
            "QPushButton { padding: 1px 6px; } "
            "QLineEdit { padding: 1px 4px; } "
            "QTableWidget::item { padding: 0px 3px; }"
        )

        top_row = QHBoxLayout()
        top_row.setSpacing(self.spacing)
        layout.addLayout(top_row)

        for label, callback in (
            ("Load", self.load_database),
            ("Save", self.save_database),
            ("Save As", self.save_database_as),
        ):
            button = QPushButton(label)
            button.setFixedHeight(control_height)
            button.clicked.connect(callback)
            top_row.addWidget(button)
            if label == "Save":
                self.save_button = button

        top_row.addSpacing(self.spacing)
        for label, callback in self.actions.items():
            button = QPushButton(label)
            button.setFixedHeight(control_height)
            button.clicked.connect(lambda _checked=False, name=label, func=callback: self.run_action(name, func))
            top_row.addWidget(button)
        top_row.addStretch(1)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(self.spacing)
        layout.addLayout(filter_row)
        filter_row.addWidget(QLabel("Filter"))
        self.filter_box = QLineEdit()
        self.filter_box.setFixedHeight(control_height)
        self.filter_box.setPlaceholderText("subject == 102394")
        self.filter_box.returnPressed.connect(self.apply_filter)
        filter_row.addWidget(self.filter_box, stretch=1)

        filter_button = QPushButton("Apply")
        filter_button.setFixedHeight(control_height)
        filter_button.clicked.connect(self.apply_filter)
        filter_row.addWidget(filter_button)

        clear_button = QPushButton("Clear")
        clear_button.setFixedHeight(control_height)
        clear_button.clicked.connect(self.clear_filter)
        filter_row.addWidget(clear_button)

        nav = QHBoxLayout()
        nav.setSpacing(self.spacing)
        layout.addLayout(nav)
        for label, callback in (
            ("|<", self.first_record),
            ("<", self.previous_record),
            (">", self.next_record),
            (">|", self.last_record),
        ):
            button = QPushButton(label)
            button.setFixedHeight(control_height)
            button.clicked.connect(callback)
            button.setFixedWidth(38)
            nav.addWidget(button)
        nav.addStretch(1)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, 2)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(max(18, (self.font_size or 8) + 12))
        self.table.setShowGrid(False)
        self.table.itemChanged.connect(self._cell_changed)
        layout.addWidget(self.table)

        self.resize(300, 560)

    def current_record_index(self) -> Any | None:
        if not self.filtered_index:
            return None
        return self.filtered_index[self.position]

    def current_record(self) -> pd.Series | None:
        index = self.current_record_index()
        if index is None:
            return None
        return self.db.loc[index].copy()

    def load_database(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load MATLAB database",
            str(self.filename.parent if self.filename else Path.cwd()),
            "MATLAB files (*.mat);;All files (*.*)",
        )
        if not filename:
            return
        try:
            self.db = load_mat_database(filename)
            self.filename = Path(filename)
            self.filtered_index = list(self.db.index)
            self.position = 0
            self.dirty = False
            self.filter_box.clear()
            self._refresh_view()
            self._update_window_state()
        except Exception as exc:  # pragma: no cover - GUI error path
            self._show_error("Load failed", exc)

    def save_database(self) -> None:
        if self.filename is None:
            self.save_database_as()
            return
        self._save_to_file(self.filename)

    def save_database_as(self) -> None:
        if self.db.empty:
            QMessageBox.information(self, "Nothing to save", "The database is empty.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save MATLAB database",
            str(self.filename if self.filename else Path.cwd() / "db.mat"),
            "MATLAB files (*.mat);;All files (*.*)",
        )
        if not filename:
            return
        self._save_to_file(Path(filename))

    def _save_to_file(self, filename: str | Path) -> None:
        try:
            save_mat_database(self.db, filename)
            self.filename = Path(filename)
            self.dirty = False
            self._update_window_state()
            self.statusBar().showMessage(f"Saved {filename}", 4000)
        except Exception as exc:  # pragma: no cover - GUI error path
            self._show_error("Save failed", exc)

    def apply_filter(self) -> None:
        expression = self.filter_box.text().strip()
        if not expression:
            self.clear_filter()
            return
        try:
            filtered = _filter_database(self.db, expression)
        except Exception as exc:
            self._show_error("Filter failed", exc)
            return
        self.filtered_index = list(filtered.index)
        self.position = 0
        self._refresh_view()

    def clear_filter(self) -> None:
        self.filter_box.clear()
        self.filtered_index = list(self.db.index)
        self.position = 0
        self._refresh_view()

    def first_record(self) -> None:
        self._move_to(0)

    def previous_record(self) -> None:
        self._move_to(self.position - 1)

    def next_record(self) -> None:
        self._move_to(self.position + 1)

    def last_record(self) -> None:
        self._move_to(len(self.filtered_index) - 1)

    def _move_to(self, position: int) -> None:
        if not self.filtered_index:
            return
        self.position = max(0, min(position, len(self.filtered_index) - 1))
        self._refresh_view()

    def run_action(self, name: str, callback: RecordAction) -> None:
        record = self.current_record()
        index = self.current_record_index()
        if record is None or index is None:
            QMessageBox.information(self, "No record", "There is no current record.")
            return
        try:
            result = callback(record)
        except Exception as exc:  # pragma: no cover - GUI error path
            self._show_error(f"{name} failed", exc)
            return

        updated_record = _normalize_action_result(result)
        if updated_record is not None:
            self._update_current_row(updated_record)
            self._set_dirty(True)
            self.statusBar().showMessage(f"{name} updated the current record.", 4000)
        else:
            self.statusBar().showMessage(f"{name} finished.", 4000)

    def _update_current_row(self, values: Mapping[str, Any]) -> None:
        index = self.current_record_index()
        if index is None:
            return
        for column, value in values.items():
            if column not in self.db.columns:
                self.db[column] = pd.Series([np.nan] * len(self.db), index=self.db.index, dtype=object)
            self.db.at[index, column] = value
        if index not in self.filtered_index:
            self.filtered_index.append(index)
        self._refresh_view()

    def _refresh_view(self) -> None:
        self._updating_table = True
        self.table.setRowCount(0)

        if self.db.empty:
            self.status_label.setText("No database loaded.")
            self._update_window_state()
            self._updating_table = False
            return

        if not self.filtered_index:
            self.status_label.setText("No records match the current filter.")
            self._update_window_state()
            self._updating_table = False
            return

        self.position = max(0, min(self.position, len(self.filtered_index) - 1))
        record = self.current_record()
        if record is None:
            self._updating_table = False
            return

        self.status_label.setText(f"Record {self.position + 1} of {len(self.filtered_index)}")
        self.table.setRowCount(len(record.index))
        for row, column in enumerate(record.index):
            field_item = QTableWidgetItem(str(column))
            field_item.setFlags(field_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value_item = QTableWidgetItem(_format_value(record[column]))
            value_item.setData(Qt.ItemDataRole.UserRole, column)
            self.table.setItem(row, 0, field_item)
            self.table.setItem(row, 1, value_item)

        self._updating_table = False
        self._update_window_state()

    def _cell_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item.column() != 1:
            return
        index = self.current_record_index()
        column = item.data(Qt.ItemDataRole.UserRole)
        if index is None or column is None:
            return
        original = self.db.at[index, column]
        new_value = _parse_edited_value(item.text(), original)
        if not self._values_equal(original, new_value):
            self.db.at[index, column] = new_value
            self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self.dirty = dirty
        self._update_window_state()

    def _update_window_state(self) -> None:
        filename = self.filename.name if self.filename is not None else "Untitled"
        marker = "*" if self.dirty else ""
        self.setWindowTitle(f"NoviTrack database browser - {filename}{marker}")
        if self.save_button is not None:
            self.save_button.setEnabled(self.dirty and not self.db.empty)

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return bool(np.array_equal(left, right, equal_nan=True))
            except TypeError:
                return False
        if _is_missing(left) and _is_missing(right):
            return True
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    def _show_error(self, title: str, exc: Exception) -> None:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        QMessageBox.critical(self, title, f"{exc}\n\n{details}")


def browse_nt_database(
    db: pd.DataFrame | None = None,
    *,
    filename: str | Path | None = None,
    actions: Mapping[str, RecordAction] | None = None,
    font_size: int | None = None,
    spacing: int | None = None,
    yaml_file: str | Path | None = None,
) -> NTDatabaseBrowser:
    """Open a NoviTrack database browser and return the window instance.

    Parameters
    ----------
    db:
        Database DataFrame. If omitted and ``filename`` is given, the file is
        loaded with :func:`load_mat_database`.
    filename:
        Optional ``.mat`` database path. Used for initial loading and as the
        default save location.
    actions:
        Mapping from button labels to callables. Each callable receives the
        current record as a pandas Series. If it returns a mapping or Series,
        the current row is updated with those values.
    font_size:
        Optional GUI font size in points. Defaults to ``fontsize`` from
        ``nt_default_parameters.yaml``.
    spacing:
        Optional spacing in pixels between browser controls. Defaults to
        ``nt_database_browser_spacing`` from ``nt_default_parameters.yaml``.
    yaml_file:
        Optional parameter YAML file used to read the default GUI settings.
    """
    global _LAST_WINDOW

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    if db is None and filename is not None:
        db = load_mat_database(filename)

    window = NTDatabaseBrowser(
        db,
        filename=filename,
        actions=actions,
        font_size=font_size,
        spacing=spacing,
        yaml_file=yaml_file,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    _OPEN_WINDOWS.append(window)
    _LAST_WINDOW = window
    return window


nt_browse_database = browse_nt_database


__all__ = ["NTDatabaseBrowser", "browse_nt_database", "nt_browse_database"]


def _main() -> int:
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    browse_nt_database(filename=filename)
    return app.exec() if owns_app else 0


if __name__ == "__main__":
    exit_code = _main()
    if exit_code:
        raise SystemExit(exit_code)
