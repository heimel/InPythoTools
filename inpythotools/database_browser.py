"""Small PyQt6 browser for pandas record databases.

The browser shows one database record at a time and exposes record-level
actions, such as analysis and result plotting, as ordinary Python callbacks.
It is intentionally lightweight so it can be started comfortably from Spyder.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import QEventLoop, Qt
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

from .mat_database import load_mat_database, save_mat_database


RecordAction = Callable[[pd.Series], Any]
SessionFolderResolver = Callable[[pd.Series], tuple[Path, bool] | Path | str]
_OPEN_WINDOWS: list["DatabaseBrowser"] = []
_LAST_WINDOW: "DatabaseBrowser | None" = None
_APP: QApplication | None = None
_SIMPLE_COMPARISON_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(==|!=)\s*([^\s'\"]+)\s*$")
_TRAILING_INT_RE = re.compile(r"^(.*?)(\d+)(\D*)$")
_PERSISTENT_TAG = "persistent"


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


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _last_record_position(index: list[Any]) -> int:
    return max(0, len(index) - 1)


def _next_record_index(index: pd.Index) -> Any:
    numeric_values = [value for value in index if isinstance(value, (int, np.integer))]
    if numeric_values:
        return max(numeric_values) + 1

    next_index = len(index)
    while next_index in index:
        next_index += 1
    return next_index


def _increment_record_value(value: Any) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value) + 1
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value) + 1
    if isinstance(value, str):
        match = _TRAILING_INT_RE.match(value)
        if match is None:
            return value
        prefix, digits, suffix = match.groups()
        incremented = str(int(digits) + 1).zfill(len(digits))
        return f"{prefix}{incremented}{suffix}"
    return value


def _has_persistent_tag(obj: Any) -> bool:
    if getattr(obj, "tag", None) == _PERSISTENT_TAG:
        return True
    try:
        return obj.property("tag") == _PERSISTENT_TAG
    except AttributeError:
        return False


def _filename_with_selected_extension(filename: str | Path, selected_filter: str) -> Path:
    path = Path(filename)
    if path.suffix:
        return path
    if "Python pickle" in selected_filter:
        return path.with_suffix(".pkl")
    if "Excel workbook" in selected_filter:
        return path.with_suffix(".xlsx")
    if "CSV file" in selected_filter:
        return path.with_suffix(".csv")
    return path.with_suffix(".mat")


def _ensure_qapplication() -> QApplication:
    global _APP

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    _APP = app
    return app


class DatabaseBrowser(QMainWindow):
    """Browse and act on one row at a time from a pandas DataFrame."""

    def __init__(
        self,
        db: pd.DataFrame | None = None,
        *,
        filename: str | Path | None = None,
        actions: Mapping[str, RecordAction] | None = None,
        session_folder_resolver: SessionFolderResolver | None = None,
        window_title_prefix: str = "Database browser",
        font_size: int | None = None,
        spacing: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tag = _PERSISTENT_TAG
        self.setProperty("tag", _PERSISTENT_TAG)

        self.db = db.copy() if db is not None else pd.DataFrame()
        self.filename = Path(filename) if filename is not None else None
        self.filtered_index: list[Any] = list(self.db.index)
        self.position = _last_record_position(self.filtered_index)
        self._updating_table = False
        self.dirty = False
        self.save_button: QPushButton | None = None
        self.session_folder_resolver = session_folder_resolver
        self.window_title_prefix = window_title_prefix
        self.font_size = font_size
        self.spacing = spacing if spacing is not None else 2

        if actions is None:
            actions = {}
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
            ("Export", self.export_database),
            ("Close figs", self.close_nonpersistent_figures),
            ("Explore", self.explore_session_folder),
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
            ("-", self.delete_current_record),
            ("+", self.duplicate_current_record),
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
            self.position = _last_record_position(self.filtered_index)
            self.dirty = False
            self.filter_box.clear()
            self._refresh_view()
            self._update_window_state()
        except Exception as exc:  # pragma: no cover - GUI error path
            self._show_error("Load failed", exc)

    def save_database(self) -> None:
        if self.filename is None:
            self.export_database()
            return
        self._save_to_file(self.filename)

    def export_database(self) -> None:
        if self.db.empty:
            QMessageBox.information(self, "Nothing to save", "The database is empty.")
            return
        export_db = self.db
        export_selection = False
        if self._has_active_filter():
            choice = self._ask_export_scope()
            if choice is None:
                return
            export_selection = choice == "selection"
            if export_selection:
                export_db = self.db.loc[self.filtered_index]

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export database",
            str(self.filename if self.filename else Path.cwd() / "db.mat"),
            "MATLAB database (*.mat);;Python pickle (*.pkl *.pickle);;Excel workbook (*.xlsx);;CSV file (*.csv);;All files (*.*)",
        )
        if not filename:
            return
        self._save_to_file(
            _filename_with_selected_extension(filename, selected_filter),
            db=export_db,
            update_filename=not export_selection,
        )

    def save_database_as(self) -> None:
        self.export_database()

    def _save_to_file(
        self,
        filename: str | Path,
        *,
        db: pd.DataFrame | None = None,
        update_filename: bool = True,
    ) -> None:
        try:
            export_db = self.db if db is None else db
            filename = Path(filename)
            suffix = filename.suffix.lower()
            if suffix == ".mat":
                save_mat_database(export_db, filename)
            elif suffix in {".pkl", ".pickle"}:
                export_db.to_pickle(filename)
            elif suffix in {".xlsx", ".xls"}:
                export_db.to_excel(filename, index=False)
            elif suffix == ".csv":
                export_db.to_csv(filename, index=False)
            else:
                raise ValueError(
                    f"Unsupported export extension {suffix!r}. Use .mat, .pkl, .xlsx, or .csv."
                )
            if update_filename:
                self.filename = Path(filename)
                self.dirty = False
                self._update_window_state()
            self.statusBar().showMessage(f"Saved {filename}", 4000)
        except Exception as exc:  # pragma: no cover - GUI error path
            self._show_error("Save failed", exc)

    def delete_current_record(self) -> None:
        index = self.current_record_index()
        if index is None:
            QMessageBox.information(self, "No record", "There is no current record to delete.")
            return
        indexes_to_delete = self._confirm_delete_indexes(index)
        if not indexes_to_delete:
            return

        self.db = self.db.drop(index=indexes_to_delete)
        self.filtered_index = [value for value in self.filtered_index if value not in indexes_to_delete]
        if not self.filtered_index:
            self.filtered_index = list(self.db.index)
        self.position = max(0, min(self.position, len(self.filtered_index) - 1))
        self._set_dirty(True)
        self._refresh_view()

    def duplicate_current_record(self) -> None:
        record = self.current_record()
        if record is None:
            QMessageBox.information(self, "No record", "There is no current record to duplicate.")
            return

        new_record = record.copy()
        for column in ("sessionid", "sessnr"):
            if column in new_record.index:
                new_record[column] = _increment_record_value(new_record[column])
        if "measures" in new_record.index:
            new_record["measures"] = {}
        if "comment" in new_record.index:
            new_record["comment"] = ""

        new_index = _next_record_index(self.db.index)
        new_row = pd.DataFrame([new_record.to_dict()], index=[new_index])
        current_index = self.current_record_index()
        insert_at = self.db.index.get_loc(current_index) + 1
        before = self.db.iloc[:insert_at]
        after = self.db.iloc[insert_at:]
        self.db = pd.concat([before, new_row, after], sort=False)
        self.filtered_index = list(self.db.index)
        self.position = self.filtered_index.index(new_index)
        self.filter_box.clear()
        self._set_dirty(True)
        self._refresh_view()

    def close_nonpersistent_figures(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - matplotlib is a runtime dependency
            self._show_error("Close figures failed", exc)
            return

        closed = 0
        for number in plt.get_fignums():
            figure = plt.figure(number)
            if _has_persistent_tag(figure):
                continue
            plt.close(figure)
            closed += 1
        self.statusBar().showMessage(f"Closed {closed} figure(s).", 4000)

    def explore_session_folder(self) -> None:
        record = self.current_record()
        if record is None:
            QMessageBox.information(self, "No record", "There is no current record to explore.")
            return
        if self.session_folder_resolver is None:
            QMessageBox.information(self, "Explore unavailable", "No session folder resolver is configured.")
            return
        try:
            resolved = self.session_folder_resolver(record)
            if isinstance(resolved, tuple):
                folder, exists = resolved
            else:
                folder = Path(resolved)
                exists = folder.is_dir()
        except Exception as exc:  # pragma: no cover - GUI error path
            self._show_error("Explore failed", exc)
            return
        folder = Path(folder)
        if not exists:
            QMessageBox.information(self, "Folder not found", f"{folder} does not exist.")
            return
        if os.name == "nt":
            subprocess.Popen(["explorer", str(folder)])
            self.statusBar().showMessage(f"Opened {folder}", 4000)
        else:
            QMessageBox.information(self, "Explore unavailable", f"Folder path:\n{folder}")

    def _has_active_filter(self) -> bool:
        return len(self.filtered_index) != len(self.db.index) or bool(self.filter_box.text().strip())

    def _confirm_delete_indexes(self, current_index: Any) -> list[Any]:
        if self._has_active_filter() and len(self.filtered_index) > 1:
            message = QMessageBox(self)
            message.setIcon(QMessageBox.Icon.Warning)
            message.setWindowTitle("Delete records")
            message.setText(
                f"Delete the current record or all {len(self.filtered_index)} records in the current selection?"
            )
            current_button = message.addButton("Current record", QMessageBox.ButtonRole.AcceptRole)
            selection_button = message.addButton("Current selection", QMessageBox.ButtonRole.DestructiveRole)
            message.addButton(QMessageBox.StandardButton.Cancel)
            message.exec()
            clicked = message.clickedButton()
            if clicked == current_button:
                return [current_index]
            if clicked == selection_button:
                return list(self.filtered_index)
            return []

        answer = QMessageBox.question(
            self,
            "Delete record",
            "Delete the current record?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return [current_index] if answer == QMessageBox.StandardButton.Yes else []

    def _ask_export_scope(self) -> str | None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Question)
        message.setWindowTitle("Export database")
        message.setText(
            f"Export all {len(self.db)} records or only the {len(self.filtered_index)} records in the current selection?"
        )
        all_button = message.addButton("All records", QMessageBox.ButtonRole.AcceptRole)
        selection_button = message.addButton("Current selection", QMessageBox.ButtonRole.AcceptRole)
        message.addButton(QMessageBox.StandardButton.Cancel)
        message.exec()
        clicked = message.clickedButton()
        if clicked == all_button:
            return "all"
        if clicked == selection_button:
            return "selection"
        return None

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
        self.setWindowTitle(f"{self.window_title_prefix} - {filename}{marker}")
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


def browse_database(
    db: pd.DataFrame | None = None,
    *,
    filename: str | Path | None = None,
    actions: Mapping[str, RecordAction] | None = None,
    session_folder_resolver: SessionFolderResolver | None = None,
    window_title_prefix: str = "Database browser",
    font_size: int | None = None,
    spacing: int | None = None,
    block: bool | None = None,
) -> DatabaseBrowser:
    """Open a DataFrame database browser and return the window instance.

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
    session_folder_resolver:
        Optional callable used by the ``Explore`` button. It receives the current
        record and returns either a path or ``(path, exists)``.
    font_size:
        Optional GUI font size in points.
    spacing:
        Optional spacing in pixels between browser controls.
    block:
        If true, run a local Qt event loop until the browser window closes. The
        default is blocking when this call creates the QApplication, and
        non-blocking when Qt is already running.
    """
    global _LAST_WINDOW

    existing_app = QApplication.instance()
    app = _ensure_qapplication()
    if block is None:
        block = existing_app is None

    if db is None and filename is not None:
        db = load_mat_database(filename)

    window = DatabaseBrowser(
        db,
        filename=filename,
        actions=actions,
        session_folder_resolver=session_folder_resolver,
        window_title_prefix=window_title_prefix,
        font_size=font_size,
        spacing=spacing,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    _OPEN_WINDOWS.append(window)
    _LAST_WINDOW = window
    if block:
        if existing_app is None:
            app.exec()
        else:
            loop = QEventLoop()
            window.destroyed.connect(loop.quit)
            loop.exec()
    return window


NTDatabaseBrowser = DatabaseBrowser
nt_browse_database = browse_database
browse_nt_database = browse_database


__all__ = ["DatabaseBrowser", "NTDatabaseBrowser", "browse_database", "browse_nt_database", "nt_browse_database"]


def _main() -> int:
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    browse_database(filename=filename)
    return app.exec() if owns_app else 0


if __name__ == "__main__":
    exit_code = _main()
    if exit_code:
        raise SystemExit(exit_code)
