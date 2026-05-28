from __future__ import annotations

import inspect
from collections.abc import Iterable

from .logmsg import logmsg

_OPEN_MESSAGE_BOXES: list[object] = []


def _infer_caller() -> str:
    stack = inspect.stack()
    if len(stack) > 2:
        name = stack[2].function
        if name != "<module>":
            return name
    return "Error"


def _as_text(msg: str | Iterable[str] | None) -> str:
    if msg is None:
        return "[Empty message]"
    if isinstance(msg, str):
        return msg
    try:
        return "\n".join(str(item) for item in msg)
    except TypeError:
        return str(msg)


def _userize(name: str) -> str:
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else "Error"


def _show_error_dialog(message: str, title: str) -> bool:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except Exception:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    box.destroyed.connect(lambda _=None, item=box: _OPEN_MESSAGE_BOXES.remove(item) if item in _OPEN_MESSAGE_BOXES else None)
    _OPEN_MESSAGE_BOXES.append(box)
    box.show()
    box.raise_()
    box.activateWindow()
    return True


def errormsg(
    msg: str | Iterable[str] | None = None,
    halt: bool = False,
    caller: str | None = None,
) -> None:
    """Show a small error dialog and log the same message.

    This mirrors the MATLAB ``errormsg`` helper used in InVivoTools. When a
    Qt application is active, a non-modal error dialog is shown. In headless
    contexts it quietly falls back to logging only.
    """
    if caller is None or caller == "":
        caller = _infer_caller()

    message = _as_text(msg)
    _show_error_dialog(message, _userize(caller))
    logmsg(message, caller)

    if halt:
        raise RuntimeError(message)


__all__ = ["errormsg"]
