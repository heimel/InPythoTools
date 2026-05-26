from __future__ import annotations

import inspect
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO

_log_file: TextIO | None = None
_log_path: Path | None = None


def _infer_caller() -> str:
    """Return the name of the function that called logmsg()."""
    stack = inspect.stack()

    # stack[0] = _infer_caller, stack[1] = logmsg, stack[2] = caller if present
    if len(stack) > 2:
        name = stack[2].function
        if name != '<module>':
            return name

    return 'WORKSPACE'


def _as_message_list(msg: str | Iterable[str] | None) -> list[str]:
    """Normalize a message or collection of messages to a list of strings."""
    if msg is None:
        return ['[Empty message]']

    if isinstance(msg, str):
        return [msg]

    try:
        return [str(m) for m in msg]
    except TypeError:
        return [str(msg)]


def logmsg(
    msg: str | Iterable[str] | None = None,
    caller: str | None = None,
    save_to_logfile: bool = False,
) -> None:
    """
    Print a message to the console, optionally also writing to a log file.

    This is a Python version of the MATLAB function ``logmsg``.

    Parameters
    ----------
    msg : str, iterable of str, or None, optional
        Message to print. If an iterable is supplied, each item is printed on
        a separate line. If omitted or None, prints '[Empty message]'.
    caller : str or None, optional
        Name to prefix the message with. If omitted, the calling function name
        is inferred. From the interactive console, this becomes 'WORKSPACE'.
    save_to_logfile : bool, default False
        If True, a log file is created in the system temporary directory on the
        first call and reused for subsequent calls.
    """
    global _log_file, _log_path

    if caller is None or caller == '':
        caller = _infer_caller()

    if save_to_logfile and _log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _log_path = Path(tempfile.gettempdir()) / f'invivotools_logmsg_{timestamp}.txt'

        try:
            _log_file = _log_path.open('a', encoding='utf-8')
        except OSError:
            print(f'LOGMSG: Could not open log file {_log_path}')
            _log_file = None
        else:
            print(f'LOGMSG: Writing log to {_log_path}')

    messages = _as_message_list(msg)
    prefix = caller.upper()

    for item in messages:
        print(f'{prefix}: {item}')

    if _log_file is not None:
        for item in messages:
            _log_file.write(f'{prefix}: {item}\n')
        _log_file.flush()


def close_logmsg() -> None:
    """Close the log file opened by logmsg(), if any."""
    global _log_file

    if _log_file is not None:
        _log_file.close()
        _log_file = None


def get_logmsg_path() -> Path | None:
    """Return the path of the current log file, or None if no log file exists."""
    return _log_path
