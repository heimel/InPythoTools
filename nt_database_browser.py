"""Compatibility wrapper for the NoviTrack database browser.

New code can prefer ``import novitrack as nt`` and call
``nt.browse_nt_database()``.
"""

from novitrack.database_browser import (
    NTDatabaseBrowser,
    browse_nt_database,
    nt_browse_database,
)


__all__ = ["NTDatabaseBrowser", "browse_nt_database", "nt_browse_database"]


def _main() -> int:
    from pathlib import Path
    import sys

    from PyQt6.QtWidgets import QApplication

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
