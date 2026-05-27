"""User-local configuration helpers for Heimel-lab Python tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


CONFIG_FILENAME = "processparams_local.py"
APP_NAME = "inpythotools"


def user_config_dir(app_name: str = APP_NAME) -> Path:
    """Return the platform-specific user configuration directory."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / app_name
        return Path.home() / "AppData" / "Roaming" / app_name

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name

    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / app_name
    return Path.home() / ".config" / app_name


def local_config_path(
    *,
    app_name: str = APP_NAME,
    filename: str = CONFIG_FILENAME,
    config_dir: str | Path | None = None,
) -> Path:
    """Return the path to the user-local parameter override file."""
    folder = Path(config_dir) if config_dir is not None else user_config_dir(app_name)
    return folder / filename


def ensure_local_config(
    *,
    app_name: str = APP_NAME,
    filename: str = CONFIG_FILENAME,
    config_dir: str | Path | None = None,
    template_path: str | Path | None = None,
) -> Path:
    """Create the user-local parameter override file from the template if needed."""
    target = local_config_path(app_name=app_name, filename=filename, config_dir=config_dir)
    if target.exists():
        return target

    source = Path(template_path) if template_path is not None else Path(__file__).with_name(
        "processparams_local_template.py"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _open_in_editor(path: Path, editor: str | None = None) -> None:
    """Open ``path`` in a text editor."""
    editor = editor or os.environ.get("EDITOR")
    if editor:
        subprocess.Popen([editor, str(path)])
        return

    if sys.platform.startswith("win"):
        subprocess.Popen(["notepad", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-t", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def edit_local_config(
    *,
    app_name: str = APP_NAME,
    filename: str = CONFIG_FILENAME,
    config_dir: str | Path | None = None,
    template_path: str | Path | None = None,
    editor: str | None = None,
) -> Path:
    """Create the local config file if needed and open it in a text editor."""
    path = ensure_local_config(
        app_name=app_name,
        filename=filename,
        config_dir=config_dir,
        template_path=template_path,
    )
    _open_in_editor(path, editor=editor)
    return path
