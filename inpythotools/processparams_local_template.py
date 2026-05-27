"""Local parameter overrides for InPythoTools and NoviTrack.

This file is copied to your user configuration folder as ``processparams_local.py``.
Edit the copied file, not this template.
"""

from __future__ import annotations

from typing import Any


def processparams_local(params: Any) -> Any:
    """Override default analysis parameters for this computer/user."""
    if params is None:
        return {}

    # Example:
    # params.networkpathbase = r"C:\Users\yourname\OneDrive\Projects\Heimel"
    # params.nt_save_snippets = False

    return params
