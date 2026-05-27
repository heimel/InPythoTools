from unittest.mock import patch

from inpythotools.local_config import edit_local_config, ensure_local_config, local_config_path


def test_ensure_local_config_creates_processparams_file(tmp_path):
    path = ensure_local_config(config_dir=tmp_path)

    assert path == tmp_path / "processparams_local.py"
    assert path.exists()
    assert "def processparams_local" in path.read_text(encoding="utf-8")


def test_local_config_path_uses_custom_config_dir(tmp_path):
    path = local_config_path(config_dir=tmp_path)

    assert path == tmp_path / "processparams_local.py"


def test_edit_local_config_accepts_explicit_editor(tmp_path):
    with patch("inpythotools.local_config.subprocess.Popen") as popen:
        path = edit_local_config(config_dir=tmp_path, editor="test-editor")

    assert path == tmp_path / "processparams_local.py"
    popen.assert_called_once_with(["test-editor", str(path)])
