from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from inpythotools import mat_database
from inpythotools.mat_database import load_mat_database, save_mat_database


def test_load_mat_database_normalizes_empty_comment_only(tmp_path):
    mat_db = np.empty(
        (1, 1),
        dtype=[("comment", object), ("measurements", object)],
    )
    mat_db[0, 0]["comment"] = np.array([])
    mat_db[0, 0]["measurements"] = np.array([])
    filename = tmp_path / "database.mat"
    savemat(filename, {"db": mat_db})

    db = load_mat_database(filename)

    assert db.loc[0, "comment"] == ""
    assert isinstance(db.loc[0, "measurements"], np.ndarray)
    assert db.loc[0, "measurements"].size == 0


def test_save_mat_database_backs_up_existing_database(tmp_path):
    filename = tmp_path / "database.mat"
    old_db = pd.DataFrame([{"comment": "old"}])
    new_db = pd.DataFrame([{"comment": "new"}])
    save_mat_database(old_db, filename)
    original_modification_time = mat_database._modification_timestamp_ns(filename)

    save_mat_database(new_db, filename)

    assert load_mat_database(filename).loc[0, "comment"] == "new"
    backup = tmp_path / "database.mat_copy"
    assert load_mat_database(backup).loc[0, "comment"] == "old"
    assert mat_database._modification_timestamp_ns(backup) == original_modification_time


def test_save_mat_database_keeps_backup_with_matching_modification_time(
    tmp_path, monkeypatch
):
    filename = tmp_path / "database.mat"
    backup = tmp_path / "database.mat_copy"
    save_mat_database(pd.DataFrame([{"comment": "old"}]), filename)
    save_mat_database(pd.DataFrame([{"comment": "backup"}]), backup)
    monkeypatch.setattr(mat_database, "_modification_timestamp_ns", lambda path: 1)

    save_mat_database(pd.DataFrame([{"comment": "new"}]), filename)

    assert load_mat_database(filename).loc[0, "comment"] == "new"
    assert load_mat_database(backup).loc[0, "comment"] == "backup"


def test_save_failure_leaves_original_database_in_backup(tmp_path, monkeypatch):
    filename = tmp_path / "database.mat"
    save_mat_database(pd.DataFrame([{"comment": "old"}]), filename)

    def fail_save(*args, **kwargs):
        raise OSError("failed")

    monkeypatch.setattr(mat_database, "savemat", fail_save)

    with pytest.raises(OSError, match="failed"):
        save_mat_database(pd.DataFrame([{"comment": "new"}]), filename)

    assert not filename.exists()
    backup = tmp_path / "database.mat_copy"
    assert load_mat_database(backup).loc[0, "comment"] == "old"


def test_save_failure_restores_database_after_creating_broken_file(
    tmp_path, monkeypatch
):
    filename = tmp_path / "database.mat"
    save_mat_database(pd.DataFrame([{"comment": "old"}]), filename)

    def fail_after_creating_file(output_filename, *args, **kwargs):
        Path(output_filename).write_bytes(b"broken")
        raise OSError("failed after creating file")

    monkeypatch.setattr(mat_database, "savemat", fail_after_creating_file)

    with pytest.raises(OSError, match="failed after creating file"):
        save_mat_database(pd.DataFrame([{"comment": "new"}]), filename)

    backup = tmp_path / "database.mat_copy"
    assert load_mat_database(filename).loc[0, "comment"] == "old"
    assert load_mat_database(backup).loc[0, "comment"] == "old"


def test_save_mat_database_round_trips_list_of_empty_records(tmp_path):
    filename = tmp_path / "database.mat"
    db = pd.DataFrame(
        [
            {
                "measures": {
                    "event": {
                        "opto_off": {
                            "Channel1_green": {"parameters": [{}, {}]},
                        }
                    }
                }
            }
        ]
    )

    save_mat_database(db, filename)

    loaded = load_mat_database(filename)
    parameters = loaded.loc[0, "measures"]["event"]["opto_off"][
        "Channel1_green"
    ]["parameters"]
    assert parameters == [{}, {}]
