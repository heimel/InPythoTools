import numpy as np
from scipy.io import savemat

from inpythotools.mat_database import load_mat_database


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
