from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from novitrack.nt_load_video_triggers import nt_load_video_triggers


def test_loads_malformed_three_column_trigger_csv(tmp_path) -> None:
    filename = tmp_path / "0120360_20260519_001_Overhead_triggers.csv"
    filename.write_text(
        "frame,time\n"
        "0,17:16:42.959066,5.8e-05\n"
        "3273,17:18:30.348503,107.389437\n"
        "95258,18:08:45.640236,3122.68117\n"
    )

    record = {"sessionid": "0120360_20260519_001"}
    triggers, events = nt_load_video_triggers(
        record,
        "Overhead",
        params=SimpleNamespace(),
        session_path=tmp_path,
    )

    np.testing.assert_allclose(triggers, [107.389437, 3122.68117])
    np.testing.assert_allclose(events["time"], [5.8e-05, 107.389437, 3122.68117])


def test_loads_legacy_one_column_frame_trigger_csv(tmp_path) -> None:
    filename = tmp_path / "0120360_20260519_001_Overhead_triggers.csv"
    filename.write_text("frame\n30\n60\n")

    record = {"sessionid": "0120360_20260519_001"}
    triggers, events = nt_load_video_triggers(
        record,
        "Overhead",
        framerate=30.0,
        params=SimpleNamespace(),
        session_path=tmp_path,
    )

    np.testing.assert_allclose(triggers, [1.0, 2.0])
    np.testing.assert_allclose(events["time"], [1.0, 2.0])
