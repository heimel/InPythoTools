from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from novitrack.nt_change_times import (
    nt_change_neurotar_to_video_times,
    nt_change_times,
    nt_change_video_to_neurotar_times,
)


class TestNtChangeTimes(unittest.TestCase):
    def test_preserves_shape_and_flattens_triggers(self) -> None:
        from_times = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        triggers_from = np.array([[0.0, 10.0, 20.0]])
        triggers_to = np.array([[5.0], [17.0], [29.0]])

        changed, offset, multiplier = nt_change_times(from_times, triggers_from, triggers_to)

        self.assertEqual(changed.shape, from_times.shape)
        np.testing.assert_allclose(changed, from_times * 1.2 + 5.0)
        self.assertTrue(np.isclose(offset, 5.0))
        self.assertTrue(np.isclose(multiplier, 1.2))

    def test_aligns_missing_first_from_trigger(self) -> None:
        from_times = np.array([20.0, 31.0])
        triggers_from = np.array([0.0, 8.0, 20.0, 31.0])
        triggers_to = np.array([102.0, 222.0, 332.0])

        changed, offset, multiplier = nt_change_times(from_times, triggers_from, triggers_to)

        np.testing.assert_allclose(changed, [222.0, 332.0])
        self.assertTrue(np.isclose(offset, 22.0))
        self.assertTrue(np.isclose(multiplier, 10.0))

    def test_aligns_missing_last_to_trigger(self) -> None:
        from_times = np.array([0.0, 10.0])
        triggers_from = np.array([0.0, 10.0, 20.0])
        triggers_to = np.array([1.0, 21.0, 41.0, 501.0])

        changed, offset, multiplier = nt_change_times(from_times, triggers_from, triggers_to)

        np.testing.assert_allclose(changed, [1.0, 21.0])
        self.assertTrue(np.isclose(offset, 1.0))
        self.assertTrue(np.isclose(multiplier, 2.0))

    def test_single_trigger_uses_supplied_multipliers(self) -> None:
        changed, offset, multiplier = nt_change_times(
            np.array([1.0, 2.0]),
            np.array([10.0]),
            np.array([20.0]),
            multiplier_from=2.0,
            multiplier_to=4.0,
        )

        np.testing.assert_allclose(changed, [2.0, 4.0])
        self.assertTrue(np.isclose(offset, 0.0))
        self.assertTrue(np.isclose(multiplier, 2.0))

    def test_deprecated_video_neurotar_wrappers(self) -> None:
        params = SimpleNamespace(picamera_time_multiplier=2.0)

        np.testing.assert_allclose(
            nt_change_video_to_neurotar_times(np.array([12.0, 14.0]), np.array([10.0]), params),
            [1.0, 2.0],
        )
        np.testing.assert_allclose(
            nt_change_neurotar_to_video_times(np.array([1.0, 2.0]), np.array([10.0]), params),
            [12.0, 14.0],
        )


if __name__ == "__main__":
    unittest.main()
