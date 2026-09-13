import unittest

import numpy as np

from app_core.visual_attention import VisualAttentionAnalyzer


class VisualAttentionGeometryTests(unittest.TestCase):
    def test_eye_opening_ratio_for_open_eye(self):
        points = {
            0: np.array([0.0, 0.0]),
            1: np.array([10.0, 0.0]),
            2: np.array([3.0, -2.0]),
            3: np.array([3.0, 2.0]),
            4: np.array([7.0, -2.0]),
            5: np.array([7.0, 2.0]),
            6: np.array([5.0, -2.0]),
            7: np.array([5.0, 2.0]),
        }
        ratio = VisualAttentionAnalyzer._eye_opening_ratio(
            points.__getitem__,
            (0, 1),
            ((2, 3), (4, 5), (6, 7)),
        )
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 0.4, places=6)

    def test_eye_opening_ratio_decreases_for_closed_eye(self):
        points = {
            0: np.array([0.0, 0.0]),
            1: np.array([10.0, 0.0]),
            2: np.array([3.0, -0.25]),
            3: np.array([3.0, 0.25]),
            4: np.array([7.0, -0.25]),
            5: np.array([7.0, 0.25]),
            6: np.array([5.0, -0.25]),
            7: np.array([5.0, 0.25]),
        }
        ratio = VisualAttentionAnalyzer._eye_opening_ratio(
            points.__getitem__,
            (0, 1),
            ((2, 3), (4, 5), (6, 7)),
        )
        self.assertAlmostEqual(ratio, 0.05, places=6)

    def test_median_reduces_one_bad_vertical_pair(self):
        points = {
            0: np.array([0.0, 0.0]),
            1: np.array([10.0, 0.0]),
            2: np.array([3.0, -2.0]),
            3: np.array([3.0, 2.0]),
            4: np.array([7.0, -2.0]),
            5: np.array([7.0, 2.0]),
            6: np.array([5.0, -8.0]),
            7: np.array([5.0, 8.0]),
        }
        ratio = VisualAttentionAnalyzer._eye_opening_ratio(
            points.__getitem__,
            (0, 1),
            ((2, 3), (4, 5), (6, 7)),
        )
        self.assertAlmostEqual(ratio, 0.4, places=6)

    def test_eye_opening_rejects_tiny_geometry(self):
        points = {
            0: np.array([0.0, 0.0]),
            1: np.array([2.0, 0.0]),
            2: np.array([0.5, -0.5]),
            3: np.array([0.5, 0.5]),
        }
        ratio = VisualAttentionAnalyzer._eye_opening_ratio(
            points.__getitem__,
            (0, 1),
            ((2, 3),),
        )
        self.assertIsNone(ratio)

    def test_head_pose_quality_falls_with_strong_turn(self):
        frontal = VisualAttentionAnalyzer._head_pose_quality(
            head_turn_ratio=0.02,
            head_vertical_ratio=0.52,
        )
        turned = VisualAttentionAnalyzer._head_pose_quality(
            head_turn_ratio=0.30,
            head_vertical_ratio=0.52,
        )
        self.assertGreater(frontal, turned)
        self.assertGreater(frontal, 0.8)

    def test_each_eye_gets_separate_quality(self):
        left = VisualAttentionAnalyzer._eye_quality(
            eye_width_px=20.0,
            head_turn_ratio=0.15,
            head_vertical_ratio=0.52,
            is_left=True,
        )
        right = VisualAttentionAnalyzer._eye_quality(
            eye_width_px=20.0,
            head_turn_ratio=0.15,
            head_vertical_ratio=0.52,
            is_left=False,
        )
        self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()
