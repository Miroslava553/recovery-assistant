import unittest

from app_core.ocular_activity import (
    EyeActivityTracker,
    EyeSignalState,
    OcularEventType,
)


class EyeActivityTrackerTests(unittest.TestCase):
    def make_tracker(self, **overrides) -> EyeActivityTracker:
        settings = {
            "window_sec": 10.0,
            "calibration_min_samples": 5,
            "calibration_min_sec": 0.2,
            "min_observation_sec": 0.2,
            "min_eye_quality": 0.5,
            "min_head_pose_quality": 0.3,
            "partial_ratio_threshold": 0.8,
            "closed_ratio_threshold": 0.55,
            "full_blink_ratio_threshold": 0.75,
            "open_ratio_threshold": 0.82,
            "min_blink_sec": 0.05,
            "max_blink_sec": 0.7,
            "long_closure_sec": 1.0,
            "severe_closure_sec": 2.0,
            "bilateral_sync_tolerance_sec": 0.12,
            "max_sample_gap_sec": 2.0,
            "candidate_gap_tolerance_sec": 0.12,
            "reopen_confirmation_samples": 2,
            "min_reliable_fps": 5.0,
            "min_signal_coverage": 0.5,
            "smoothing_alpha": 1.0,
        }
        settings.update(overrides)
        return EyeActivityTracker(**settings)

    def update(self, tracker, left, right, at, **quality):
        return tracker.update(
            left,
            right,
            face_detected=quality.get("face_detected", True),
            data_quality=quality.get("data_quality", 0.9),
            left_eye_quality=quality.get("left_eye_quality", 0.9),
            right_eye_quality=quality.get("right_eye_quality", 0.9),
            head_pose_quality=quality.get("head_pose_quality", 0.9),
            captured_at=at,
        )

    def calibrate(self, tracker, start=0.0, left=0.30, right=0.33):
        current = start
        for _ in range(6):
            self.update(tracker, left, right, current)
            current += 0.05
        self.assertTrue(tracker.calibrated)
        return current

    def begin_window(self, tracker, current):
        tracker.reset(keep_calibration=True)
        self.update(tracker, 0.30, 0.33, current)
        return current + 0.05

    def test_low_quality_eye_is_not_used(self):
        tracker = self.make_tracker()
        result = self.update(
            tracker,
            0.30,
            0.33,
            0.0,
            left_eye_quality=0.2,
        )
        self.assertFalse(result.calibrated)
        self.assertEqual(result.reason, "left_eye_quality_too_low")
        self.assertEqual(result.left_eye_state, EyeSignalState.UNCERTAIN)

    def test_references_are_personal_and_separate(self):
        tracker = self.make_tracker()
        self.calibrate(tracker, left=0.28, right=0.35)
        result = tracker.snapshot(captured_at=0.4)
        self.assertAlmostEqual(result.left_open_eye_reference, 0.28, places=3)
        self.assertAlmostEqual(result.right_open_eye_reference, 0.35, places=3)

    def test_bilateral_blink_is_counted(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.10, 0.11, current)
        self.update(tracker, 0.09, 0.10, current + 0.05)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        result = self.update(tracker, 0.30, 0.33, current + 0.15)
        self.assertEqual(result.blink_count, 1)
        self.assertEqual(result.incomplete_blink_count, 0)
        self.assertEqual(result.left_wink_count, 0)
        self.assertEqual(result.right_wink_count, 0)

    def test_left_wink_is_not_counted_as_blink(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.09, 0.33, current)
        self.update(tracker, 0.08, 0.33, current + 0.05)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        result = self.update(tracker, 0.30, 0.33, current + 0.15)
        self.assertEqual(result.blink_count, 0)
        self.assertEqual(result.left_wink_count, 1)

    def test_incomplete_bilateral_blink_is_separate(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.235, 0.258, current)
        self.update(tracker, 0.232, 0.255, current + 0.05)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        result = self.update(tracker, 0.30, 0.33, current + 0.15)
        self.assertEqual(result.blink_count, 0)
        self.assertEqual(result.incomplete_blink_count, 1)

    def test_asynchronous_single_eye_events_are_uncertain(self):
        tracker = self.make_tracker(bilateral_sync_tolerance_sec=0.08)
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.09, 0.33, current)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        self.update(tracker, 0.30, 0.10, current + 0.20)
        self.update(tracker, 0.30, 0.33, current + 0.30)
        result = self.update(tracker, 0.30, 0.33, current + 0.35)
        self.assertEqual(result.blink_count, 0)
        self.assertGreaterEqual(result.uncertain_event_count, 1)

    def test_long_bilateral_closure_is_not_normal_blink(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.09, 0.10, current)
        self.update(tracker, 0.09, 0.10, current + 0.6)
        self.update(tracker, 0.09, 0.10, current + 1.1)
        self.update(tracker, 0.30, 0.33, current + 1.2)
        result = self.update(tracker, 0.30, 0.33, current + 1.25)
        self.assertEqual(result.blink_count, 0)
        self.assertEqual(result.long_closure_count, 1)

    def test_ongoing_two_second_closure_is_severe(self):
        tracker = self.make_tracker(max_sample_gap_sec=3.0)
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.09, 0.10, current)
        result = self.update(tracker, 0.09, 0.10, current + 2.1)
        self.assertTrue(result.severe_closure_detected)
        self.assertGreaterEqual(result.current_closure_duration_sec, 2.0)

    def test_perclos_counts_only_bilateral_closed_time(self):
        tracker = self.make_tracker(max_sample_gap_sec=2.0)
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.09, 0.10, current + 0.95)
        self.update(tracker, 0.09, 0.10, current + 1.95)
        result = self.update(tracker, 0.30, 0.33, current + 2.95)
        # Интервалы: 1 с открыты, 2 с оба закрыты.
        self.assertAlmostEqual(result.perclos, 2.0 / 3.0, places=2)

    def test_face_loss_does_not_create_false_long_closure(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.09, 0.10, current)
        self.update(
            tracker,
            None,
            None,
            current + 0.1,
            face_detected=False,
        )
        result = self.update(tracker, 0.30, 0.33, current + 2.0)
        self.assertEqual(result.long_closure_count, 0)
        self.assertFalse(result.severe_closure_detected)

    def test_low_processing_fps_blocks_data_ready(self):
        tracker = self.make_tracker(
            min_reliable_fps=10.0,
            max_sample_gap_sec=2.0,
            min_observation_sec=1.0,
        )
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.30, 0.33, current + 1.0)
        result = self.update(tracker, 0.30, 0.33, current + 2.0)
        self.assertFalse(result.data_ready)
        self.assertEqual(result.reason, "processing_fps_too_low")

    def test_missing_values_stay_none_in_feature_dict(self):
        tracker = self.make_tracker()
        result = tracker.snapshot(captured_at=0.0)
        features = result.feature_dict()
        self.assertIsNone(features["ocular_perclos"])
        self.assertIsNone(features["ocular_blink_duration_ms"])
        self.assertIsNone(features["ocular_eye_opening_ratio"])


    def test_open_reference_uses_typical_value_not_upper_tail(self):
        tracker = self.make_tracker(
            calibration_min_samples=7,
            calibration_min_sec=0.3,
        )
        values = [0.30, 0.31, 0.29, 0.30, 0.10, 0.52, 0.30]
        current = 0.0
        for value in values:
            self.update(tracker, value, value * 1.1, current)
            current += 0.05

        self.assertTrue(tracker.calibrated)
        self.assertAlmostEqual(tracker.left_open_eye_reference, 0.30, places=2)
        self.assertAlmostEqual(tracker.right_open_eye_reference, 0.33, places=2)

    def test_bilateral_blink_count_includes_shallow_confirmed_events(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))

        # Глубокое двустороннее моргание.
        self.update(tracker, 0.10, 0.11, current)
        self.update(tracker, 0.09, 0.10, current + 0.05)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        self.update(tracker, 0.30, 0.33, current + 0.15)

        # Двустороннее событие, у которого камера не зафиксировала
        # глубокий минимум. Оно всё равно является моргательным событием.
        self.update(tracker, 0.235, 0.258, current + 0.30)
        self.update(tracker, 0.232, 0.255, current + 0.35)
        self.update(tracker, 0.30, 0.33, current + 0.40)
        result = self.update(tracker, 0.30, 0.33, current + 0.45)

        self.assertEqual(result.blink_count, 1)
        self.assertEqual(result.incomplete_blink_count, 1)
        self.assertEqual(result.bilateral_blink_count, 2)

    def test_natural_blink_can_be_full_without_reaching_perclos_threshold(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))

        # Минимум около 0.67 от личного открытого состояния. Это обычное
        # двустороннее моргание, хотя отдельный строгий порог PERCLOS 0.55
        # не был пересечён.
        self.update(tracker, 0.20, 0.22, current)
        self.update(tracker, 0.19, 0.21, current + 0.05)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        result = self.update(tracker, 0.30, 0.33, current + 0.15)

        self.assertEqual(result.blink_count, 1)
        self.assertEqual(result.incomplete_blink_count, 0)

    def test_shallower_bilateral_event_remains_incomplete(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))

        # Минимум около 0.78: оба глаза синхронно прикрылись, но глубины
        # недостаточно для полного моргания по новой границе 0.75.
        self.update(tracker, 0.235, 0.258, current)
        self.update(tracker, 0.232, 0.255, current + 0.05)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        result = self.update(tracker, 0.30, 0.33, current + 0.15)

        self.assertEqual(result.blink_count, 0)
        self.assertEqual(result.incomplete_blink_count, 1)

    def test_reset_can_keep_personal_calibration(self):
        tracker = self.make_tracker()
        self.calibrate(tracker)
        tracker.reset(keep_calibration=True)
        self.assertTrue(tracker.calibrated)
        self.assertEqual(tracker.snapshot(captured_at=1.0).observation_sec, 0.0)

    def test_event_history_exposes_anonymized_event_geometry(self):
        tracker = self.make_tracker()
        current = self.begin_window(tracker, self.calibrate(tracker))
        self.update(tracker, 0.10, 0.11, current)
        self.update(tracker, 0.09, 0.10, current + 0.05)
        self.update(tracker, 0.30, 0.33, current + 0.10)
        self.update(tracker, 0.30, 0.33, current + 0.15)

        events = tracker.event_history()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, OcularEventType.BILATERAL_BLINK)
        self.assertLess(events[0].min_left_normalized, 0.5)
        self.assertLess(events[0].min_right_normalized, 0.5)
        self.assertGreater(events[0].duration_sec, 0.0)

    def test_reference_can_be_frozen_after_calibration(self):
        tracker = self.make_tracker(update_reference_after_calibration=False)
        current = self.calibrate(tracker, left=0.30, right=0.33)
        left_before = tracker.left_open_eye_reference
        right_before = tracker.right_open_eye_reference

        for index in range(20):
            self.update(
                tracker,
                0.45,
                0.50,
                current + index * 0.05,
            )

        self.assertEqual(tracker.left_open_eye_reference, left_before)
        self.assertEqual(tracker.right_open_eye_reference, right_before)


if __name__ == "__main__":
    unittest.main()
