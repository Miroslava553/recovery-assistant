import tempfile
import unittest
from pathlib import Path

from app_core.ocular_activity import OcularEventType
from app_core.ocular_validation import (
    EyeSampleFailureReason,
    EyeSampleQualityThresholds,
    ManualBlinkLabel,
    NormalizedValidationSample,
    OfflineBlinkConfig,
    ValidationDataError,
    ValidationLabelKind,
    ValidationSample,
    apply_personal_head_pose_quality,
    assess_eye_sample_quality,
    calculate_time_weighted_coverage,
    calibrated_head_pose_quality,
    calculate_validation_metrics,
    detect_offline_blinks,
    estimate_head_pose_reference,
    estimate_open_eye_reference,
    load_labels,
    load_samples,
    match_bilateral_events,
    normalize_samples,
    optimize_offline_thresholds,
    save_labels,
    save_samples,
    summarize_eye_sample_quality,
)


class OcularValidationTests(unittest.TestCase):
    @staticmethod
    def raw_sample(
        frame_index,
        timestamp,
        left=0.30,
        right=0.33,
        *,
        valid=True,
    ):
        return ValidationSample(
            frame_index=frame_index,
            timestamp_sec=timestamp,
            left_eye_opening_raw=left if valid else None,
            right_eye_opening_raw=right if valid else None,
            data_quality=0.9 if valid else 0.0,
            left_eye_quality=0.9 if valid else 0.0,
            right_eye_quality=0.9 if valid else 0.0,
            head_pose_quality=0.9 if valid else 0.0,
            face_detected=valid,
            valid=valid,
        )

    @staticmethod
    def normalized_sample(
        frame_index,
        timestamp,
        left=1.0,
        right=1.0,
        *,
        valid=True,
    ):
        return NormalizedValidationSample(
            frame_index=frame_index,
            timestamp_sec=timestamp,
            left_eye_opening_raw=left,
            right_eye_opening_raw=right,
            left_eye_opening_normalized=left if valid else None,
            right_eye_opening_normalized=right if valid else None,
            data_quality=0.9 if valid else 0.0,
            left_eye_quality=0.9 if valid else 0.0,
            right_eye_quality=0.9 if valid else 0.0,
            head_pose_quality=0.9 if valid else 0.0,
            face_detected=valid,
            valid=valid,
        )

    def blink_sequence(self, centers_and_depths):
        samples = []
        frame = 0
        timestamp = 0.0
        center_map = {round(center, 2): depth for center, depth in centers_and_depths}
        while timestamp <= max(center_map, default=0.0) + 0.50:
            left = right = 1.0
            for center, depth in centers_and_depths:
                delta = round(timestamp - center, 2)
                if abs(delta) < 0.001:
                    left = depth
                    right = depth + 0.01
                elif abs(abs(delta) - 0.05) < 0.001:
                    left = min(0.89, depth + 0.18)
                    right = min(0.90, depth + 0.18)
            samples.append(self.normalized_sample(frame, timestamp, left, right))
            frame += 1
            timestamp = round(timestamp + 0.05, 2)
        return samples

    def test_personal_head_pose_reference_accepts_nonstandard_neutral_geometry(self):
        samples = []
        for index in range(120):
            blink = index in {40, 41}
            samples.append(
                ValidationSample(
                    frame_index=index,
                    timestamp_sec=index / 30.0,
                    left_eye_opening_raw=0.08 if blink else 0.31,
                    right_eye_opening_raw=0.09 if blink else 0.34,
                    data_quality=0.90,
                    left_eye_quality=0.90,
                    right_eye_quality=0.90,
                    # Старая абсолютная формула могла считать такой ракурс
                    # плохим, хотя пользователь сидит в своей нейтральной позе.
                    head_pose_quality=0.20,
                    face_detected=True,
                    valid=False,
                    head_turn_ratio=0.04 + (0.002 if index % 2 else -0.002),
                    head_vertical_ratio=0.78 + (0.003 if index % 3 else -0.003),
                )
            )

        reference = estimate_head_pose_reference(
            samples,
            calibration_duration_sec=4.0,
        )
        personalized = apply_personal_head_pose_quality(
            samples,
            reference=reference,
        )

        self.assertAlmostEqual(reference.neutral_turn_ratio, 0.04, delta=0.005)
        self.assertAlmostEqual(reference.neutral_vertical_ratio, 0.78, delta=0.01)
        self.assertGreater(reference.sample_count, 100)
        self.assertTrue(personalized[10].valid)
        self.assertGreater(personalized[10].head_pose_quality, 0.95)

    def test_personal_head_pose_quality_rejects_strong_deviation(self):
        samples = [
            ValidationSample(
                frame_index=index,
                timestamp_sec=index / 30.0,
                left_eye_opening_raw=0.31,
                right_eye_opening_raw=0.34,
                data_quality=0.90,
                left_eye_quality=0.90,
                right_eye_quality=0.90,
                head_pose_quality=0.90,
                face_detected=True,
                valid=True,
                head_turn_ratio=0.02,
                head_vertical_ratio=0.70,
            )
            for index in range(100)
        ]
        reference = estimate_head_pose_reference(
            samples,
            calibration_duration_sec=3.3,
        )

        neutral = calibrated_head_pose_quality(
            head_turn_ratio=0.03,
            head_vertical_ratio=0.72,
            reference=reference,
        )
        severe = calibrated_head_pose_quality(
            head_turn_ratio=0.36,
            head_vertical_ratio=0.70,
            reference=reference,
        )

        self.assertGreater(neutral, 0.9)
        self.assertLess(severe, 0.35)

    def test_quality_assessment_lists_all_failed_criteria(self):
        assessment = assess_eye_sample_quality(
            face_detected=True,
            left_eye_opening_raw=0.30,
            right_eye_opening_raw=None,
            data_quality=0.40,
            left_eye_quality=0.90,
            right_eye_quality=0.30,
            head_pose_quality=0.20,
        )

        self.assertFalse(assessment.valid)
        self.assertIn(
            EyeSampleFailureReason.IMAGE_QUALITY_LOW_OR_INVALID,
            assessment.failed_reasons,
        )
        self.assertIn(
            EyeSampleFailureReason.RIGHT_EYE_QUALITY_LOW_OR_INVALID,
            assessment.failed_reasons,
        )
        self.assertIn(
            EyeSampleFailureReason.HEAD_POSE_QUALITY_LOW_OR_INVALID,
            assessment.failed_reasons,
        )
        self.assertIn(
            EyeSampleFailureReason.RIGHT_EYE_SIGNAL_MISSING_OR_INVALID,
            assessment.failed_reasons,
        )

    def test_quality_summary_explains_low_head_pose_coverage(self):
        samples = []
        for index in range(10):
            samples.append(
                ValidationSample(
                    frame_index=index,
                    timestamp_sec=index * 0.1,
                    left_eye_opening_raw=0.30,
                    right_eye_opening_raw=0.33,
                    data_quality=0.90,
                    left_eye_quality=0.90,
                    right_eye_quality=0.90,
                    head_pose_quality=0.20 if index < 7 else 0.90,
                    face_detected=True,
                    valid=index >= 7,
                )
            )

        summary = summarize_eye_sample_quality(
            samples,
            thresholds=EyeSampleQualityThresholds(),
        )

        self.assertEqual(summary["total_frame_count"], 10)
        self.assertEqual(summary["valid_frame_count"], 3)
        self.assertAlmostEqual(summary["frame_coverage"], 0.3)
        self.assertEqual(
            summary["reason_counts"][
                EyeSampleFailureReason.HEAD_POSE_QUALITY_LOW_OR_INVALID.value
            ],
            7,
        )
        self.assertEqual(
            summary["dominant_primary_reason"],
            EyeSampleFailureReason.HEAD_POSE_QUALITY_LOW_OR_INVALID.value,
        )

    def test_quality_summary_can_exclude_calibration(self):
        samples = [
            ValidationSample(
                frame_index=0,
                timestamp_sec=0.0,
                left_eye_opening_raw=None,
                right_eye_opening_raw=None,
                data_quality=0.0,
                left_eye_quality=0.0,
                right_eye_quality=0.0,
                head_pose_quality=0.0,
                face_detected=False,
                valid=False,
            ),
            self.raw_sample(1, 5.0),
            self.raw_sample(2, 5.1),
        ]

        summary = summarize_eye_sample_quality(samples, start_after_sec=5.0)

        self.assertEqual(summary["total_frame_count"], 2)
        self.assertEqual(summary["valid_frame_count"], 2)
        self.assertAlmostEqual(summary["frame_coverage"], 1.0)

    def test_reference_uses_median_and_ignores_short_blink(self):
        samples = []
        for index in range(100):
            left = 0.10 if index in {25, 26} else 0.30
            right = 0.11 if index in {25, 26} else 0.33
            samples.append(self.raw_sample(index, index * 0.05, left, right))

        reference = estimate_open_eye_reference(samples, calibration_sec=4.95)

        self.assertAlmostEqual(reference.left_open_eye_reference, 0.30, places=4)
        self.assertAlmostEqual(reference.right_open_eye_reference, 0.33, places=4)
        self.assertEqual(reference.valid_sample_count, 100)

    def test_reference_rejects_too_few_valid_samples(self):
        samples = [self.raw_sample(index, index * 0.1, valid=index < 5) for index in range(40)]
        with self.assertRaises(ValidationDataError):
            estimate_open_eye_reference(samples, calibration_sec=3.9)

    def test_detector_finds_bilateral_blink_and_separates_wink(self):
        samples = self.blink_sequence([(1.0, 0.45)])
        # Добавляем левое подмигивание позже.
        for index, sample in enumerate(samples):
            if abs(sample.timestamp_sec - 1.35) < 0.001:
                samples[index] = self.normalized_sample(
                    sample.frame_index,
                    sample.timestamp_sec,
                    0.40,
                    1.0,
                )
            elif abs(sample.timestamp_sec - 1.30) < 0.001:
                samples[index] = self.normalized_sample(
                    sample.frame_index,
                    sample.timestamp_sec,
                    0.88,
                    1.0,
                )

        events = detect_offline_blinks(
            samples,
            OfflineBlinkConfig(smoothing_alpha=1.0),
        )

        self.assertEqual(
            sum(event.event_type is OcularEventType.BILATERAL_BLINK for event in events),
            1,
        )
        self.assertEqual(
            sum(event.event_type is OcularEventType.LEFT_WINK for event in events),
            1,
        )

    def test_invalid_gap_cancels_candidate(self):
        samples = [
            self.normalized_sample(0, 0.00),
            self.normalized_sample(1, 0.05, 0.80, 0.81),
            self.normalized_sample(2, 0.30, valid=False),
            self.normalized_sample(3, 0.35),
            self.normalized_sample(4, 0.40),
        ]
        events = detect_offline_blinks(
            samples,
            OfflineBlinkConfig(smoothing_alpha=1.0),
        )
        self.assertEqual(events, [])

    def test_matching_is_one_to_one(self):
        samples = self.blink_sequence([(1.0, 0.45), (1.5, 0.48)])
        events = detect_offline_blinks(
            samples,
            OfflineBlinkConfig(smoothing_alpha=1.0),
        )
        labels = [
            ManualBlinkLabel(20, 1.01, ValidationLabelKind.FULL_BLINK),
            ManualBlinkLabel(30, 1.49, ValidationLabelKind.FULL_BLINK),
        ]
        matched, false_positives, missed = match_bilateral_events(events, labels)
        self.assertEqual(len(matched), 2)
        self.assertEqual(false_positives, [])
        self.assertEqual(missed, [])

    def test_metrics_report_false_positive_and_miss(self):
        samples = self.blink_sequence([(1.0, 0.45), (1.5, 0.48)])
        events = detect_offline_blinks(
            samples,
            OfflineBlinkConfig(smoothing_alpha=1.0),
        )
        labels = [
            ManualBlinkLabel(20, 1.0, ValidationLabelKind.FULL_BLINK),
            ManualBlinkLabel(40, 2.0, ValidationLabelKind.FULL_BLINK),
        ]
        metrics = calculate_validation_metrics(events, labels, tolerance_sec=0.2)
        self.assertEqual(metrics.true_positive_count, 1)
        self.assertEqual(metrics.false_positive_count, 1)
        self.assertEqual(metrics.false_negative_count, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_optimizer_finds_perfect_detection_on_clean_signal(self):
        centers_and_depths = [
            (5.50, 0.48),
            (6.20, 0.50),
            (6.90, 0.46),
            (7.60, 0.86),
            (8.30, 0.85),
            (9.00, 0.87),
        ]
        samples = self.blink_sequence(centers_and_depths)
        labels = [
            ManualBlinkLabel(
                int(center / 0.05),
                center,
                ValidationLabelKind.FULL_BLINK if depth < 0.70 else ValidationLabelKind.INCOMPLETE_BLINK,
            )
            for center, depth in centers_and_depths
        ]

        result = optimize_offline_thresholds(
            samples,
            labels,
            calibration_sec=5.0,
            min_bilateral_labels=5,
        )

        self.assertAlmostEqual(result.metrics.precision, 1.0)
        self.assertAlmostEqual(result.metrics.recall, 1.0)
        self.assertTrue(result.classification_threshold_validated)
        self.assertIsNotNone(result.classification_balanced_accuracy)

    def test_save_and_load_samples_and_labels(self):
        samples = [
            self.raw_sample(0, 0.0),
            self.raw_sample(1, 0.05, valid=False),
        ]
        labels = [
            ManualBlinkLabel(0, 0.0, ValidationLabelKind.FULL_BLINK),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_samples(root / "signals.csv", samples)
            save_labels(root / "labels.json", labels)
            loaded_samples = load_samples(root / "signals.csv")
            loaded_labels = load_labels(root / "labels.json")

        self.assertEqual(loaded_samples, samples)
        self.assertEqual(loaded_labels, labels)

    def test_time_weighted_coverage_ignores_large_camera_gap(self):
        samples = [
            self.normalized_sample(0, 0.00, valid=True),
            self.normalized_sample(1, 0.05, valid=True),
            self.normalized_sample(2, 0.10, valid=False),
            self.normalized_sample(3, 0.15, valid=False),
            # Разрыв 1.85 с не должен приписываться предыдущему кадру.
            self.normalized_sample(4, 2.00, valid=True),
            self.normalized_sample(5, 2.05, valid=True),
        ]
        coverage = calculate_time_weighted_coverage(samples)
        self.assertAlmostEqual(coverage, 0.75, places=6)

    def test_normalization_preserves_invalid_samples_as_none(self):
        samples = [self.raw_sample(index, index * 0.05) for index in range(80)]
        samples.append(self.raw_sample(80, 4.0, valid=False))
        reference = estimate_open_eye_reference(samples, calibration_sec=3.5)
        normalized = normalize_samples(samples, reference)
        self.assertAlmostEqual(normalized[0].left_eye_opening_normalized, 1.0)
        self.assertIsNone(normalized[-1].left_eye_opening_normalized)
        self.assertFalse(normalized[-1].valid)


if __name__ == "__main__":
    unittest.main()
