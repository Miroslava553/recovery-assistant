import unittest
from datetime import date, timedelta

from app_core.baseline_engine import (
    BaselineObservation,
    FeatureChannel,
    PersonalBaselineEngine,
    WorkContext,
)


class PersonalBaselineEngineTests(unittest.TestCase):
    def make_engine(self) -> PersonalBaselineEngine:
        return PersonalBaselineEngine(
            calibration_days=5,
            baseline_window_minutes=120.0,
            min_windows_per_feature=5,
            min_samples_for_stats=3,
            max_windows_per_feature=50,
        )

    @staticmethod
    def observation(
        *,
        day_offset: int = 0,
        context: WorkContext = WorkContext.TYPING,
        features=None,
        minutes: float = 30.0,
        user_present: bool = True,
        returning: bool = False,
        severe_ocular_event: bool = False,
        fatigue: float | None = None,
        sleepiness: float | None = None,
        typing_quality: float = 0.95,
        ocular_quality: float = 0.95,
        liveness: float = 0.95,
    ) -> BaselineObservation:
        if features is None:
            features = {
                "typing_keys_per_minute": 220.0,
            }

        return BaselineObservation(
            workday=date(2026, 7, 1) + timedelta(days=day_offset),
            minutes_since_workday_start=minutes,
            context=context,
            features=features,
            user_present=user_present,
            returning=returning,
            severe_ocular_event=severe_ocular_event,
            self_report_fatigue_0_10=fatigue,
            self_report_sleepiness_kss_1_9=sleepiness,
            channel_quality={
                FeatureChannel.TYPING: typing_quality,
                FeatureChannel.OCULAR: ocular_quality,
                FeatureChannel.MOUSE: 0.95,
                FeatureChannel.POSTURE: 0.95,
                FeatureChannel.GENERIC: 0.95,
            },
            liveness_score=liveness,
        )

    def test_personal_contexts_are_separate(self):
        engine = self.make_engine()

        for index, value in enumerate([200, 205, 210]):
            engine.observe(
                self.observation(
                    day_offset=index,
                    context=WorkContext.TYPING,
                    features={"typing_keys_per_minute": value},
                )
            )

        for index, value in enumerate([80, 85, 90]):
            engine.observe(
                self.observation(
                    day_offset=index,
                    context=WorkContext.READING,
                    features={"typing_keys_per_minute": value},
                )
            )

        typing_stats = engine.stats(
            WorkContext.TYPING,
            "typing_keys_per_minute",
        )
        reading_stats = engine.stats(
            WorkContext.READING,
            "typing_keys_per_minute",
        )

        self.assertIsNotNone(typing_stats)
        self.assertIsNotNone(reading_stats)
        self.assertEqual(typing_stats.median, 205.0)
        self.assertEqual(reading_stats.median, 85.0)

    def test_calibration_requires_five_distinct_days(self):
        engine = self.make_engine()

        for day_offset in range(4):
            engine.observe(self.observation(day_offset=day_offset))

        self.assertFalse(engine.initial_calibration_complete)
        self.assertEqual(engine.calibration_day_count, 4)

        engine.observe(self.observation(day_offset=4))

        self.assertTrue(engine.initial_calibration_complete)
        self.assertEqual(engine.calibration_day_count, 5)

    def test_same_day_does_not_count_twice(self):
        engine = self.make_engine()

        engine.observe(self.observation(day_offset=0))
        engine.observe(
            self.observation(
                day_offset=0,
                features={"typing_correction_ratio": 0.05},
            )
        )

        self.assertEqual(engine.calibration_day_count, 1)

    def test_rejects_window_after_first_two_hours(self):
        engine = self.make_engine()

        result = engine.observe(
            self.observation(minutes=121.0)
        )

        self.assertFalse(result.accepted)
        self.assertIn("двух часов", result.reason)

    def test_rejects_returning_and_tired_windows(self):
        engine = self.make_engine()

        returning = engine.observe(
            self.observation(returning=True)
        )
        fatigued = engine.observe(
            self.observation(fatigue=6.0)
        )
        sleepy = engine.observe(
            self.observation(sleepiness=7.0)
        )

        self.assertFalse(returning.accepted)
        self.assertFalse(fatigued.accepted)
        self.assertFalse(sleepy.accepted)

    def test_visual_feature_requires_liveness(self):
        engine = self.make_engine()

        result = engine.observe(
            self.observation(
                features={"ocular_perclos": 0.08},
                liveness=0.40,
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn(
            "Liveness",
            result.rejected_features["ocular_perclos"],
        )

    def test_typing_feature_does_not_require_camera_liveness(self):
        engine = self.make_engine()

        result = engine.observe(
            self.observation(
                features={"typing_keys_per_minute": 220.0},
                liveness=0.10,
            )
        )

        self.assertTrue(result.accepted)

    def test_missing_feature_is_not_saved_as_zero(self):
        engine = self.make_engine()

        result = engine.observe(
            self.observation(
                features={
                    "typing_keys_per_minute": None,
                    "typing_correction_ratio": 0.05,
                }
            )
        )

        self.assertEqual(
            engine.sample_count(
                WorkContext.TYPING,
                "typing_keys_per_minute",
            ),
            0,
        )
        self.assertEqual(
            engine.sample_count(
                WorkContext.TYPING,
                "typing_correction_ratio",
            ),
            1,
        )
        self.assertIn(
            "typing_keys_per_minute",
            result.rejected_features,
        )

    def test_zero_mad_uses_safe_feature_scale(self):
        engine = self.make_engine()

        for day_offset in range(3):
            engine.observe(
                self.observation(
                    day_offset=day_offset,
                    features={"typing_keys_per_minute": 220.0},
                )
            )

        stats = engine.stats(
            WorkContext.TYPING,
            "typing_keys_per_minute",
        )
        self.assertIsNotNone(stats)
        self.assertEqual(stats.mad, 0.0)
        self.assertGreaterEqual(stats.robust_scale, 10.0)

        comparison = engine.compare(
            WorkContext.TYPING,
            "typing_keys_per_minute",
            230.0,
        )
        self.assertAlmostEqual(comparison.robust_z, 1.0)

    def test_ready_requires_samples_and_days(self):
        engine = self.make_engine()

        for day_offset in range(5):
            engine.observe(
                self.observation(
                    day_offset=day_offset,
                    features={
                        "typing_keys_per_minute": 210.0 + day_offset,
                    },
                )
            )

        stats = engine.stats(
            WorkContext.TYPING,
            "typing_keys_per_minute",
        )
        self.assertIsNotNone(stats)
        self.assertTrue(stats.ready)
        self.assertEqual(stats.day_count, 5)
        self.assertEqual(stats.sample_count, 5)
        self.assertEqual(stats.confidence, 1.0)

    def test_after_calibration_refresh_must_be_explicit(self):
        engine = self.make_engine()

        for day_offset in range(5):
            engine.observe(self.observation(day_offset=day_offset))

        rejected = engine.observe(
            self.observation(day_offset=5)
        )
        accepted = engine.observe(
            self.observation(day_offset=5),
            allow_refresh=True,
        )

        self.assertFalse(rejected.accepted)
        self.assertTrue(accepted.accepted)

    def test_comparison_is_preliminary_during_calibration(self):
        engine = self.make_engine()

        for day_offset, value in enumerate([200.0, 210.0, 220.0]):
            engine.observe(
                self.observation(
                    day_offset=day_offset,
                    features={"typing_keys_per_minute": value},
                )
            )

        comparison = engine.compare(
            WorkContext.TYPING,
            "typing_keys_per_minute",
            190.0,
        )

        self.assertTrue(comparison.available)
        self.assertFalse(comparison.ready)
        self.assertLess(comparison.robust_z, 0.0)
        self.assertLess(comparison.confidence, 1.0)
        self.assertIn("предварительное", comparison.reason)


if __name__ == "__main__":
    unittest.main()