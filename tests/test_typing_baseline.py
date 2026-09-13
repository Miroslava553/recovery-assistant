import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from app_core.baseline_engine import PersonalBaselineEngine, WorkContext
from app_core.state_machine import UserState
from app_core.typing_activity import TypingMetrics
from app_core.typing_baseline import TypingBaselineService


class TypingBaselineServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def metrics(
        *,
        observation_sec: float = 60.0,
        relevant_key_count: int = 60,
        keys_per_minute: float = 60.0,
        data_ready: bool = True,
        long_pause_count: int = 2,
        burst_count: int = 3,
    ) -> TypingMetrics:
        return TypingMetrics(
            observation_sec=observation_sec,
            data_ready=data_ready,
            text_key_count=max(0, relevant_key_count - 3),
            correction_key_count=min(3, relevant_key_count),
            relevant_key_count=relevant_key_count,
            keys_per_minute=keys_per_minute,
            mean_interval_sec=0.50,
            median_interval_sec=0.40,
            interval_std_sec=0.10,
            rhythm_cv=0.20,
            longest_pause_sec=3.0,
            long_pause_count=long_pause_count,
            burst_count=burst_count,
            correction_ratio=(
                min(3, relevant_key_count) / relevant_key_count
                if relevant_key_count
                else 0.0
            ),
            current_idle_sec=0.5,
        )

    def make_service(
        self,
        *,
        profile_id: int = 1,
    ) -> TypingBaselineService:
        baseline = PersonalBaselineEngine(
            calibration_days=5,
            min_windows_per_feature=5,
            min_samples_for_stats=3,
            max_windows_per_feature=100,
        )
        return TypingBaselineService(
            self.db_path,
            profile_id=profile_id,
            baseline=baseline,
            sample_interval_sec=60.0,
            min_observation_sec=50.0,
            target_relevant_keys=25,
        )

    def test_metrics_are_converted_to_rates(self):
        features = TypingBaselineService.features_from_metrics(
            self.metrics(
                observation_sec=60.0,
                long_pause_count=2,
                burst_count=3,
            )
        )

        self.assertEqual(features["typing_long_pause_rate_per_min"], 2.0)
        self.assertEqual(features["typing_burst_rate_per_min"], 3.0)
        self.assertEqual(features["typing_mean_interval_sec"], 0.50)

    def test_passive_work_does_not_create_zero_typing_baseline(self):
        service = self.make_service()

        result = service.observe(
            self.metrics(),
            state=UserState.PASSIVE_WORK,
            workday=date(2026, 7, 1),
            minutes_since_workday_start=30.0,
            now_monotonic=60.0,
        )

        self.assertFalse(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual(service.stored_sample_count(), 0)

    def test_short_or_sparse_window_is_rejected(self):
        service = self.make_service()

        short = service.observe(
            self.metrics(observation_sec=20.0),
            state=UserState.ACTIVE_WORK,
            workday=date(2026, 7, 1),
            minutes_since_workday_start=30.0,
            now_monotonic=60.0,
        )

        sparse = service.observe(
            self.metrics(
                observation_sec=60.0,
                relevant_key_count=10,
                data_ready=False,
            ),
            state=UserState.ACTIVE_WORK,
            workday=date(2026, 7, 1),
            minutes_since_workday_start=30.0,
            now_monotonic=120.0,
        )

        self.assertFalse(short.accepted)
        self.assertFalse(sparse.accepted)
        self.assertEqual(service.stored_sample_count(), 0)

    def test_only_one_overlapping_sample_per_minute(self):
        service = self.make_service()

        first = service.observe(
            self.metrics(),
            state=UserState.ACTIVE_WORK,
            workday=date(2026, 7, 1),
            minutes_since_workday_start=30.0,
            captured_at=datetime(2026, 7, 1, 9, 1, tzinfo=timezone.utc),
            now_monotonic=60.0,
        )
        second = service.observe(
            self.metrics(),
            state=UserState.ACTIVE_WORK,
            workday=date(2026, 7, 1),
            minutes_since_workday_start=31.0,
            captured_at=datetime(2026, 7, 1, 9, 1, 10, tzinfo=timezone.utc),
            now_monotonic=70.0,
        )

        self.assertTrue(first.accepted)
        self.assertFalse(second.attempted)
        self.assertEqual(service.stored_sample_count(), 7)

    def test_samples_survive_restart_and_complete_five_day_calibration(self):
        start_day = date(2026, 7, 1)

        for day_offset in range(5):
            service = self.make_service()
            day = start_day + timedelta(days=day_offset)
            result = service.observe(
                self.metrics(keys_per_minute=200.0 + day_offset),
                state=UserState.ACTIVE_WORK,
                workday=day,
                minutes_since_workday_start=30.0,
                captured_at=datetime(
                    2026,
                    7,
                    1 + day_offset,
                    9,
                    1,
                    tzinfo=timezone.utc,
                ),
                now_monotonic=60.0,
            )
            self.assertTrue(result.accepted)

        restored = self.make_service()
        stats = restored.baseline.stats(
            WorkContext.TYPING,
            "typing_keys_per_minute",
        )

        self.assertIsNotNone(stats)
        self.assertTrue(restored.baseline.initial_calibration_complete)
        self.assertEqual(restored.baseline.calibration_day_count, 5)
        self.assertEqual(stats.day_count, 5)
        self.assertEqual(stats.sample_count, 5)
        self.assertEqual(stats.median, 202.0)

    def test_different_profiles_do_not_share_norm(self):
        first_profile = self.make_service(profile_id=1)
        second_profile = self.make_service(profile_id=2)

        first_profile.observe(
            self.metrics(keys_per_minute=240.0),
            state=UserState.ACTIVE_WORK,
            workday=date(2026, 7, 1),
            minutes_since_workday_start=30.0,
            captured_at=datetime(2026, 7, 1, 9, 1, tzinfo=timezone.utc),
            now_monotonic=60.0,
        )

        self.assertEqual(first_profile.stored_sample_count(), 7)
        self.assertEqual(second_profile.stored_sample_count(), 0)


if __name__ == "__main__":
    unittest.main()