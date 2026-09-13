from __future__ import annotations

import unittest

from app_core.recovery_engine import (
    AdaptiveRecoveryEngine,
    AssessmentReliability,
    RecommendationKind,
    RecoverySignals,
    TypingDeviation,
    WorkloadLevel,
)
from app_core.state_machine import UserState


def signals(
    captured_at: float,
    *,
    state: UserState = UserState.ACTIVE_WORK,
    baseline_ready: bool = False,
    deviations: dict[str, TypingDeviation] | None = None,
    ocular_quality: float = 0.9,
    ocular_coverage: float = 0.95,
) -> RecoverySignals:
    return RecoverySignals(
        captured_at=captured_at,
        state=state,
        fatigue_analysis_allowed=state in {
            UserState.ACTIVE_WORK,
            UserState.PASSIVE_WORK,
        },
        typing_data_ready=True,
        typing_baseline_ready=baseline_ready,
        typing_calibration_progress=1.0 if baseline_ready else 0.2,
        typing_deviations=deviations or {},
        ocular_calibrated=True,
        ocular_data_ready=True,
        ocular_quality=ocular_quality,
        ocular_signal_coverage=ocular_coverage,
        ocular_blink_rate_per_min=15.0,
        ocular_perclos=0.02,
        ocular_long_closure_count=0,
        ocular_severe_closure_detected=False,
    )


class AdaptiveRecoveryEngineTests(unittest.TestCase):
    def test_time_based_eye_rest_is_explainable_fallback(self) -> None:
        engine = AdaptiveRecoveryEngine(
            eye_rest_after_sec=10.0,
            microbreak_after_sec=20.0,
            recovery_break_after_sec=30.0,
            meaningful_break_sec=5.0,
            alert_cooldown_sec=0.0,
        )
        engine.update(signals(0.0))
        engine.update(signals(11.0))
        assessment = engine.update(signals(16.0))
        self.assertIsNotNone(assessment.recommendation)
        assert assessment.recommendation is not None
        self.assertEqual(assessment.recommendation.kind, RecommendationKind.EYE_REST)
        self.assertIn("длительность", assessment.decision_basis)

    def test_meaningful_break_resets_continuous_work_and_level(self) -> None:
        engine = AdaptiveRecoveryEngine(
            eye_rest_after_sec=10.0,
            microbreak_after_sec=20.0,
            recovery_break_after_sec=30.0,
            meaningful_break_sec=5.0,
            level_rise_confirmations=1,
        )
        engine.update(signals(0.0))
        engine.update(signals(5.0))
        assessment = engine.update(signals(10.0))
        self.assertGreaterEqual(engine.continuous_work_sec, 10.0)
        self.assertGreaterEqual(assessment.workload_level, WorkloadLevel.EARLY)
        engine.update(signals(15.0, state=UserState.BREAK))
        assessment = engine.update(signals(20.0, state=UserState.BREAK))
        self.assertEqual(engine.continuous_work_sec, 0.0)
        self.assertEqual(assessment.workload_level, WorkloadLevel.STABLE)

    def test_typing_requires_persistence_and_personal_baseline(self) -> None:
        deviations = {
            "typing_keys_per_minute": TypingDeviation(
                "typing_keys_per_minute", 100.0, 180.0, -2.0, True, 1.0
            ),
            "typing_correction_ratio": TypingDeviation(
                "typing_correction_ratio", 0.20, 0.05, 2.0, True, 1.0
            ),
        }
        engine = AdaptiveRecoveryEngine(
            eye_rest_after_sec=100.0,
            microbreak_after_sec=200.0,
            recovery_break_after_sec=300.0,
            meaningful_break_sec=5.0,
            alert_cooldown_sec=0.0,
            typing_evaluation_interval_sec=1.0,
            typing_persistence_samples=3,
            typing_required_positive_samples=2,
            level_rise_confirmations=1,
        )
        engine.update(signals(0.0, baseline_ready=True, deviations=deviations))
        engine.update(signals(5.0, baseline_ready=True, deviations=deviations))
        assessment = engine.update(signals(10.0, baseline_ready=True, deviations=deviations))
        codes = {item.code for item in assessment.evidence}
        self.assertIn("typing_slowdown", codes)
        self.assertIn("typing_more_corrections", codes)
        self.assertEqual(assessment.workload_level, WorkloadLevel.SUSTAINED)

    def test_ocular_channel_is_diagnostic_by_default(self) -> None:
        engine = AdaptiveRecoveryEngine(
            eye_rest_after_sec=100.0,
            microbreak_after_sec=200.0,
            recovery_break_after_sec=300.0,
        )
        assessment = engine.update(signals(0.0))
        ocular = [item for item in assessment.evidence if item.code == "ocular_diagnostic_available"]
        self.assertEqual(len(ocular), 1)
        self.assertFalse(ocular[0].decision_ready)
        self.assertFalse(assessment.ocular_used_for_decision)

    def test_very_high_self_report_is_immediate_priority(self) -> None:
        engine = AdaptiveRecoveryEngine(
            eye_rest_after_sec=100.0,
            microbreak_after_sec=200.0,
            recovery_break_after_sec=300.0,
            alert_cooldown_sec=0.0,
        )
        engine.record_self_report(
            fatigue_sp_1_7=7.0,
            sleepiness_kss_1_9=6.0,
            captured_at=0.0,
        )
        assessment = engine.update(signals(1.0))
        self.assertEqual(assessment.workload_level, WorkloadLevel.RECOVERY_PRIORITY)
        self.assertIsNotNone(assessment.recommendation)
        assert assessment.recommendation is not None
        self.assertEqual(assessment.recommendation.kind, RecommendationKind.RECOVERY_BREAK)

    def test_strong_self_report_requires_hysteresis_before_level_four(self) -> None:
        engine = AdaptiveRecoveryEngine(
            eye_rest_after_sec=100.0,
            microbreak_after_sec=200.0,
            recovery_break_after_sec=300.0,
            alert_cooldown_sec=0.0,
            level_rise_confirmations=3,
        )
        engine.record_self_report(
            fatigue_sp_1_7=6.0,
            sleepiness_kss_1_9=5.0,
            captured_at=0.0,
        )
        first = engine.update(signals(1.0))
        second = engine.update(signals(2.0))
        third = engine.update(signals(3.0))
        self.assertEqual(first.workload_level, WorkloadLevel.STABLE)
        self.assertEqual(second.workload_level, WorkloadLevel.STABLE)
        self.assertEqual(third.workload_level, WorkloadLevel.EXPRESSED)

    def test_reliability_is_separate_from_workload_level(self) -> None:
        engine = AdaptiveRecoveryEngine(
            eye_rest_after_sec=100.0,
            microbreak_after_sec=200.0,
            recovery_break_after_sec=300.0,
        )
        assessment = engine.update(signals(0.0, baseline_ready=True))
        self.assertEqual(assessment.workload_level, WorkloadLevel.STABLE)
        self.assertEqual(assessment.reliability, AssessmentReliability.HIGH)

    def test_only_five_workload_levels_exist(self) -> None:
        self.assertEqual([int(level) for level in WorkloadLevel], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
