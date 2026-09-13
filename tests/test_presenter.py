from __future__ import annotations

import time
import unittest
from dataclasses import dataclass

from app_core.evidence import EvidenceSource, RecoveryEvidence
from app_core.recovery_engine import (
    WORKLOAD_SUMMARIES,
    WORKLOAD_TITLES,
    RecoveryAssessment,
    RecoveryRecommendation,
    RecommendationKind,
    RecoverySignals,
    WorkloadLevel,
)
from app_core.state_machine import UserState
from ui.presenter import build_main_screen_view

EYE_REST = 45 * 60.0
MICROBREAK = 70 * 60.0
RECOVERY = 105 * 60.0


@dataclass
class FakeOcular:
    signal_coverage: float = 0.94


@dataclass
class FakeSnapshot:
    face_detected: bool | None = True
    camera_status: str = "face_detected"
    seconds_in_state: float = 0.0
    ocular_metrics: FakeOcular = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ocular_metrics is None:
            self.ocular_metrics = FakeOcular()


def signals(**kwargs) -> RecoverySignals:
    defaults = dict(
        captured_at=time.monotonic(),
        state=UserState.ACTIVE_WORK,
        fatigue_analysis_allowed=True,
        typing_data_ready=True,
        typing_baseline_ready=True,
        typing_calibration_progress=1.0,
        typing_deviations={},
        ocular_calibrated=True,
        ocular_data_ready=True,
        ocular_quality=0.8,
        ocular_signal_coverage=0.94,
        ocular_blink_rate_per_min=14.0,
        ocular_perclos=0.1,
        ocular_long_closure_count=0,
        ocular_severe_closure_detected=False,
    )
    defaults.update(kwargs)
    return RecoverySignals(**defaults)


def assessment(
    *,
    level: WorkloadLevel = WorkloadLevel.STABLE,
    work_sec: float = 0.0,
    evidence: tuple[RecoveryEvidence, ...] = (),
    recommendation: RecoveryRecommendation | None = None,
    fatigue_sp: int | None = None,
    sleepiness_kss: int | None = None,
    ocular_used: bool = False,
    state: UserState = UserState.ACTIVE_WORK,
) -> RecoveryAssessment:
    return RecoveryAssessment(
        captured_at=time.monotonic(),
        state=state,
        status_label="",
        continuous_work_sec=work_sec,
        current_break_sec=0.0,
        typing_calibration_progress=1.0,
        evidence=evidence,
        recommendation=recommendation,
        recommendation_is_new=False,
        decision_basis="",
        ocular_used_for_decision=ocular_used,
        workload_level=level,
        workload_title=WORKLOAD_TITLES[level],
        workload_summary=WORKLOAD_SUMMARIES[level],
        self_report_fatigue_sp=fatigue_sp,
        self_report_sleepiness_kss=sleepiness_kss,
    )


def view(snapshot=None, sig=None, ass=None, **kwargs):
    return build_main_screen_view(
        snapshot or FakeSnapshot(),
        sig or signals(),
        ass or assessment(),
        now=time.monotonic(),
        eye_rest_after_sec=EYE_REST,
        microbreak_after_sec=MICROBREAK,
        recovery_break_after_sec=RECOVERY,
        **kwargs,
    )


class TimerTests(unittest.TestCase):
    def test_progress_resets_between_thresholds(self):
        """Полоска показывает путь до ближайшего порога, а не до последнего."""

        early = view(ass=assessment(work_sec=40 * 60))
        just_after = view(ass=assessment(work_sec=46 * 60))
        self.assertGreater(early.work_progress, 0.8)
        self.assertLess(just_after.work_progress, 0.2)

    def test_threshold_label_counts_down_in_minutes(self):
        self.assertIn("36 мин", view(ass=assessment(work_sec=10 * 60)).next_threshold_text)
        self.assertIn("21 мин", view(ass=assessment(work_sec=50 * 60)).next_threshold_text)

    def test_all_thresholds_passed(self):
        result = view(ass=assessment(work_sec=200 * 60))
        self.assertEqual(result.work_progress, 1.0)
        self.assertIn("рекомендован", result.next_threshold_text)

    def test_work_time_is_formatted(self):
        self.assertEqual(view(ass=assessment(work_sec=4663)).work_time_text, "01:17:43")


class CameraCardTests(unittest.TestCase):
    def test_good_coverage_is_green(self):
        result = view()
        self.assertEqual(result.signal_tone, "good")
        self.assertEqual(result.signal_value_text, "94%")
        self.assertEqual(result.signal_bars, 5)

    def test_partial_coverage_is_warning(self):
        result = view(snapshot=FakeSnapshot(ocular_metrics=FakeOcular(0.62)))
        self.assertEqual(result.signal_tone, "warn")

    def test_dark_room_gets_an_actionable_hint(self):
        result = view(
            snapshot=FakeSnapshot(
                face_detected=False, camera_status="too_dark", ocular_metrics=FakeOcular(0.2)
            )
        )
        self.assertEqual(result.signal_tone, "bad")
        self.assertIn("лампу", result.signal_hint)

    def test_no_camera_shows_no_percentage(self):
        result = view(vision_available=False)
        self.assertIsNone(result.signal_percent)
        self.assertEqual(result.signal_tone, "off")
        self.assertEqual(result.signal_bars, 0)

    def test_no_camera_changes_monitoring_badge(self):
        self.assertIn("без камеры", view(vision_available=False).monitoring_text)


class ReasonsTests(unittest.TestCase):
    def test_only_decision_ready_evidence_is_shown(self):
        """Глазной канал диагностический — он не должен выглядеть как основание."""

        result = view(
            ass=assessment(
                evidence=(
                    RecoveryEvidence(
                        code="long_continuous_work",
                        source=EvidenceSource.SESSION,
                        severity=2,
                        title="Длительная непрерывная работа",
                        detail="",
                        decision_ready=True,
                    ),
                    RecoveryEvidence(
                        code="ocular_diagnostic_available",
                        source=EvidenceSource.OCULAR,
                        severity=1,
                        title="Глазной сигнал доступен",
                        detail="",
                        decision_ready=False,
                    ),
                )
            )
        )
        self.assertEqual(len(result.reasons), 1)
        self.assertEqual(result.reasons[0].channel, "session")
        self.assertEqual(result.reasons[0].channel_label, "сессия")

    def test_empty_evidence_still_says_something(self):
        result = view()
        self.assertEqual(len(result.reasons), 1)
        self.assertTrue(result.reasons[0].dim)

    def test_no_more_than_four_reasons(self):
        many = tuple(
            RecoveryEvidence(
                code=f"code_{index}",
                source=EvidenceSource.TYPING,
                severity=2,
                title=f"Признак {index}",
                detail="",
                decision_ready=True,
            )
            for index in range(6)
        )
        self.assertLessEqual(len(view(ass=assessment(evidence=many)).reasons), 4)


class ActionTests(unittest.TestCase):
    def test_buttons_appear_only_with_an_active_recommendation(self):
        self.assertFalse(view().show_action_buttons)

        recommendation = RecoveryRecommendation(
            recommendation_id=1,
            kind=RecommendationKind.MICROBREAK,
            title="Рекомендуется короткий перерыв",
            message="Прервите работу примерно на 3 минуты.",
            suggested_break_sec=180.0,
            reason_codes=(),
            created_at=time.monotonic(),
        )
        result = view(ass=assessment(recommendation=recommendation))
        self.assertTrue(result.show_action_buttons)
        self.assertEqual(result.action_title, "Рекомендуется короткий перерыв")

    def test_break_state_switches_the_button_label(self):
        """Кнопка перерыва доступна всегда, но при активном перерыве меняет смысл."""

        self.assertFalse(view().break_active)
        self.assertTrue(view(break_active=True).break_active)

    def test_snooze_and_dismiss_stay_tied_to_a_recommendation(self):
        self.assertFalse(view().show_action_buttons)

    def test_high_level_without_recommendation_never_says_business_as_usual(self):
        """Cooldown не должен превращать совет в противоположный."""

        result = view(ass=assessment(level=WorkloadLevel.EXPRESSED))
        self.assertNotIn("обычном режиме", result.action_title)
        self.assertFalse(result.show_action_buttons)


class BreakViewTests(unittest.TestCase):
    """Во время перерыва экран показывает перерыв, а не рабочие показатели."""

    def _break(self, *, break_sec: float = 60.0, meaningful: float = 180.0):
        ass = assessment(level=WorkloadLevel.EXPRESSED)
        ass = RecoveryAssessment(
            **{
                **{f.name: getattr(ass, f.name) for f in ass.__dataclass_fields__.values()},
                "current_break_sec": break_sec,
            }
        )
        return build_main_screen_view(
            FakeSnapshot(),
            signals(),
            ass,
            now=time.monotonic(),
            break_active=True,
            meaningful_break_sec=meaningful,
            eye_rest_after_sec=EYE_REST,
            microbreak_after_sec=MICROBREAK,
            recovery_break_after_sec=RECOVERY,
        )

    def test_state_says_a_break_started(self):
        self.assertEqual(self._break().level_title, "Вы начали перерыв")

    def test_camera_reports_paused_measurement(self):
        result = self._break()
        self.assertEqual(result.signal_tone, "off")
        self.assertIn("не оцениваются", result.signal_hint)

    def test_timer_shows_break_duration_not_work(self):
        result = self._break(break_sec=41.0)
        self.assertEqual(result.work_label, "Перерыв")
        self.assertEqual(result.work_time_text, "00:00:41")

    def test_short_break_explains_when_it_counts(self):
        self.assertIn("засчитается", self._break(break_sec=30.0).next_threshold_text)

    def test_long_break_is_counted(self):
        self.assertIn("засчитан", self._break(break_sec=200.0).next_threshold_text)

    def test_snooze_and_dismiss_are_hidden_during_a_break(self):
        self.assertFalse(self._break().show_action_buttons)


class AfterBreakTests(unittest.TestCase):
    """После перерыва счётчик сессии продолжается сразу, без выдержки.

    Раньше здесь было состояние RETURNING: экран показывал «Отсчёт
    приостановлен» и обратный отсчёт до возобновления. Эти тесты падают
    на старом коде.
    """

    def _after_break(self, work_sec: float = 740.0):
        return build_main_screen_view(
            FakeSnapshot(),
            signals(state=UserState.ACTIVE_WORK),
            assessment(state=UserState.ACTIVE_WORK, work_sec=work_sec),
            now=time.monotonic(),
            eye_rest_after_sec=EYE_REST,
            microbreak_after_sec=MICROBREAK,
            recovery_break_after_sec=RECOVERY,
        )

    def test_counter_runs_immediately(self):
        result = self._after_break()
        self.assertEqual(result.work_label, "Непрерывная работа")
        self.assertEqual(result.work_time_text, "00:12:20")

    def test_no_countdown_before_the_counter_resumes(self):
        text = self._after_break().next_threshold_text
        self.assertNotIn("возобновится", text)

    def test_screen_does_not_explain_frozen_counters(self):
        titles = [block.title for block in self._after_break().details]
        self.assertNotIn("Почему счётчики стоят", titles)


class ThresholdWordingTests(unittest.TestCase):
    def test_stage_names_are_not_shown_to_the_user(self):
        """Внутренние названия порогов пользователю ничего не говорят."""

        for seconds in (10 * 60, 50 * 60, 80 * 60):
            text = view(ass=assessment(work_sec=seconds)).next_threshold_text
            with self.subTest(seconds=seconds):
                self.assertNotIn("разгрузк", text)
                self.assertNotIn("микроперерыв", text)


class DetailsTests(unittest.TestCase):
    def test_missing_self_report_is_explained(self):
        texts = " ".join(block.title for block in view().details)
        self.assertIn("Самооценка не участвует", texts)

    def test_present_self_report_is_quoted(self):
        result = view(ass=assessment(fatigue_sp=6, sleepiness_kss=7))
        block = next(b for b in result.details if b.channel == "self_report")
        self.assertIn("6 из 7", block.text)
        self.assertIn("7 из 9", block.text)

    def test_unfinished_baseline_is_explained_with_progress(self):
        result = view(
            sig=signals(typing_baseline_ready=False, typing_calibration_progress=0.4)
        )
        block = next(b for b in result.details if "норма печати" in b.title)
        self.assertIn("40%", block.text)

    def test_diagnostic_eye_channel_is_disclosed(self):
        titles = [block.title for block in view().details]
        self.assertIn("Глаза только измеряются", titles)

    def test_details_never_overflow_the_screen(self):
        self.assertLessEqual(len(view().details), 5)


class LevelTests(unittest.TestCase):
    def test_level_number_matches_the_engine(self):
        for level in WorkloadLevel:
            with self.subTest(level=level):
                self.assertEqual(view(ass=assessment(level=level)).level, int(level))

    def test_every_level_has_a_distinct_title(self):
        titles = {view(ass=assessment(level=level)).level_title for level in WorkloadLevel}
        self.assertEqual(len(titles), len(WorkloadLevel))


if __name__ == "__main__":
    unittest.main()
