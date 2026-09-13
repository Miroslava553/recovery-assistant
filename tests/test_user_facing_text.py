from __future__ import annotations

import time
import unittest

from app_core.recovery_engine import (
    RecoveryAssessment,
    WorkloadLevel,
    WORKLOAD_SUMMARIES,
    WORKLOAD_TITLES,
)
from app_core.state_machine import UserState
from ui.formatting import (
    format_assessment_age,
    neighbor_description,
    recommended_action,
    self_report_description,
    user_facing_summary,
)
from ui.labels import FATIGUE_SP_LABELS, SLEEPINESS_KSS_LABELS


class UserFacingSummaryTests(unittest.TestCase):
    def _assessment(
        self,
        *,
        state: UserState = UserState.ACTIVE_WORK,
        level: WorkloadLevel = WorkloadLevel.STABLE,
    ) -> RecoveryAssessment:
        return RecoveryAssessment(
            captured_at=time.monotonic(),
            state=state,
            status_label="test",
            continuous_work_sec=54.0,
            current_break_sec=0.0,
            typing_calibration_progress=0.2,
            evidence=(),
            recommendation=None,
            recommendation_is_new=False,
            decision_basis="test",
            ocular_used_for_decision=False,
            workload_level=level,
            workload_title=WORKLOAD_TITLES[level],
            workload_summary=WORKLOAD_SUMMARIES[level],
        )

    def test_active_state_uses_formal_five_level_label(self):
        title, detail = user_facing_summary(
            self._assessment(level=WorkloadLevel.SUSTAINED),
            demo_mode=False,
        )
        self.assertEqual(title, "Нагрузка становится устойчивой")
        self.assertIn("сохраняются", detail)

    def test_break_is_not_a_sixth_workload_level(self):
        title, detail = user_facing_summary(
            self._assessment(state=UserState.BREAK),
            demo_mode=False,
        )
        self.assertEqual(title, "Восстановительный интервал идёт")
        self.assertIn("не накапливается", detail)

    def test_self_report_labels_have_distinct_neighbor_descriptions(self):
        six = self_report_description(SLEEPINESS_KSS_LABELS, 6)
        seven = self_report_description(SLEEPINESS_KSS_LABELS, 7)
        self.assertIn("желания лечь спать", six[1])
        self.assertIn("определённо хочу спать", seven[1])
        comparison = neighbor_description(SLEEPINESS_KSS_LABELS, 6)
        self.assertIn("Выше: 7", comparison)

    def test_all_scale_points_have_user_explanations(self):
        self.assertEqual(set(FATIGUE_SP_LABELS), set(range(1, 8)))
        self.assertEqual(set(SLEEPINESS_KSS_LABELS), set(range(1, 10)))
        for labels in (FATIGUE_SP_LABELS, SLEEPINESS_KSS_LABELS):
            for short, detail in labels.values():
                self.assertGreaterEqual(len(short), 5)
                self.assertGreaterEqual(len(detail), 20)


class RecommendedActionTests(unittest.TestCase):
    """Совет на главном экране следует уровню, а не наличию уведомления."""

    def _assessment(
        self,
        *,
        state: UserState = UserState.ACTIVE_WORK,
        level: WorkloadLevel = WorkloadLevel.STABLE,
    ) -> RecoveryAssessment:
        return RecoveryAssessment(
            captured_at=time.monotonic(),
            state=state,
            status_label="",
            continuous_work_sec=0.0,
            current_break_sec=0.0,
            typing_calibration_progress=0.0,
            evidence=(),
            recommendation=None,
            recommendation_is_new=False,
            decision_basis="",
            ocular_used_for_decision=False,
            workload_level=level,
            workload_title=WORKLOAD_TITLES[level],
            workload_summary=WORKLOAD_SUMMARIES[level],
        )

    def test_expressed_level_never_advises_business_as_usual(self):
        """Главная исправленная ошибка: cooldown не меняет смысл совета."""

        for level in (WorkloadLevel.EXPRESSED, WorkloadLevel.RECOVERY_PRIORITY):
            with self.subTest(level=level):
                title, text = recommended_action(self._assessment(level=level))
                self.assertNotIn("обычном режиме", title)
                self.assertNotIn("обычном режиме", text)

    def test_every_level_has_its_own_action(self):
        titles = {
            recommended_action(self._assessment(level=level))[0]
            for level in WorkloadLevel
        }
        self.assertEqual(len(titles), len(WorkloadLevel))

    def test_stable_level_allows_normal_work(self):
        title, _ = recommended_action(self._assessment(level=WorkloadLevel.STABLE))
        self.assertIn("обычном режиме", title)

    def test_break_state_overrides_level(self):
        title, _ = recommended_action(
            self._assessment(state=UserState.BREAK, level=WorkloadLevel.EXPRESSED)
        )
        self.assertIn("перерыв", title.lower())


class AssessmentAgeTests(unittest.TestCase):
    def test_fresh_assessment_reads_as_now(self):
        self.assertEqual(format_assessment_age(0.0), "сейчас")
        self.assertEqual(format_assessment_age(4.9), "сейчас")

    def test_seconds_minutes_and_hours(self):
        self.assertEqual(format_assessment_age(30.0), "30 с назад")
        self.assertEqual(format_assessment_age(120.0), "2 мин назад")
        self.assertEqual(format_assessment_age(3720.0), "1 ч 02 мин назад")

    def test_negative_age_is_clamped(self):
        self.assertEqual(format_assessment_age(-5.0), "сейчас")


if __name__ == "__main__":
    unittest.main()
