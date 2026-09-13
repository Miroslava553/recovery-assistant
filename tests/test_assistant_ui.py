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
from assistant_app import (
    FATIGUE_SP_LABELS,
    SLEEPINESS_KSS_LABELS,
    neighbor_description,
    self_report_description,
    user_facing_summary,
)


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


if __name__ == "__main__":
    unittest.main()
