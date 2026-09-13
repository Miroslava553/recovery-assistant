import unittest

from app_core.state_machine import (
    PresenceSignals,
    UserState,
    UserStateManager,
)


class UserStateManagerTests(unittest.TestCase):
    def make_manager(self) -> UserStateManager:
        manager = UserStateManager(
            active_input_threshold_sec=2.0,
            away_input_threshold_sec=5.0,
            away_face_threshold_sec=5.0,
            baseline_warmup_sec=4.0,
            transition_debounce_sec=1.0,
        )
        manager.reset(now=0.0)
        return manager

    def settle(
        self,
        manager: UserStateManager,
        signals: PresenceSignals,
        *,
        start: float,
    ):
        manager.update(signals, now=start)
        return manager.update(signals, now=start + 1.1)

    def test_recent_input_becomes_active_work(self):
        manager = self.make_manager()
        decision = self.settle(
            manager,
            PresenceSignals(True, 0.2, True, 0.95),
            start=0.0,
        )
        self.assertEqual(decision.state, UserState.ACTIVE_WORK)
        self.assertTrue(decision.fatigue_analysis_allowed)

    def test_visible_user_reading_is_passive_work(self):
        manager = self.make_manager()
        decision = self.settle(
            manager,
            PresenceSignals(True, 8.0, True, 0.95),
            start=0.0,
        )
        self.assertEqual(decision.state, UserState.PASSIVE_WORK)
        self.assertTrue(decision.fatigue_analysis_allowed)

    def test_mouse_inactivity_alone_is_not_away(self):
        manager = self.make_manager()
        decision = self.settle(
            manager,
            PresenceSignals(True, 30.0, True, 0.95),
            start=0.0,
        )
        self.assertEqual(decision.state, UserState.PASSIVE_WORK)

    def test_face_missing_and_long_idle_becomes_away(self):
        manager = self.make_manager()
        signals = PresenceSignals(False, 20.0, None, 0.80)

        manager.update(signals, now=0.0)
        manager.update(signals, now=5.1)
        decision = manager.update(signals, now=6.2)

        self.assertEqual(decision.state, UserState.AWAY)
        self.assertFalse(decision.fatigue_analysis_allowed)

    def _returned_from_away(self, manager, *, at: float = 7.0):
        """Довести автомат до AWAY и вернуть человека к работе."""

        away_signals = PresenceSignals(False, 20.0, None, 0.80)
        manager.update(away_signals, now=0.0)
        manager.update(away_signals, now=5.1)
        manager.update(away_signals, now=6.2)
        return manager.update(PresenceSignals(True, 0.1, True, 0.95), now=at)

    def test_return_from_away_resumes_work_immediately(self):
        """Раньше здесь был защитный период, и счётчик сессии стоял минуту."""

        manager = self.make_manager()
        decision = self._returned_from_away(manager)

        self.assertEqual(decision.state, UserState.ACTIVE_WORK)
        self.assertTrue(decision.fatigue_analysis_allowed)

    def test_personal_norm_is_not_updated_right_after_returning(self):
        """Первые секунды после перерыва не типичны: человек разгоняется."""

        manager = self.make_manager()
        decision = self._returned_from_away(manager)

        self.assertTrue(decision.baseline_warmup)
        self.assertFalse(decision.baseline_update_allowed)

    def test_personal_norm_resumes_after_the_warmup(self):
        manager = self.make_manager()
        self._returned_from_away(manager)

        working = PresenceSignals(True, 0.1, True, 0.95)
        decision = manager.update(working, now=7.0 + 4.0)

        self.assertFalse(decision.baseline_warmup)
        self.assertTrue(decision.baseline_update_allowed)

    def test_zero_warmup_disables_the_guard(self):
        manager = UserStateManager(
            active_input_threshold_sec=2.0,
            away_input_threshold_sec=5.0,
            away_face_threshold_sec=5.0,
            baseline_warmup_sec=0.0,
            transition_debounce_sec=1.0,
        )
        manager.reset(now=0.0)
        decision = self._returned_from_away(manager)

        self.assertFalse(decision.baseline_warmup)
        self.assertTrue(decision.baseline_update_allowed)

    def test_manual_break_is_immediate(self):
        manager = self.make_manager()
        decision = manager.update(
            PresenceSignals(
                True,
                0.1,
                True,
                0.95,
                manual_break=True,
            ),
            now=0.0,
        )

        self.assertEqual(decision.state, UserState.BREAK)
        self.assertFalse(decision.fatigue_analysis_allowed)

    def test_unknown_when_no_sensors_are_available(self):
        manager = self.make_manager()
        decision = manager.update(
            PresenceSignals(None, None, None, 0.0),
            now=0.0,
        )
        self.assertEqual(decision.state, UserState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
