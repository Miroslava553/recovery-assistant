import unittest
from datetime import date

import numpy as np

from app_core.face_presence import FacePresenceSnapshot
from app_core.input_activity import InputActivitySnapshot
from app_core.ocular_activity import EyeSignalState, OcularMetrics
from app_core.session_monitor import SessionMonitor
from app_core.state_machine import StateDecision, UserState
from app_core.typing_activity import TypingMetrics
from app_core.typing_baseline import TypingBaselineResult
from app_core.visual_attention import VisualAttentionSnapshot


class FakeFaceSensor:
    def __init__(self) -> None:
        self.frame = object()
        self.started = False
        self.snapshot_calls = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.started = False

    def snapshot(self) -> FacePresenceSnapshot:
        self.snapshot_calls += 1
        return FacePresenceSnapshot(
            face_detected=True,
            data_quality=0.90,
            detection_confidence=0.95,
            brightness=120.0,
            sharpness=100.0,
            reason="face_detected",
            frame=self.frame,
            face_box=(10, 10, 100, 100),
        )


class FakeInputSensor:
    def snapshot(self) -> InputActivitySnapshot:
        return InputActivitySnapshot(
            idle_seconds=0.2,
            recent_input=True,
            captured_at=1.0,
        )


class FakeTypingSensor:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def snapshot(self) -> TypingMetrics:
        return TypingMetrics(
            observation_sec=60.0,
            data_ready=True,
            text_key_count=57,
            correction_key_count=3,
            relevant_key_count=60,
            keys_per_minute=60.0,
            mean_interval_sec=0.50,
            median_interval_sec=0.40,
            interval_std_sec=0.10,
            rhythm_cv=0.20,
            longest_pause_sec=3.0,
            long_pause_count=2,
            burst_count=3,
            correction_ratio=0.05,
            current_idle_sec=0.2,
        )


class FakeAttentionAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, frame, *, data_quality: float) -> VisualAttentionSnapshot:
        self.calls += 1
        return VisualAttentionSnapshot(
            gaze_on_screen=True,
            face_landmarks_detected=True,
            horizontal_gaze_ratio=0.5,
            head_turn_ratio=0.0,
            head_vertical_ratio=0.5,
            reason="looking_toward_screen",
            left_eye_opening_ratio=0.30,
            right_eye_opening_ratio=0.33,
            left_eye_quality=0.90,
            right_eye_quality=0.88,
            head_pose_quality=0.92,
        )

    def close(self) -> None:
        pass


class FakeOcularTracker:
    def __init__(self) -> None:
        self.update_calls = []
        self.reset_calls = []

    @staticmethod
    def _metrics(reason: str = "eyes_open") -> OcularMetrics:
        return OcularMetrics(
            observation_sec=12.0,
            elapsed_window_sec=12.0,
            data_ready=True,
            calibrated=True,
            left_eye_opening_ratio=0.30,
            right_eye_opening_ratio=0.33,
            normalized_left_eye_opening=1.0,
            normalized_right_eye_opening=1.0,
            left_open_eye_reference=0.30,
            right_open_eye_reference=0.33,
            left_eye_state=EyeSignalState.OPEN,
            right_eye_state=EyeSignalState.OPEN,
            eyes_closed=False,
            current_closure_duration_sec=0.0,
            perclos=0.05,
            blink_count=2,
            incomplete_blink_count=1,
            left_wink_count=0,
            right_wink_count=0,
            uncertain_event_count=0,
            blink_rate_per_min=10.0,
            mean_blink_duration_ms=150.0,
            longest_blink_duration_ms=170.0,
            long_closure_count=0,
            long_closure_rate_per_min=0.0,
            severe_closure_detected=False,
            processed_fps=25.0,
            valid_sample_rate_hz=24.0,
            signal_coverage=0.95,
            processed_sample_count=300,
            valid_sample_count=285,
            quality=0.88,
            reason=reason,
        )

    def update(
        self,
        left_eye_opening_ratio,
        right_eye_opening_ratio,
        *,
        face_detected,
        data_quality,
        left_eye_quality,
        right_eye_quality,
        head_pose_quality,
        captured_at,
    ) -> OcularMetrics:
        self.update_calls.append(
            {
                "left_eye_opening_ratio": left_eye_opening_ratio,
                "right_eye_opening_ratio": right_eye_opening_ratio,
                "face_detected": face_detected,
                "data_quality": data_quality,
                "left_eye_quality": left_eye_quality,
                "right_eye_quality": right_eye_quality,
                "head_pose_quality": head_pose_quality,
                "captured_at": captured_at,
            }
        )
        return self._metrics()

    def reset(self, *, keep_calibration: bool = True) -> None:
        self.reset_calls.append(keep_calibration)


class FakeStateManager:
    def reset(self, *, now=None) -> None:
        pass

    def update(self, signals, *, now=None) -> StateDecision:
        return StateDecision(
            state=UserState.ACTIVE_WORK,
            previous_state=UserState.UNKNOWN,
            changed=True,
            reason="Тестовая активная работа.",
            fatigue_analysis_allowed=True,
            baseline_update_allowed=True,
            seconds_in_state=0.0,
        )


class FakeTypingBaselineService:
    def __init__(self) -> None:
        self.calls = []

    def observe(self, metrics, **kwargs) -> TypingBaselineResult:
        self.calls.append((metrics, kwargs))
        return TypingBaselineResult(
            attempted=True,
            accepted=True,
            persisted=True,
            typing_quality=1.0,
            calibration_day_count=1,
            calibration_progress=0.2,
            initial_calibration_complete=False,
            accepted_features=("typing_keys_per_minute",),
            reason="Окно принято.",
        )


class SessionMonitorTests(unittest.TestCase):
    def make_monitor(
        self,
        *,
        baseline_service=None,
        include_diagnostic_frame: bool = False,
    ):
        face_sensor = FakeFaceSensor()
        attention = FakeAttentionAnalyzer()
        ocular = FakeOcularTracker()
        monitor = SessionMonitor(
            face_sensor=face_sensor,
            input_sensor=FakeInputSensor(),
            typing_sensor=FakeTypingSensor(),
            attention_analyzer=attention,
            ocular_tracker=ocular,
            state_manager=FakeStateManager(),
            typing_baseline_service=baseline_service,
            background_visual_processing=False,
            include_diagnostic_frame=include_diagnostic_frame,
        )
        return monitor, face_sensor, attention, ocular

    def test_typing_baseline_is_observed_inside_session(self):
        baseline_service = FakeTypingBaselineService()
        monitor, _, _, _ = self.make_monitor(baseline_service=baseline_service)
        try:
            monitor.start()
            snapshot = monitor.snapshot()
        finally:
            monitor.close()
        self.assertIsNotNone(snapshot.typing_baseline)
        self.assertTrue(snapshot.typing_baseline.accepted)
        self.assertEqual(len(baseline_service.calls), 1)
        _, kwargs = baseline_service.calls[0]
        self.assertEqual(kwargs["state"], UserState.ACTIVE_WORK)
        self.assertIsInstance(kwargs["workday"], date)

    def test_background_mode_does_not_return_raw_frame(self):
        monitor, _, _, _ = self.make_monitor(include_diagnostic_frame=False)
        try:
            monitor.start()
            snapshot = monitor.snapshot()
        finally:
            monitor.close()
        self.assertIsNone(snapshot.frame)
        self.assertIsNone(snapshot.face_box)

    def test_diagnostic_mode_can_return_temporary_frame(self):
        monitor, face_sensor, _, _ = self.make_monitor(include_diagnostic_frame=True)
        try:
            monitor.start()
            snapshot = monitor.snapshot()
        finally:
            monitor.close()
        self.assertIs(snapshot.frame, face_sensor.frame)
        self.assertEqual(snapshot.face_box, (10, 10, 100, 100))

    def test_both_eye_measurements_reach_ocular_tracker(self):
        monitor, _, _, ocular = self.make_monitor()
        try:
            monitor.start()
            snapshot = monitor.snapshot()
        finally:
            monitor.close()
        self.assertEqual(len(ocular.update_calls), 1)
        call = ocular.update_calls[0]
        self.assertAlmostEqual(call["left_eye_opening_ratio"], 0.30)
        self.assertAlmostEqual(call["right_eye_opening_ratio"], 0.33)
        self.assertAlmostEqual(call["left_eye_quality"], 0.90)
        self.assertAlmostEqual(call["right_eye_quality"], 0.88)
        self.assertTrue(snapshot.ocular_metrics.data_ready)
        self.assertEqual(snapshot.ocular_status, "Глаза открыты")

    def test_no_background_worker_processes_one_frame_per_snapshot(self):
        monitor, face_sensor, attention, _ = self.make_monitor()
        try:
            monitor.start()
            monitor.snapshot()
            monitor.snapshot()
        finally:
            monitor.close()
        self.assertEqual(face_sensor.snapshot_calls, 2)
        self.assertEqual(attention.calls, 2)

    def test_latest_visual_frame_returns_independent_copy(self):
        monitor, face_sensor, _, _ = self.make_monitor()
        face_sensor.frame = np.zeros((2, 3, 3), dtype=np.uint8)
        try:
            monitor.start()
            monitor.snapshot()
            captured_at, frame = monitor.latest_visual_frame()
        finally:
            monitor.close()
        self.assertIsNotNone(captured_at)
        self.assertIsNotNone(frame)
        self.assertIsNot(frame, face_sensor.frame)
        frame[0, 0, 0] = 255
        self.assertEqual(face_sensor.frame[0, 0, 0], 0)

    def test_start_resets_window_but_keeps_eye_calibration(self):
        monitor, _, _, ocular = self.make_monitor()
        try:
            monitor.start()
        finally:
            monitor.close()
        self.assertEqual(ocular.reset_calls, [True])


if __name__ == "__main__":
    unittest.main()
