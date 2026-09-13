import unittest

import numpy as np

import ocular_validation as ui


class OcularValidationUiTests(unittest.TestCase):
    def test_space_starts_only_before_recording(self):
        self.assertEqual(
            ui._capture_action_from_key(32, recording=False),
            "start",
        )
        self.assertEqual(
            ui._capture_action_from_key(32, recording=True),
            "none",
        )

    def test_enter_saves_only_during_recording(self):
        for key in (10, 13):
            self.assertEqual(
                ui._capture_action_from_key(key, recording=True),
                "save",
            )
            self.assertEqual(
                ui._capture_action_from_key(key, recording=False),
                "none",
            )

    def test_escape_closes_in_both_states(self):
        self.assertEqual(
            ui._capture_action_from_key(27, recording=False),
            "close",
        )
        self.assertEqual(
            ui._capture_action_from_key(27, recording=True),
            "close",
        )

    def test_letter_keys_are_not_required(self):
        for key in (ord("s"), ord("S"), ord("q"), ord("Q"), -1):
            self.assertEqual(
                ui._capture_action_from_key(key, recording=False),
                "none",
            )

    def test_capture_display_keeps_camera_pixels_unscaled(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = 123
        buttons = {}

        display = ui._build_capture_display(
            frame,
            recording=False,
            elapsed=0.0,
            calibration_sec=5.0,
            measured_fps=29.0,
            valid=True,
            valid_count=0,
            sample_count=0,
            attention=ui._empty_attention(),
            data_quality=0.9,
            button_rects=buttons,
        )

        self.assertEqual(display.shape, (480, 1200, 3))
        self.assertTrue(np.array_equal(display[:, :640], frame))
        self.assertIn("start", buttons)
        self.assertNotIn("save", buttons)

    def test_quality_status_requires_target_coverage(self):
        self.assertEqual(
            ui._quality_status({"total_frame_count": 100, "frame_coverage": 0.84}),
            "fail",
        )
        self.assertEqual(
            ui._quality_status({"total_frame_count": 100, "frame_coverage": 0.85}),
            "pass",
        )
        self.assertEqual(
            ui._quality_status({"total_frame_count": 0, "frame_coverage": 0.0}),
            "insufficient_data",
        )

    def test_quality_report_names_dominant_failure(self):
        reason = "head_pose_quality_low_or_invalid"
        summary = {
            "frame_coverage": 0.25,
            "valid_frame_count": 25,
            "total_frame_count": 100,
            "reason_counts": {reason: 75},
            "reason_percent_of_frames": {reason: 0.75},
            "thresholds": {
                "min_data_quality": 0.55,
                "min_eye_quality": 0.55,
                "min_head_pose_quality": 0.35,
            },
        }
        text = ui._format_quality_report_text(
            {
                "status": "fail",
                "target_frame_coverage": 0.85,
                "all_recording": summary,
                "blink_test_phase": summary,
            }
        )

        self.assertIn("ракурс головы ниже порога", text)
        self.assertIn("75.0%", text)

    def test_quality_command_analyzes_existing_session_without_video(self):
        import json
        import tempfile
        from pathlib import Path

        from app_core.ocular_validation import ValidationSample, save_samples

        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            session.mkdir()
            samples = []
            for index in range(10):
                samples.append(
                    ValidationSample(
                        frame_index=index,
                        timestamp_sec=5.0 + index * 0.1,
                        left_eye_opening_raw=0.30,
                        right_eye_opening_raw=0.33,
                        data_quality=0.90,
                        left_eye_quality=0.90,
                        right_eye_quality=0.90,
                        head_pose_quality=0.20 if index < 8 else 0.90,
                        face_detected=True,
                        valid=index >= 8,
                    )
                )
            save_samples(session / "signals.csv", samples)
            (session / "metadata.json").write_text(
                json.dumps({"schema_version": 2, "calibration_sec": 5.0}),
                encoding="utf-8",
            )

            result = ui.generate_quality_diagnostics(session)

            self.assertEqual(result, 0)
            self.assertTrue((session / "quality_report.txt").exists())
            payload = json.loads((session / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "insufficient_data")
            self.assertAlmostEqual(
                payload["blink_test_phase"]["frame_coverage"],
                0.2,
            )

    def test_quality_command_recalibrates_old_absolute_head_pose_formula(self):
        import json
        import tempfile
        from pathlib import Path

        from app_core.ocular_validation import ValidationSample, save_samples

        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            session.mkdir()
            samples = []
            for index in range(180):
                timestamp = index / 30.0
                samples.append(
                    ValidationSample(
                        frame_index=index,
                        timestamp_sec=timestamp,
                        left_eye_opening_raw=0.31,
                        right_eye_opening_raw=0.34,
                        data_quality=0.90,
                        left_eye_quality=0.90,
                        right_eye_quality=0.90,
                        # Имитируем старую сессию: абсолютная формула
                        # ошибочно забраковала нейтральный ракурс.
                        head_pose_quality=0.20,
                        face_detected=True,
                        valid=False,
                        head_turn_ratio=0.03,
                        head_vertical_ratio=0.78,
                    )
                )
            save_samples(session / "signals.csv", samples)
            (session / "metadata.json").write_text(
                json.dumps({"schema_version": 3, "calibration_sec": 5.0}),
                encoding="utf-8",
            )

            result = ui.generate_quality_diagnostics(session)

            self.assertEqual(result, 0)
            payload = json.loads(
                (session / "quality_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["head_pose_model"], "personal_calibrated")
            self.assertEqual(payload["status"], "pass")
            self.assertAlmostEqual(
                payload["legacy_blink_test_phase"]["frame_coverage"],
                0.0,
            )
            self.assertAlmostEqual(
                payload["blink_test_phase"]["frame_coverage"],
                1.0,
            )

    def test_recording_display_shows_save_button(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        buttons = {}

        ui._build_capture_display(
            frame,
            recording=True,
            elapsed=6.0,
            calibration_sec=5.0,
            measured_fps=29.0,
            valid=True,
            valid_count=100,
            sample_count=100,
            attention=ui._empty_attention(),
            data_quality=0.9,
            button_rects=buttons,
        )

        self.assertIn("save", buttons)
        self.assertNotIn("start", buttons)
        self.assertTrue(ui._point_in_rect(700, 440, buttons["save"]))


class CaptureSessionFlowTests(unittest.TestCase):
    class FakeSensor:
        def __init__(self, **_kwargs):
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def start(self):
            return None

        def snapshot(self):
            from types import SimpleNamespace

            return SimpleNamespace(
                frame=self.frame.copy(),
                data_quality=0.9,
                brightness=120.0,
                sharpness=100.0,
            )

        def close(self):
            return None

    class FakeAnalyzer:
        def __init__(self, **_kwargs):
            pass

        def analyze(self, _frame, *, data_quality):
            from app_core.visual_attention import VisualAttentionSnapshot

            return VisualAttentionSnapshot(
                gaze_on_screen=True,
                face_landmarks_detected=True,
                horizontal_gaze_ratio=0.5,
                head_turn_ratio=0.0,
                head_vertical_ratio=0.5,
                reason="ok",
                left_eye_opening_ratio=0.30,
                right_eye_opening_ratio=0.32,
                left_eye_quality=0.9,
                right_eye_quality=0.9,
                head_pose_quality=0.9,
            )

        def close(self):
            return None

    class FakeWriter:
        def __init__(self):
            self.write_count = 0
            self.released = False

        def write(self, _frame):
            self.write_count += 1

        def release(self):
            self.released = True

    def test_capture_can_start_with_space_and_save_with_enter(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        writer = self.FakeWriter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(ui, "FacePresenceSensor", self.FakeSensor),
                patch.object(ui, "VisualAttentionAnalyzer", self.FakeAnalyzer),
                patch.object(ui, "_open_video_writer", return_value=(writer, "TEST")),
                patch.object(ui, "_window_is_visible", return_value=True),
                patch.object(ui.cv2, "namedWindow"),
                patch.object(ui.cv2, "setMouseCallback"),
                patch.object(ui.cv2, "imshow"),
                patch.object(ui.cv2, "waitKeyEx", side_effect=[32, 13]),
                patch.object(ui.cv2, "destroyWindow"),
                patch.object(ui.time, "sleep"),
            ):
                result = ui.capture_session(
                    camera_index=0,
                    calibration_sec=3.0,
                    target_fps=1000.0,
                    sessions_root=root,
                )

            self.assertEqual(result, 0)
            self.assertEqual(writer.write_count, 1)
            session_dirs = [item for item in root.iterdir() if item.is_dir()]
            self.assertEqual(len(session_dirs), 1)
            self.assertTrue((session_dirs[0] / "signals.csv").exists())
            self.assertTrue((session_dirs[0] / "metadata.json").exists())

    def test_closing_before_recording_is_clean_exit(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(ui, "FacePresenceSensor", self.FakeSensor),
                patch.object(ui, "VisualAttentionAnalyzer", self.FakeAnalyzer),
                patch.object(ui, "_window_is_visible", return_value=True),
                patch.object(ui.cv2, "namedWindow"),
                patch.object(ui.cv2, "setMouseCallback"),
                patch.object(ui.cv2, "imshow"),
                patch.object(ui.cv2, "waitKeyEx", return_value=27),
                patch.object(ui.cv2, "destroyWindow"),
                patch.object(ui.time, "sleep"),
            ):
                result = ui.capture_session(
                    camera_index=0,
                    calibration_sec=3.0,
                    target_fps=1000.0,
                    sessions_root=root,
                )

            self.assertEqual(result, 0)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
