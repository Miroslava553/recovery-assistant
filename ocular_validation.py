from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app_core.face_presence import FacePresenceSensor
from app_core.ocular_validation import (
    EyeSampleFailureReason,
    EyeSampleQualityAssessment,
    EyeSampleQualityThresholds,
    HeadPoseReference,
    ManualBlinkLabel,
    OfflineBlinkConfig,
    ValidationDataError,
    ValidationLabelKind,
    ValidationSample,
    apply_personal_head_pose_quality,
    assess_eye_sample_quality,
    build_report_payload,
    calibrated_head_pose_quality,
    calculate_validation_metrics,
    detect_offline_blinks,
    estimate_head_pose_reference,
    estimate_open_eye_reference,
    load_labels,
    load_samples,
    normalize_samples,
    optimize_offline_thresholds,
    render_validation_plot,
    save_events,
    save_labels,
    save_normalized_samples,
    save_report_files,
    save_samples,
    summarize_eye_sample_quality,
)
from app_core.visual_attention import VisualAttentionAnalyzer, VisualAttentionSnapshot


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "validation_sessions"
WINDOW_CAPTURE = "Ocular validation capture"
WINDOW_REVIEW = "Ocular validation review"
QUALITY_TARGET_COVERAGE = 0.85

QUALITY_REASON_LABELS_RU = {
    EyeSampleFailureReason.FACE_NOT_DETECTED.value: "лицо не найдено",
    EyeSampleFailureReason.IMAGE_QUALITY_LOW_OR_INVALID.value: "качество изображения ниже порога",
    EyeSampleFailureReason.LEFT_EYE_QUALITY_LOW_OR_INVALID.value: "качество левого глаза ниже порога",
    EyeSampleFailureReason.RIGHT_EYE_QUALITY_LOW_OR_INVALID.value: "качество правого глаза ниже порога",
    EyeSampleFailureReason.HEAD_POSE_QUALITY_LOW_OR_INVALID.value: "ракурс головы ниже порога",
    EyeSampleFailureReason.LEFT_EYE_SIGNAL_MISSING_OR_INVALID.value: "сигнал левого глаза отсутствует/некорректен",
    EyeSampleFailureReason.RIGHT_EYE_SIGNAL_MISSING_OR_INVALID.value: "сигнал правого глаза отсутствует/некорректен",
}

QUALITY_REASON_LABELS_EN = {
    EyeSampleFailureReason.FACE_NOT_DETECTED.value: "face missing",
    EyeSampleFailureReason.IMAGE_QUALITY_LOW_OR_INVALID.value: "image quality low",
    EyeSampleFailureReason.LEFT_EYE_QUALITY_LOW_OR_INVALID.value: "left-eye quality low",
    EyeSampleFailureReason.RIGHT_EYE_QUALITY_LOW_OR_INVALID.value: "right-eye quality low",
    EyeSampleFailureReason.HEAD_POSE_QUALITY_LOW_OR_INVALID.value: "head-pose quality low",
    EyeSampleFailureReason.LEFT_EYE_SIGNAL_MISSING_OR_INVALID.value: "left-eye signal invalid",
    EyeSampleFailureReason.RIGHT_EYE_SIGNAL_MISSING_OR_INVALID.value: "right-eye signal invalid",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Локальная измерительная система для проверки глазного сенсора. "
            "Видео создаётся только после явного запуска capture и остаётся "
            "внутри выбранной папки."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture",
        help="Записать локальную диагностическую сессию.",
    )
    capture.add_argument("--camera", type=int, default=0)
    capture.add_argument("--calibration-sec", type=float, default=5.0)
    capture.add_argument("--target-fps", type=float, default=30.0)
    capture.add_argument("--sessions-root", type=Path, default=DEFAULT_SESSIONS_ROOT)

    review = subparsers.add_parser(
        "review",
        help="Вручную разметить моргания на локальном видео.",
    )
    review.add_argument("session", type=Path)

    report = subparsers.add_parser(
        "report",
        help="Посчитать precision/recall и подобрать пороги для этой сессии.",
    )
    report.add_argument("session", type=Path)
    report.add_argument("--matching-tolerance-sec", type=float, default=0.25)

    quality = subparsers.add_parser(
        "quality",
        help="Разобрать причины непригодности уже записанной сессии.",
    )
    quality.add_argument("session", type=Path)

    delete_video = subparsers.add_parser(
        "delete-video",
        help="Удалить диагностическое видео, сохранив сигналы и отчёт.",
    )
    delete_video.add_argument("session", type=Path)
    delete_video.add_argument("--force", action="store_true")

    subparsers.add_parser(
        "list",
        help="Показать локальные валидационные сессии.",
    ).add_argument("--sessions-root", type=Path, default=DEFAULT_SESSIONS_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            return capture_session(
                camera_index=args.camera,
                calibration_sec=args.calibration_sec,
                target_fps=args.target_fps,
                sessions_root=args.sessions_root,
            )
        if args.command == "review":
            return review_session(args.session)
        if args.command == "report":
            return generate_report(
                args.session,
                matching_tolerance_sec=args.matching_tolerance_sec,
            )
        if args.command == "quality":
            return generate_quality_diagnostics(args.session)
        if args.command == "delete-video":
            return delete_video(args.session, force=args.force)
        if args.command == "list":
            return list_sessions(args.sessions_root)
    except (ValidationDataError, RuntimeError, OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


def capture_session(
    *,
    camera_index: int,
    calibration_sec: float,
    target_fps: float,
    sessions_root: Path,
) -> int:
    if calibration_sec < 3.0:
        raise ValueError("Калибровка должна длиться не менее 3 секунд.")
    if target_fps <= 0.0:
        raise ValueError("target_fps должен быть больше нуля.")

    print("Диагностический режим готов.", flush=True)
    print("1. Щёлкни по окну камеры мышью.", flush=True)
    print("2. Нажми ПРОБЕЛ один раз или кнопку START в окне.", flush=True)
    print("3. Первые секунды спокойно смотри на экран — это калибровка.", flush=True)
    print("4. После надписи BLINK TEST сделай не менее 20 обычных морганий.", flush=True)
    print("5. Нажми ENTER или кнопку SAVE, чтобы сохранить сессию.", flush=True)
    print("Важно: моргания здесь не считаются в реальном времени.", flush=True)
    print("Режим записывает видео и числовой сигнал для последующей разметки.", flush=True)

    sensor = FacePresenceSensor(
        camera_index=camera_index,
        min_detection_confidence=0.55,
        run_face_detection=False,
    )
    analyzer = VisualAttentionAnalyzer(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smoothing_window=1,
    )

    quality_thresholds = EyeSampleQualityThresholds()
    samples: list[ValidationSample] = []
    frame_times: deque[float] = deque(maxlen=120)
    failure_reason_counts: Counter[str] = Counter()
    primary_reason_counts: Counter[str] = Counter()
    session_dir: Path | None = None
    video_path: Path | None = None
    video_writer: cv2.VideoWriter | None = None
    video_codec = ""
    recording = False
    recording_started_at = 0.0
    frame_index = 0
    valid_sample_count = 0
    last_frame: np.ndarray | None = None
    capture_error: Exception | None = None
    stop_reason = "window_closed"
    pose_reference: HeadPoseReference | None = None

    ui_requests = {"start": False, "save": False}
    button_rects: dict[str, tuple[int, int, int, int]] = {}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        if _point_in_rect(x, y, button_rects.get("start")):
            ui_requests["start"] = True
        if _point_in_rect(x, y, button_rects.get("save")):
            ui_requests["save"] = True

    cv2.namedWindow(WINDOW_CAPTURE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_CAPTURE, on_mouse)
    try:
        sensor.start()
        period = 1.0 / target_fps
        while True:
            loop_started = time.monotonic()
            face = sensor.snapshot()
            frame = face.frame
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                attention = _empty_attention()
            else:
                last_frame = frame
                attention = analyzer.analyze(
                    frame,
                    data_quality=face.data_quality,
                )

            face_detected = attention.face_landmarks_detected
            now = time.monotonic()
            frame_times.append(now)
            measured_fps = _timestamp_rate(frame_times)
            elapsed = now - recording_started_at if recording else 0.0

            if recording and elapsed >= calibration_sec and pose_reference is None:
                try:
                    pose_reference = estimate_head_pose_reference(
                        samples,
                        calibration_duration_sec=calibration_sec,
                    )
                    print(
                        "Персональная норма ракурса головы рассчитана: "
                        f"turn={pose_reference.neutral_turn_ratio:.3f}, "
                        f"vertical={pose_reference.neutral_vertical_ratio:.3f}",
                        flush=True,
                    )
                except ValidationDataError:
                    # В первые кадры после окончания калибровки выборка может
                    # быть ещё недостаточной. Итоговая проверка повторится при
                    # сохранении сессии и выдаст точную причину.
                    pose_reference = None

            effective_head_pose_quality = attention.head_pose_quality
            if pose_reference is not None:
                effective_head_pose_quality = calibrated_head_pose_quality(
                    head_turn_ratio=attention.head_turn_ratio,
                    head_vertical_ratio=attention.head_vertical_ratio,
                    reference=pose_reference,
                )

            quality_assessment = assess_eye_sample_quality(
                face_detected=face_detected,
                left_eye_opening_raw=attention.left_eye_opening_ratio,
                right_eye_opening_raw=attention.right_eye_opening_ratio,
                data_quality=face.data_quality,
                left_eye_quality=attention.left_eye_quality,
                right_eye_quality=attention.right_eye_quality,
                head_pose_quality=effective_head_pose_quality,
                thresholds=quality_thresholds,
            )
            valid = quality_assessment.valid

            if recording:
                assert video_writer is not None
                video_writer.write(frame)
                samples.append(
                    ValidationSample(
                        frame_index=frame_index,
                        timestamp_sec=elapsed,
                        left_eye_opening_raw=attention.left_eye_opening_ratio,
                        right_eye_opening_raw=attention.right_eye_opening_ratio,
                        data_quality=face.data_quality,
                        left_eye_quality=attention.left_eye_quality,
                        right_eye_quality=attention.right_eye_quality,
                        head_pose_quality=effective_head_pose_quality,
                        face_detected=face_detected,
                        valid=valid,
                        brightness=face.brightness,
                        sharpness=face.sharpness,
                        horizontal_gaze_ratio=attention.horizontal_gaze_ratio,
                        head_turn_ratio=attention.head_turn_ratio,
                        head_vertical_ratio=attention.head_vertical_ratio,
                    )
                )
                frame_index += 1
                if valid:
                    valid_sample_count += 1
                else:
                    for reason in quality_assessment.failed_reasons:
                        failure_reason_counts[reason.value] += 1
                    if quality_assessment.primary_reason is not None:
                        primary_reason_counts[quality_assessment.primary_reason.value] += 1

            display = _build_capture_display(
                frame,
                recording=recording,
                elapsed=elapsed,
                calibration_sec=calibration_sec,
                measured_fps=measured_fps,
                valid=valid,
                valid_count=valid_sample_count,
                sample_count=len(samples),
                attention=attention,
                data_quality=face.data_quality,
                button_rects=button_rects,
                quality_assessment=quality_assessment,
                failure_reason_counts=failure_reason_counts,
                effective_head_pose_quality=effective_head_pose_quality,
                pose_reference_ready=pose_reference is not None,
            )
            cv2.imshow(WINDOW_CAPTURE, display)
            key = cv2.waitKeyEx(10)
            action = _capture_action_from_key(key, recording=recording)

            if ui_requests["start"]:
                ui_requests["start"] = False
                if not recording:
                    action = "start"
            if ui_requests["save"]:
                ui_requests["save"] = False
                if recording:
                    action = "save"

            if action == "start":
                if last_frame is None:
                    print("Камера ещё не предоставила пригодный кадр.", flush=True)
                else:
                    session_dir = _new_session_directory(sessions_root)
                    video_path = session_dir / "validation_video.avi"
                    video_writer, video_codec = _open_video_writer(
                        video_path,
                        target_fps=target_fps,
                        frame_size=(last_frame.shape[1], last_frame.shape[0]),
                    )
                    recording = True
                    recording_started_at = time.monotonic()
                    samples.clear()
                    frame_index = 0
                    valid_sample_count = 0
                    failure_reason_counts.clear()
                    primary_reason_counts.clear()
                    pose_reference = None
                    stop_reason = "saved_by_user"
                    print()
                    print("=== ЗАПИСЬ НАЧАЛАСЬ ===", flush=True)
                    print(f"Папка сессии: {session_dir}", flush=True)
                    print(
                        f"Первые {calibration_sec:.0f} с держи глаза обычно открытыми.",
                        flush=True,
                    )

            if action == "save":
                stop_reason = "saved_by_user"
                print("Остановка и сохранение записи...", flush=True)
                break

            if action == "close":
                stop_reason = "closed_by_user"
                if recording:
                    print("Окно закрыто: текущая запись будет сохранена.", flush=True)
                break

            if not _window_is_visible(WINDOW_CAPTURE):
                stop_reason = "window_closed"
                if recording:
                    print("Окно камеры закрыто: текущая запись будет сохранена.", flush=True)
                break

            remaining = period - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)

    except Exception as error:
        capture_error = error
    finally:
        if video_writer is not None:
            video_writer.release()
        sensor.close()
        analyzer.close()
        try:
            cv2.destroyWindow(WINDOW_CAPTURE)
        except cv2.error:
            pass

    if session_dir is not None and samples:
        legacy_samples = list(samples)
        pose_reference_error: str | None = None
        try:
            pose_reference = estimate_head_pose_reference(
                samples,
                calibration_duration_sec=calibration_sec,
            )
            samples = apply_personal_head_pose_quality(
                samples,
                reference=pose_reference,
                thresholds=quality_thresholds,
            )
        except ValidationDataError as error:
            pose_reference = None
            pose_reference_error = str(error)

        save_samples(session_dir / "signals.csv", samples)
        duration_sec = samples[-1].timestamp_sec
        quality_payload = _build_quality_payload(
            samples,
            calibration_sec=calibration_sec,
            thresholds=quality_thresholds,
            head_pose_reference=pose_reference,
            pose_reference_error=pose_reference_error,
            legacy_samples=legacy_samples,
        )
        quality_all = quality_payload["all_recording"]
        quality_blink_test = quality_payload["blink_test_phase"]
        assert isinstance(quality_all, dict)
        assert isinstance(quality_blink_test, dict)
        quality_status = str(quality_payload["status"])
        _atomic_json(session_dir / "quality_report.json", quality_payload)
        (session_dir / "quality_report.txt").write_text(
            _format_quality_report_text(quality_payload),
            encoding="utf-8",
        )

        metadata = {
            "schema_version": 3,
            "created_at": datetime.now().astimezone().isoformat(),
            "camera_index": camera_index,
            "calibration_sec": calibration_sec,
            "target_fps": target_fps,
            "measured_fps": _sample_rate_from_validation_samples(samples),
            "sample_count": len(samples),
            "valid_sample_count": int(quality_all["valid_frame_count"]),
            "signal_coverage": float(quality_all["frame_coverage"]),
            "blink_test_signal_coverage": float(quality_blink_test["frame_coverage"]),
            "quality_status": quality_status,
            "quality_thresholds": quality_all["thresholds"],
            "quality_report_file": "quality_report.json",
            "head_pose_model": "personal_calibrated" if pose_reference is not None else "broad_fallback",
            "head_pose_reference": (
                _head_pose_reference_payload(pose_reference)
                if pose_reference is not None
                else None
            ),
            "head_pose_reference_error": pose_reference_error,
            "duration_sec": duration_sec,
            "video_file": video_path.name if video_path is not None else None,
            "video_codec": video_codec,
            "stop_reason": stop_reason,
            "privacy": (
                "Видео создано только в явно запущенном диагностическом режиме. "
                "Обычный main.py видео не записывает."
            ),
        }
        _atomic_json(session_dir / "metadata.json", metadata)
        print(f"Сессия сохранена: {session_dir}", flush=True)
        print(f"Записано кадров: {len(samples)}", flush=True)
        _print_quality_summary(quality_payload)
        if quality_status == "pass":
            print("Следующая команда:", flush=True)
            print(f'python ocular_validation.py review "{session_dir}"', flush=True)
        else:
            print(
                "Эта сессия сохранена, но пока не годится для настройки детектора.",
                flush=True,
            )
            print(
                f"Подробности: {session_dir / 'quality_report.txt'}",
                flush=True,
            )
    elif session_dir is not None:
        print("Сессия не сохранена: не было записано ни одного кадра.", flush=True)
        _remove_empty_directory(session_dir)
    else:
        print("Запись не запускалась; файлы не создавались.", flush=True)

    if capture_error is not None:
        log_path = _write_capture_error_log(capture_error, session_dir=session_dir)
        raise RuntimeError(
            "Диагностическая запись завершилась с ошибкой: "
            f"{type(capture_error).__name__}: {capture_error}. "
            f"Полный traceback сохранён в {log_path}"
        ) from capture_error
    return 0


def generate_quality_diagnostics(session: Path) -> int:
    """Пересчитывает причины непригодности для уже записанной сессии."""
    session_dir = _resolve_session_dir(session)
    legacy_samples = load_samples(session_dir / "signals.csv")
    metadata = _load_metadata(session_dir)
    calibration_sec = float(metadata.get("calibration_sec", 5.0))
    thresholds = _quality_thresholds_from_metadata(metadata)
    samples, pose_reference, pose_reference_error = _personalize_pose_samples(
        legacy_samples,
        metadata=metadata,
        calibration_sec=calibration_sec,
        thresholds=thresholds,
    )
    payload = _build_quality_payload(
        samples,
        calibration_sec=calibration_sec,
        thresholds=thresholds,
        head_pose_reference=pose_reference,
        pose_reference_error=pose_reference_error,
        legacy_samples=legacy_samples,
    )

    _atomic_json(session_dir / "quality_report.json", payload)
    (session_dir / "quality_report.txt").write_text(
        _format_quality_report_text(payload),
        encoding="utf-8",
    )

    all_summary = payload["all_recording"]
    test_summary = payload["blink_test_phase"]
    assert isinstance(all_summary, dict)
    assert isinstance(test_summary, dict)
    metadata["schema_version"] = max(int(metadata.get("schema_version", 1)), 3)
    metadata["valid_sample_count"] = int(all_summary["valid_frame_count"])
    metadata["signal_coverage"] = float(all_summary["frame_coverage"])
    metadata["blink_test_signal_coverage"] = float(test_summary["frame_coverage"])
    metadata["quality_status"] = str(payload["status"])
    metadata["quality_thresholds"] = all_summary["thresholds"]
    metadata["quality_report_file"] = "quality_report.json"
    metadata["head_pose_model"] = (
        "personal_calibrated" if pose_reference is not None else "broad_fallback"
    )
    metadata["head_pose_reference"] = (
        _head_pose_reference_payload(pose_reference)
        if pose_reference is not None
        else None
    )
    metadata["head_pose_reference_error"] = pose_reference_error
    _atomic_json(session_dir / "metadata.json", metadata)

    print(f"Сессия: {session_dir}", flush=True)
    _print_quality_summary(payload)
    print(f"Подробный отчёт: {session_dir / 'quality_report.txt'}", flush=True)
    if payload["status"] == "pass":
        print("Сигнал допускается к ручной разметке этой сессии.", flush=True)
    else:
        print(
            "Сначала устрани главную причину браковки и повтори запись.",
            flush=True,
        )
    return 0


def review_session(session: Path) -> int:
    session_dir = _resolve_session_dir(session)
    raw_samples = load_samples(session_dir / "signals.csv")
    metadata = _load_metadata(session_dir)
    calibration_sec = float(metadata.get("calibration_sec", 5.0))
    samples, _pose_reference, _pose_error = _personalize_pose_samples(
        raw_samples,
        metadata=metadata,
        calibration_sec=calibration_sec,
        thresholds=_quality_thresholds_from_metadata(metadata),
    )
    video_file = metadata.get("video_file") or "validation_video.avi"
    video_path = session_dir / str(video_file)
    if not video_path.exists():
        raise ValidationDataError(
            "Видео уже удалено или не найдено. Без него ручная разметка невозможна."
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(len(samples), video_frame_count) if video_frame_count > 0 else len(samples)
    if frame_count <= 0:
        capture.release()
        raise ValidationDataError("В сессии нет кадров для разметки.")

    labels_path = session_dir / "labels.json"
    labels = load_labels(labels_path)
    history: list[list[ManualBlinkLabel]] = []
    current_frame = 0
    playing = False
    trackbar_position = [0]
    internal_trackbar_update = [False]

    def on_trackbar(value: int) -> None:
        if not internal_trackbar_update[0]:
            trackbar_position[0] = value

    cv2.namedWindow(WINDOW_REVIEW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", WINDOW_REVIEW, 0, frame_count - 1, on_trackbar)

    try:
        while True:
            if trackbar_position[0] != current_frame:
                current_frame = trackbar_position[0]
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Не удалось прочитать кадр {current_frame}.")

            sample = samples[current_frame]
            display = _build_review_display(
                frame,
                sample=sample,
                labels=labels,
                playing=playing,
            )
            cv2.imshow(WINDOW_REVIEW, display)
            delay = 1 if playing else 30
            key = cv2.waitKeyEx(delay)

            if playing:
                current_frame = min(frame_count - 1, current_frame + 1)
                if current_frame >= frame_count - 1:
                    playing = False

            if key in {10, 13, 27}:
                break
            if key == 32:
                playing = not playing
            elif key in {2424832, 81}:
                playing = False
                current_frame = max(0, current_frame - 1)
            elif key in {2555904, 83}:
                playing = False
                current_frame = min(frame_count - 1, current_frame + 1)
            elif key == ord("1"):
                history.append(list(labels))
                labels = _replace_nearby_label(
                    labels,
                    ManualBlinkLabel(
                        frame_index=current_frame,
                        timestamp_sec=sample.timestamp_sec,
                        kind=ValidationLabelKind.FULL_BLINK,
                    ),
                )
                save_labels(labels_path, labels)
            elif key == ord("2"):
                history.append(list(labels))
                labels = _replace_nearby_label(
                    labels,
                    ManualBlinkLabel(
                        frame_index=current_frame,
                        timestamp_sec=sample.timestamp_sec,
                        kind=ValidationLabelKind.INCOMPLETE_BLINK,
                    ),
                )
                save_labels(labels_path, labels)
            elif key == ord("3"):
                history.append(list(labels))
                labels = _replace_nearby_label(
                    labels,
                    ManualBlinkLabel(
                        frame_index=current_frame,
                        timestamp_sec=sample.timestamp_sec,
                        kind=ValidationLabelKind.LEFT_WINK,
                    ),
                )
                save_labels(labels_path, labels)
            elif key == ord("4"):
                history.append(list(labels))
                labels = _replace_nearby_label(
                    labels,
                    ManualBlinkLabel(
                        frame_index=current_frame,
                        timestamp_sec=sample.timestamp_sec,
                        kind=ValidationLabelKind.RIGHT_WINK,
                    ),
                )
                save_labels(labels_path, labels)
            elif key in {8, 127, 3014656, 65535}:
                history.append(list(labels))
                labels = _delete_nearest_label(labels, current_frame, radius_frames=4)
                save_labels(labels_path, labels)
            elif key == ord("0") and history:
                labels = history.pop()
                save_labels(labels_path, labels)

            if not _window_is_visible(WINDOW_REVIEW):
                break

            internal_trackbar_update[0] = True
            cv2.setTrackbarPos("frame", WINDOW_REVIEW, current_frame)
            trackbar_position[0] = current_frame
            internal_trackbar_update[0] = False

    finally:
        save_labels(labels_path, labels)
        capture.release()
        cv2.destroyAllWindows()

    bilateral_count = sum(label.kind.is_bilateral for label in labels)
    print(f"Разметка сохранена: {labels_path}")
    print(f"Двусторонних морганий размечено: {bilateral_count}")
    print("Следующая команда:")
    print(f'python ocular_validation.py report "{session_dir}"')
    return 0


def generate_report(
    session: Path,
    *,
    matching_tolerance_sec: float,
) -> int:
    session_dir = _resolve_session_dir(session)
    raw_samples = load_samples(session_dir / "signals.csv")
    labels = load_labels(session_dir / "labels.json")
    metadata = _load_metadata(session_dir)
    calibration_sec = float(metadata.get("calibration_sec", 5.0))
    samples, _pose_reference, pose_error = _personalize_pose_samples(
        raw_samples,
        metadata=metadata,
        calibration_sec=calibration_sec,
        thresholds=_quality_thresholds_from_metadata(metadata),
    )
    if pose_error is not None:
        raise ValidationDataError(
            "Нельзя строить отчёт морганий: личная калибровка ракурса "
            f"не получена. {pose_error}"
        )

    reference = estimate_open_eye_reference(
        samples,
        calibration_sec=calibration_sec,
    )
    normalized = normalize_samples(samples, reference)
    default_config = OfflineBlinkConfig()
    default_events = detect_offline_blinks(
        normalized,
        default_config,
        start_after_sec=calibration_sec,
    )
    default_metrics = calculate_validation_metrics(
        default_events,
        [label for label in labels if label.timestamp_sec > calibration_sec],
        tolerance_sec=matching_tolerance_sec,
    )
    optimized = optimize_offline_thresholds(
        normalized,
        labels,
        calibration_sec=calibration_sec,
        default_config=default_config,
        tolerance_sec=matching_tolerance_sec,
    )
    optimized_events = detect_offline_blinks(
        normalized,
        optimized.config,
        start_after_sec=calibration_sec,
    )

    save_normalized_samples(session_dir / "normalized_signals.csv", normalized)
    save_events(session_dir / "events_default.json", default_events)
    save_events(session_dir / "events_optimized.json", optimized_events)
    payload = build_report_payload(
        reference=reference,
        default_config=default_config,
        default_metrics=default_metrics,
        optimized=optimized,
        labels=labels,
        samples=normalized,
        calibration_sec=calibration_sec,
    )
    save_report_files(session_dir, payload)
    render_validation_plot(
        session_dir / "signal_plot.png",
        normalized,
        labels,
        optimized_events,
        optimized.config,
        calibration_sec=calibration_sec,
    )

    print(f"Отчёт создан: {session_dir / 'report.txt'}")
    print(f"График: {session_dir / 'signal_plot.png'}")
    print("Параметры, подобранные только для этой сессии:")
    print(
        "  full="
        f"{optimized.config.full_blink_ratio_threshold:.2f}, "
        "partial="
        f"{optimized.config.partial_ratio_threshold:.2f}, "
        "open="
        f"{optimized.config.open_ratio_threshold:.2f}, "
        "smoothing="
        f"{optimized.config.smoothing_alpha:.2f}"
    )
    print(
        "  precision="
        f"{_format_metric(optimized.metrics.precision)}, "
        "recall="
        f"{_format_metric(optimized.metrics.recall)}, "
        "F1="
        f"{_format_metric(optimized.metrics.f1)}"
    )
    if not optimized.classification_threshold_validated:
        print(
            "Порог full/incomplete НЕ подтверждён: нужны как минимум "
            "2 размеченных полных и 2 намеренно неполных моргания."
        )
    print(
        "Порог не переносится в main.py автоматически. Сначала нужны "
        "повторные сессии и проверка на других условиях."
    )
    return 0


def delete_video(session: Path, *, force: bool) -> int:
    session_dir = _resolve_session_dir(session)
    metadata = _load_metadata(session_dir)
    video_file = metadata.get("video_file") or "validation_video.avi"
    video_path = session_dir / str(video_file)
    if not video_path.exists():
        print("Видео уже отсутствует.")
        return 0
    if not force and not (session_dir / "labels.json").exists():
        raise ValidationDataError(
            "Разметка ещё не создана. Сначала выполни review либо добавь --force."
        )
    video_path.unlink()
    metadata["video_deleted_at"] = datetime.now().astimezone().isoformat()
    _atomic_json(session_dir / "metadata.json", metadata)
    print(f"Видео удалено: {video_path}")
    return 0


def list_sessions(sessions_root: Path) -> int:
    root = sessions_root.expanduser().resolve()
    if not root.exists():
        print("Валидационных сессий пока нет.")
        return 0
    directories = sorted(
        (item for item in root.iterdir() if item.is_dir()),
        reverse=True,
    )
    if not directories:
        print("Валидационных сессий пока нет.")
        return 0
    for directory in directories:
        signals = "signals" if (directory / "signals.csv").exists() else "no-signals"
        labels = "labels" if (directory / "labels.json").exists() else "no-labels"
        report = "report" if (directory / "report.json").exists() else "no-report"
        video = "video" if (directory / "validation_video.avi").exists() else "no-video"
        print(f"{directory}  [{signals}, {labels}, {report}, {video}]")
    return 0


def _build_capture_display(
    frame: np.ndarray,
    *,
    recording: bool,
    elapsed: float,
    calibration_sec: float,
    measured_fps: float | None,
    valid: bool,
    valid_count: int,
    sample_count: int,
    attention: VisualAttentionSnapshot,
    data_quality: float,
    button_rects: dict[str, tuple[int, int, int, int]],
    quality_assessment: EyeSampleQualityAssessment | None = None,
    failure_reason_counts: Counter[str] | None = None,
    effective_head_pose_quality: float | None = None,
    pose_reference_ready: bool = False,
) -> np.ndarray:
    panel_width = 560
    height, width = frame.shape[:2]
    display = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    display[:, :width] = frame
    display[:, width:] = (28, 28, 28)

    if recording:
        cv2.rectangle(display, (2, 2), (width - 3, height - 3), (0, 0, 255), 4)
        cv2.circle(display, (width + 28, 28), 9, (0, 0, 255), -1, cv2.LINE_AA)

    x = width + 22
    coverage = valid_count / sample_count if sample_count else 0.0
    left = attention.left_eye_opening_ratio
    right = attention.right_eye_opening_ratio

    if not recording:
        title = "READY - NOT RECORDING"
        title_color = (235, 235, 235)
        instruction = "Click START or press SPACE once"
        phase_detail = "No files are being created"
    elif elapsed < calibration_sec:
        remaining = max(0.0, calibration_sec - elapsed)
        title = "REC  |  CALIBRATION"
        title_color = (0, 190, 255)
        instruction = "Keep eyes normally open"
        phase_detail = f"Blink test starts in {remaining:4.1f} s"
    else:
        title = "REC  |  BLINK TEST"
        title_color = (90, 230, 90)
        instruction = "Blink normally at least 20 times"
        phase_detail = "Press ENTER or click SAVE when done"

    cv2.putText(
        display,
        title,
        (x, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.73,
        title_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        instruction,
        (x, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        phase_detail,
        (x, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if recording and elapsed < calibration_sec:
        progress = min(1.0, elapsed / calibration_sec)
        bar_x1, bar_y1 = x, 112
        bar_x2, bar_y2 = width + panel_width - 24, 126
        cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 1)
        fill_x = int(bar_x1 + (bar_x2 - bar_x1) * progress)
        cv2.rectangle(display, (bar_x1, bar_y1), (fill_x, bar_y2), (0, 190, 255), -1)

    current_reject = _format_current_rejection(quality_assessment)
    top_failures = _top_failure_lines(
        failure_reason_counts or Counter(),
        sample_count=sample_count,
        limit=3,
    )
    coverage_color = (90, 230, 90) if coverage >= QUALITY_TARGET_COVERAGE else (0, 180, 255)

    details = [
        f"Elapsed: {elapsed:6.2f} s  |  samples: {sample_count}",
        f"Processed FPS: {_format_metric(measured_fps)}",
        f"Current frame: {'VALID' if valid else 'INVALID'}",
        f"Reject now: {current_reject}",
        f"Coverage: {coverage:5.1%}  |  target: {QUALITY_TARGET_COVERAGE:.0%}",
        (
            "Image / head quality: "
            f"{data_quality:.2f} / "
            f"{(effective_head_pose_quality if effective_head_pose_quality is not None else attention.head_pose_quality):.2f}"
        ),
        f"Head-pose model: {'PERSONAL' if pose_reference_ready else 'CALIBRATING/FALLBACK'}",
        (
            "Eye quality L/R: "
            f"{attention.left_eye_quality:.2f} / {attention.right_eye_quality:.2f}"
        ),
        f"Eye raw L/R: {_format_metric(left)} / {_format_metric(right)}",
        "Rejection rates (overlap allowed):",
        *top_failures,
        "Live blink count: OFF; use REVIEW + REPORT",
    ]
    y = 146
    for index, line in enumerate(details):
        color = (225, 225, 225)
        if line.startswith("Coverage:"):
            color = coverage_color
        elif line.startswith("Current frame:") and not valid:
            color = (0, 180, 255)
        elif line.startswith("Reject now:") and not valid:
            color = (0, 180, 255)
        cv2.putText(
            display,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 22

    button_rects.clear()
    button_y1 = max(10, height - 58)
    button_y2 = height - 14
    button_x1 = x
    button_x2 = width + panel_width - 24
    if recording:
        button_rects["save"] = (button_x1, button_y1, button_x2, button_y2)
        _draw_button(
            display,
            button_rects["save"],
            label="SAVE SESSION  (ENTER)",
            color=(35, 65, 180),
        )
    else:
        button_rects["start"] = (button_x1, button_y1, button_x2, button_y2)
        _draw_button(
            display,
            button_rects["start"],
            label="START RECORDING  (SPACE)",
            color=(45, 135, 55),
        )
    return display


def _format_current_rejection(
    assessment: EyeSampleQualityAssessment | None,
) -> str:
    if assessment is None or assessment.valid:
        return "none"
    labels = [
        QUALITY_REASON_LABELS_EN.get(reason.value, reason.value)
        for reason in assessment.failed_reasons
    ]
    if len(labels) <= 2:
        return "; ".join(labels)
    return "; ".join(labels[:2]) + f"; +{len(labels) - 2} more"


def _top_failure_lines(
    counts: Counter[str],
    *,
    sample_count: int,
    limit: int,
) -> list[str]:
    if sample_count <= 0 or not counts:
        return ["  no recorded failures yet"]
    ranked = sorted(
        ((reason, count) for reason, count in counts.items() if count > 0),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    if not ranked:
        return ["  no recorded failures yet"]
    return [
        f"  {QUALITY_REASON_LABELS_EN.get(reason, reason)}: {count / sample_count:5.1%}"
        for reason, count in ranked
    ]


def _build_quality_payload(
    samples: Sequence[ValidationSample],
    *,
    calibration_sec: float,
    thresholds: EyeSampleQualityThresholds,
    head_pose_reference: HeadPoseReference | None = None,
    pose_reference_error: str | None = None,
    legacy_samples: Sequence[ValidationSample] | None = None,
) -> dict[str, object]:
    all_summary = summarize_eye_sample_quality(
        samples,
        thresholds=thresholds,
    )
    blink_test_summary = summarize_eye_sample_quality(
        samples,
        thresholds=thresholds,
        start_after_sec=calibration_sec,
    )
    status = _quality_status(blink_test_summary)
    if head_pose_reference is None:
        status = "insufficient_data"

    legacy_blink_summary = None
    if legacy_samples is not None:
        legacy_blink_summary = summarize_eye_sample_quality(
            legacy_samples,
            thresholds=thresholds,
            start_after_sec=calibration_sec,
        )

    return {
        "schema_version": 2,
        "target_frame_coverage": QUALITY_TARGET_COVERAGE,
        "status": status,
        "head_pose_model": (
            "personal_calibrated"
            if head_pose_reference is not None
            else "unavailable"
        ),
        "head_pose_reference": (
            _head_pose_reference_payload(head_pose_reference)
            if head_pose_reference is not None
            else None
        ),
        "head_pose_reference_error": pose_reference_error,
        "legacy_blink_test_phase": legacy_blink_summary,
        "all_recording": all_summary,
        "blink_test_phase": blink_test_summary,
        "interpretation": (
            "Частоты отдельных причин могут суммироваться больше 100%, "
            "потому что один кадр способен провалить несколько критериев. "
            "Ракурс головы оценивается относительно личной нейтральной позы "
            "из первых секунд, а не относительно универсальной пропорции лица."
        ),
    }


def _head_pose_reference_payload(reference: HeadPoseReference) -> dict[str, object]:
    return {
        "neutral_turn_ratio": reference.neutral_turn_ratio,
        "neutral_vertical_ratio": reference.neutral_vertical_ratio,
        "turn_mad": reference.turn_mad,
        "vertical_mad": reference.vertical_mad,
        "sample_count": reference.sample_count,
        "calibration_duration_sec": reference.calibration_duration_sec,
    }


def _head_pose_reference_from_metadata(
    metadata: dict[str, object],
) -> HeadPoseReference | None:
    raw = metadata.get("head_pose_reference")
    if not isinstance(raw, dict):
        return None
    try:
        return HeadPoseReference(
            neutral_turn_ratio=float(raw["neutral_turn_ratio"]),
            neutral_vertical_ratio=float(raw["neutral_vertical_ratio"]),
            turn_mad=float(raw.get("turn_mad", 0.0)),
            vertical_mad=float(raw.get("vertical_mad", 0.0)),
            sample_count=int(raw.get("sample_count", 1)),
            calibration_duration_sec=float(raw.get("calibration_duration_sec", 5.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _personalize_pose_samples(
    samples: Sequence[ValidationSample],
    *,
    metadata: dict[str, object],
    calibration_sec: float,
    thresholds: EyeSampleQualityThresholds,
) -> tuple[list[ValidationSample], HeadPoseReference | None, str | None]:
    reference = _head_pose_reference_from_metadata(metadata)
    try:
        if reference is None:
            reference = estimate_head_pose_reference(
                samples,
                calibration_duration_sec=calibration_sec,
            )
        personalized = apply_personal_head_pose_quality(
            samples,
            reference=reference,
            thresholds=thresholds,
        )
        return personalized, reference, None
    except ValidationDataError as error:
        return list(samples), None, str(error)


def _quality_thresholds_from_metadata(
    metadata: dict[str, object],
) -> EyeSampleQualityThresholds:
    raw = metadata.get("quality_thresholds")
    if not isinstance(raw, dict):
        return EyeSampleQualityThresholds()
    try:
        return EyeSampleQualityThresholds(
            min_data_quality=float(raw.get("min_data_quality", 0.55)),
            min_eye_quality=float(raw.get("min_eye_quality", 0.55)),
            min_head_pose_quality=float(raw.get("min_head_pose_quality", 0.35)),
        )
    except (TypeError, ValueError):
        raise ValidationDataError(
            "В metadata.json сохранены некорректные пороги качества."
        )


def _quality_status(blink_test_summary: dict[str, object]) -> str:
    total = int(blink_test_summary.get("total_frame_count", 0))
    coverage = float(blink_test_summary.get("frame_coverage", 0.0))
    if total <= 0:
        return "insufficient_data"
    return "pass" if coverage >= QUALITY_TARGET_COVERAGE else "fail"


def _sorted_reason_stats(summary: dict[str, object]) -> list[tuple[str, int, float]]:
    counts = summary.get("reason_counts")
    percentages = summary.get("reason_percent_of_frames")
    if not isinstance(counts, dict) or not isinstance(percentages, dict):
        return []
    rows: list[tuple[str, int, float]] = []
    for reason, raw_count in counts.items():
        count = int(raw_count)
        if count <= 0:
            continue
        rows.append((str(reason), count, float(percentages.get(reason, 0.0))))
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def _format_quality_report_text(payload: dict[str, object]) -> str:
    all_summary = payload.get("all_recording")
    test_summary = payload.get("blink_test_phase")
    if not isinstance(all_summary, dict) or not isinstance(test_summary, dict):
        raise ValueError("Некорректный quality report payload.")

    status = str(payload.get("status", "unknown")).upper()
    target = float(payload.get("target_frame_coverage", QUALITY_TARGET_COVERAGE))
    lines = [
        "ОТЧЁТ О ПРИГОДНОСТИ ГЛАЗНОГО СИГНАЛА",
        "",
        f"Статус: {status}",
        f"Целевое покрытие: {target:.1%}",
        (
            "Покрытие всей записи: "
            f"{float(all_summary.get('frame_coverage', 0.0)):.1%} "
            f"({int(all_summary.get('valid_frame_count', 0))}/"
            f"{int(all_summary.get('total_frame_count', 0))} кадров)"
        ),
        (
            "Покрытие после калибровки: "
            f"{float(test_summary.get('frame_coverage', 0.0)):.1%} "
            f"({int(test_summary.get('valid_frame_count', 0))}/"
            f"{int(test_summary.get('total_frame_count', 0))} кадров)"
        ),
    ]

    legacy_summary = payload.get("legacy_blink_test_phase")
    if isinstance(legacy_summary, dict):
        lines.append(
            "Покрытие по прежней абсолютной формуле ракурса: "
            f"{float(legacy_summary.get('frame_coverage', 0.0)):.1%}"
        )

    reference = payload.get("head_pose_reference")
    if isinstance(reference, dict):
        lines.extend(
            [
                "",
                "Персональная норма ракурса головы:",
                f"- нейтральный поворот: {float(reference.get('neutral_turn_ratio', 0.0)):.4f}",
                f"- нейтральная вертикальная пропорция: {float(reference.get('neutral_vertical_ratio', 0.0)):.4f}",
                f"- кадров калибровки: {int(reference.get('sample_count', 0))}",
            ]
        )
    else:
        error = payload.get("head_pose_reference_error")
        lines.extend(
            [
                "",
                "Персональная норма ракурса не рассчитана.",
                f"Причина: {error or 'неизвестна'}",
            ]
        )

    lines.extend(
        [
        "",
        "Причины непригодности после калибровки:",
        "Проценты могут суммироваться больше 100%: причины пересекаются.",
        ]
    )
    rows = _sorted_reason_stats(test_summary)
    if not rows:
        lines.append("- причин непригодности не зафиксировано")
    else:
        for reason, count, fraction in rows:
            label = QUALITY_REASON_LABELS_RU.get(reason, reason)
            lines.append(f"- {label}: {count} кадров ({fraction:.1%} всех кадров этапа)")

    thresholds = test_summary.get("thresholds")
    if isinstance(thresholds, dict):
        lines.extend(
            [
                "",
                "Использованные пороги:",
                f"- качество изображения: {float(thresholds.get('min_data_quality', 0.0)):.2f}",
                f"- качество каждого глаза: {float(thresholds.get('min_eye_quality', 0.0)):.2f}",
                f"- качество ракурса головы: {float(thresholds.get('min_head_pose_quality', 0.0)):.2f}",
            ]
        )
    return "\n".join(lines) + "\n"


def _print_quality_summary(payload: dict[str, object]) -> None:
    test_summary = payload.get("blink_test_phase")
    if not isinstance(test_summary, dict):
        return
    status = str(payload.get("status", "unknown")).upper()
    coverage = float(test_summary.get("frame_coverage", 0.0))
    valid = int(test_summary.get("valid_frame_count", 0))
    total = int(test_summary.get("total_frame_count", 0))
    print()
    print(f"=== КАЧЕСТВО СЕССИИ: {status} ===", flush=True)
    print(
        f"Покрытие после калибровки: {coverage:.1%} "
        f"({valid}/{total}), цель: не ниже {QUALITY_TARGET_COVERAGE:.0%}",
        flush=True,
    )
    legacy_summary = payload.get("legacy_blink_test_phase")
    if isinstance(legacy_summary, dict):
        legacy_coverage = float(legacy_summary.get("frame_coverage", 0.0))
        print(
            "Прежняя абсолютная формула давала: "
            f"{legacy_coverage:.1%}",
            flush=True,
        )
    reference = payload.get("head_pose_reference")
    if isinstance(reference, dict):
        print(
            "Личная норма ракурса: "
            f"turn={float(reference.get('neutral_turn_ratio', 0.0)):.3f}, "
            f"vertical={float(reference.get('neutral_vertical_ratio', 0.0)):.3f}, "
            f"кадров={int(reference.get('sample_count', 0))}",
            flush=True,
        )
    else:
        print(
            "Личная норма ракурса не получена: "
            f"{payload.get('head_pose_reference_error') or 'неизвестная причина'}",
            flush=True,
        )
    rows = _sorted_reason_stats(test_summary)
    if rows:
        print("Причины браковки (могут пересекаться):", flush=True)
        for reason, count, fraction in rows[:5]:
            label = QUALITY_REASON_LABELS_RU.get(reason, reason)
            print(f"- {label}: {count} кадров ({fraction:.1%})", flush=True)
    else:
        print("Причины браковки не зафиксированы.", flush=True)


def _build_review_display(
    frame: np.ndarray,
    *,
    sample: ValidationSample,
    labels: Sequence[ManualBlinkLabel],
    playing: bool,
) -> np.ndarray:
    panel_width = 500
    height, width = frame.shape[:2]
    display = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    display[:, :width] = frame
    display[:, width:] = (32, 32, 32)
    x = width + 18

    nearest = [
        label
        for label in labels
        if abs(label.frame_index - sample.frame_index) <= 2
    ]
    current_label = nearest[0].kind.value if nearest else "none"
    counts = {
        kind: sum(label.kind is kind for label in labels)
        for kind in ValidationLabelKind
    }
    lines = [
        ("MANUAL LABELING", 0.72),
        (f"Frame: {sample.frame_index}", 0.53),
        (f"Time: {sample.timestamp_sec:.3f} s", 0.53),
        (f"Playing: {'yes' if playing else 'no'}", 0.53),
        (f"Current label: {current_label}", 0.50),
        ("", 0.45),
        (f"1 full blink: {counts[ValidationLabelKind.FULL_BLINK]}", 0.49),
        (f"2 incomplete: {counts[ValidationLabelKind.INCOMPLETE_BLINK]}", 0.49),
        (f"3 left wink: {counts[ValidationLabelKind.LEFT_WINK]}", 0.49),
        (f"4 right wink: {counts[ValidationLabelKind.RIGHT_WINK]}", 0.49),
        ("", 0.45),
        ("Space       play/pause", 0.47),
        ("Arrow keys  previous/next frame", 0.44),
        ("1/2/3/4     label deepest frame", 0.44),
        ("Backspace   delete nearby label", 0.44),
        ("0           undo", 0.47),
        ("Enter/Esc   save and exit", 0.44),
    ]
    y = 34
    for text, scale in lines:
        if text:
            cv2.putText(
                display,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        y += 27
    return display


def _replace_nearby_label(
    labels: Sequence[ManualBlinkLabel],
    new_label: ManualBlinkLabel,
    *,
    radius_frames: int = 3,
) -> list[ManualBlinkLabel]:
    result = [
        label
        for label in labels
        if abs(label.frame_index - new_label.frame_index) > radius_frames
    ]
    result.append(new_label)
    result.sort(key=lambda item: item.timestamp_sec)
    return result


def _delete_nearest_label(
    labels: Sequence[ManualBlinkLabel],
    frame_index: int,
    *,
    radius_frames: int,
) -> list[ManualBlinkLabel]:
    candidates = [
        (abs(label.frame_index - frame_index), index)
        for index, label in enumerate(labels)
        if abs(label.frame_index - frame_index) <= radius_frames
    ]
    if not candidates:
        return list(labels)
    _, delete_index = min(candidates)
    return [label for index, label in enumerate(labels) if index != delete_index]


def _capture_action_from_key(key: int, *, recording: bool) -> str:
    """Map layout-independent keys to capture actions."""

    if key == 32 and not recording:  # Space
        return "start"
    if key in {10, 13} and recording:  # Enter
        return "save"
    if key == 27:  # Escape
        return "close"
    return "none"


def _point_in_rect(
    x: int,
    y: int,
    rect: tuple[int, int, int, int] | None,
) -> bool:
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _draw_button(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (230, 230, 230), 1)
    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        1,
    )
    text_x = x1 + max(8, (x2 - x1 - text_width) // 2)
    text_y = y1 + max(text_height + 4, (y2 - y1 + text_height) // 2)
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _window_is_visible(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1.0
    except cv2.error:
        return False


def _write_capture_error_log(
    error: Exception,
    *,
    session_dir: Path | None,
) -> Path:
    log_path = (session_dir or PROJECT_ROOT) / "last_capture_error.txt"
    details = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    log_path.write_text(details, encoding="utf-8")
    return log_path


def _open_video_writer(
    path: Path,
    *,
    target_fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("MJPG", "XVID"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            target_fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("OpenCV не смог создать локальный AVI-файл.")


def _new_session_directory(root: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = resolved_root / base_name
    suffix = 1
    while candidate.exists():
        candidate = resolved_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False)
    return candidate


def _resolve_session_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValidationDataError(f"Не найдена папка сессии: {resolved}")
    return resolved


def _load_metadata(session_dir: Path) -> dict[str, object]:
    path = session_dir / "metadata.json"
    if not path.exists():
        raise ValidationDataError(f"Не найден metadata.json: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationDataError("metadata.json повреждён.") from error
    if not isinstance(payload, dict):
        raise ValidationDataError("metadata.json должен содержать объект.")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _timestamp_rate(timestamps: deque[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0.0:
        return None
    return (len(timestamps) - 1) / elapsed


def _sample_rate_from_validation_samples(
    samples: Sequence[ValidationSample],
) -> float | None:
    if len(samples) < 2:
        return None
    elapsed = samples[-1].timestamp_sec - samples[0].timestamp_sec
    if elapsed <= 0.0:
        return None
    return (len(samples) - 1) / elapsed


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _empty_attention() -> VisualAttentionSnapshot:
    return VisualAttentionSnapshot(
        gaze_on_screen=None,
        face_landmarks_detected=False,
        horizontal_gaze_ratio=None,
        head_turn_ratio=None,
        head_vertical_ratio=None,
        reason="frame_unavailable",
    )


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
