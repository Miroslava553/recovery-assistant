from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

from app_core.face_presence import FacePresenceSensor, FacePresenceSnapshot
from app_core.input_activity import InputActivityError, WindowsInputActivitySensor
from app_core.ocular_activity import EyeActivityTracker, OcularMetrics
from app_core.screen_lock import ScreenLockSensor
from app_core.state_machine import PresenceSignals, UserState, UserStateManager
from app_core.typing_activity import (
    GlobalKeyboardTimingSensor,
    TypingActivityCollector,
    TypingMetrics,
)
from app_core.typing_baseline import TypingBaselineResult, TypingBaselineService
from app_core.visual_attention import VisualAttentionAnalyzer, VisualAttentionSnapshot


STATE_LABELS: dict[UserState, str] = {
    UserState.UNKNOWN: "Определяю состояние…",
    UserState.ACTIVE_WORK: "Работа с вводом",
    UserState.PASSIVE_WORK: "Чтение или просмотр",
    UserState.AWAY: "Пользователь отошёл",
    UserState.BREAK: "Перерыв",
}

CAMERA_STATUS_LABELS: dict[str, str] = {
    "face_detected": "Лицо обнаружено",
    "face_not_detected": "Лицо не видно",
    "too_dark": "Недостаточно света",
    "too_bright": "Слишком яркое освещение",
    "image_quality_too_low": "Изображение недостаточно чёткое",
    "camera_frame_unavailable": "Камера не передаёт изображение",
    "camera_unavailable": "Камера недоступна; работа без визуального канала",
}

ATTENTION_STATUS_LABELS: dict[str, str] = {
    "looking_toward_screen": "Взгляд направлен к экрану",
    "head_turned_away": "Голова повёрнута в сторону",
    "eyes_turned_away": "Взгляд отведён от экрана",
    "head_position_outside_range": "Положение головы не позволяет определить внимание",
    "face_landmarks_not_detected": "Не удалось определить направление взгляда",
    "face_geometry_unreliable": "Положение лица распознано недостаточно надёжно",
    "image_quality_too_low": "Недостаточно качества для анализа взгляда",
    "frame_unavailable": "Изображение камеры недоступно",
    "face_not_available": "Лицо не видно",
}

OCULAR_STATUS_LABELS: dict[str, str] = {
    "not_started": "Анализ глаз ещё не начат",
    "calibrating_open_eyes": "Калибруется личное раскрытие глаз",
    "eyes_open": "Глаза открыты",
    "blink_candidate": "Проверяется движение век",
    "bilateral_closure": "Оба глаза закрыты",
    "face_not_available": "Лицо не видно",
    "eye_data_quality_too_low": "Недостаточно качества для анализа глаз",
    "left_eye_quality_too_low": "Левый глаз виден недостаточно надёжно",
    "right_eye_quality_too_low": "Правый глаз виден недостаточно надёжно",
    "head_pose_too_extreme": "Ракурс головы слишком сложный для анализа глаз",
    "eye_geometry_unavailable": "Не удалось измерить раскрытие обоих глаз",
    "eye_geometry_unreliable": "Геометрия глаз распознана ненадёжно",
    "processing_fps_too_low": "Частота обработки недостаточна для надёжного счёта",
    "eye_signal_coverage_too_low": "Слишком мало пригодных данных глаз",
    "reset": "Окно анализа глаз сброшено",
}


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    captured_at: float
    captured_datetime: datetime
    state: UserState
    state_label: str
    state_reason: str
    seconds_in_state: float
    camera_status: str
    attention_status: str
    ocular_status: str
    face_detected: bool | None
    gaze_on_screen: bool | None
    input_idle_sec: float | None
    fatigue_analysis_allowed: bool
    baseline_update_allowed: bool
    typing_metrics: TypingMetrics
    typing_baseline: TypingBaselineResult | None
    ocular_metrics: OcularMetrics
    visual_age_sec: float
    frame: Any
    face_box: tuple[int, int, int, int] | None


@dataclass(frozen=True, slots=True)
class _VisualBundle:
    captured_at: float
    face: FacePresenceSnapshot
    attention: VisualAttentionSnapshot
    ocular: OcularMetrics


class SessionMonitor:
    """
    Объединяет рабочее состояние, клавиатуру и высокочастотный визуальный канал.

    Камера и Face Mesh обрабатываются в отдельном потоке с целевой частотой.
    Поэтому счёт морганий не зависит от того, как часто интерфейс запрашивает
    SessionSnapshot. Сырые кадры не сохраняются.
    """

    def __init__(
        self,
        *,
        face_sensor: FacePresenceSensor | None = None,
        input_sensor: WindowsInputActivitySensor | None = None,
        typing_sensor: GlobalKeyboardTimingSensor | None = None,
        attention_analyzer: VisualAttentionAnalyzer | None = None,
        ocular_tracker: EyeActivityTracker | None = None,
        state_manager: UserStateManager | None = None,
        typing_baseline_service: TypingBaselineService | None = None,
        target_visual_fps: float = 30.0,
        background_visual_processing: bool = True,
        include_diagnostic_frame: bool = False,
        screen_lock_sensor: ScreenLockSensor | None = None,
        vision_required: bool = True,
    ) -> None:
        if target_visual_fps <= 0:
            raise ValueError("target_visual_fps должен быть больше нуля.")

        self.face_sensor = face_sensor or FacePresenceSensor(
            camera_index=0,
            min_detection_confidence=0.55,
            run_face_detection=False,
        )
        self.input_sensor = input_sensor or WindowsInputActivitySensor(
            recent_threshold_sec=8.0,
        )
        if typing_sensor is None:
            collector = TypingActivityCollector(
                window_sec=60.0,
                long_pause_sec=2.0,
                burst_gap_sec=1.5,
                min_observation_sec=50.0,
                min_relevant_keys=25,
            )
            typing_sensor = GlobalKeyboardTimingSensor(collector)
        self.typing_sensor = typing_sensor

        self.attention_analyzer = attention_analyzer or VisualAttentionAnalyzer(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smoothing_window=7,
        )
        self.ocular_tracker = ocular_tracker or EyeActivityTracker(
            window_sec=60.0,
            calibration_min_samples=60,
            calibration_min_sec=3.0,
            min_observation_sec=10.0,
            min_eye_quality=0.55,
            min_head_pose_quality=0.35,

            # Временные безопасные значения только для диагностического
            # монитора. Производственные пороги не подбираются вручную:
            # их проверяет ocular_validation.py на размеченных сессиях.
            closed_ratio_threshold=0.55,
            full_blink_ratio_threshold=0.82,
            partial_ratio_threshold=0.90,
            open_ratio_threshold=0.94,
            min_blink_sec=0.03,
            max_blink_sec=0.70,
            long_closure_sec=1.00,
            severe_closure_sec=2.00,
            bilateral_sync_tolerance_sec=0.12,
            max_sample_gap_sec=0.20,
            candidate_gap_tolerance_sec=0.12,
            reopen_confirmation_samples=2,
            min_reliable_fps=15.0,
            min_signal_coverage=0.70,
            smoothing_alpha=0.75,
        )
        self.state_manager = state_manager or UserStateManager(
            active_input_threshold_sec=8.0,
            away_input_threshold_sec=20.0,
            away_face_threshold_sec=12.0,
            baseline_warmup_sec=60.0,
            low_quality_threshold=0.45,
            transition_debounce_sec=2.0,
        )

        self.typing_baseline_service = typing_baseline_service
        self.target_visual_fps = float(target_visual_fps)
        self.background_visual_processing = bool(background_visual_processing)
        self.include_diagnostic_frame = bool(include_diagnostic_frame)
        self.screen_lock_sensor = screen_lock_sensor or ScreenLockSensor()
        self.vision_required = bool(vision_required)

        self._vision_available = True
        self._vision_unavailable_reason: str | None = None
        self._manual_break = False
        self._running = False
        self._closed = False
        self._workday: date | None = None
        self._workday_started_at = 0.0

        self._visual_lock = threading.Lock()
        self._visual_stop = threading.Event()
        self._visual_thread: threading.Thread | None = None
        self._visual_bundle: _VisualBundle | None = None
        self._visual_error: BaseException | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def vision_available(self) -> bool:
        """Работает ли визуальный канал в текущем запуске."""

        return self._vision_available

    @property
    def vision_unavailable_reason(self) -> str | None:
        """Почему камера не открылась, если она не открылась."""

        return self._vision_unavailable_reason

    @property
    def manual_break(self) -> bool:
        return self._manual_break

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Монитор уже был окончательно закрыт.")
        if self._running:
            return

        self._vision_available = True
        self._vision_unavailable_reason = None
        try:
            self.face_sensor.start()
        except Exception as error:  # камера занята, отсутствует или запрещена
            if self.vision_required:
                raise
            self._vision_available = False
            self._vision_unavailable_reason = str(error) or error.__class__.__name__

        try:
            self.typing_sensor.start()
        except Exception:
            if self._vision_available:
                self.face_sensor.stop()
            raise

        now = time.monotonic()
        now_datetime = datetime.now().astimezone()
        self.state_manager.reset(now=now)
        self.ocular_tracker.reset(keep_calibration=True)
        self._manual_break = False
        self._workday = now_datetime.date()
        self._workday_started_at = now
        self._visual_stop.clear()
        self._visual_error = None
        self._running = True

        if self.background_visual_processing:
            # Первый кадр получаем синхронно, чтобы snapshot() сразу имел данные.
            self._process_visual_once()
            self._visual_thread = threading.Thread(
                target=self._visual_loop,
                name="fatigue-visual-worker",
                daemon=True,
            )
            self._visual_thread.start()

    def stop(self) -> None:
        if not self._running:
            return

        self._visual_stop.set()
        visual_thread = self._visual_thread
        if visual_thread is not None and visual_thread.is_alive():
            visual_thread.join(timeout=2.0)
        self._visual_thread = None

        typing_error: BaseException | None = None
        try:
            self.typing_sensor.stop()
        except BaseException as error:
            typing_error = error
        finally:
            self.face_sensor.stop()
            self._running = False

        if typing_error is not None:
            raise RuntimeError("Не удалось корректно остановить датчик клавиатуры.") from typing_error

    def close(self) -> None:
        if self._closed:
            return
        stop_error: BaseException | None = None
        try:
            self.stop()
        except BaseException as error:
            stop_error = error
        finally:
            try:
                self.face_sensor.close()
            finally:
                self.attention_analyzer.close()
                self._closed = True
        if stop_error is not None:
            raise stop_error

    def set_manual_break(self, enabled: bool) -> None:
        self._manual_break = bool(enabled)

    def latest_visual_frame(self) -> tuple[float | None, Any]:
        """
        Возвращает последний кадр визуального потока для интерфейса.

        Кадр копируется, чтобы поток компьютерного зрения и поток Tkinter
        не использовали один и тот же массив одновременно. Метод не запускает
        новый анализ и не влияет на состояние, baseline или счётчики.
        """

        if not self._running:
            return None, None

        with self._visual_lock:
            bundle = self._visual_bundle
            if bundle is None or bundle.face.frame is None:
                return None, None
            return bundle.captured_at, bundle.face.frame.copy()

    def snapshot(self) -> SessionSnapshot:
        if not self._running:
            raise RuntimeError("Сначала запусти монитор методом start().")

        captured_at = time.monotonic()
        captured_datetime = datetime.now().astimezone()
        self._roll_workday_if_needed(captured_datetime, captured_at)

        if not self.background_visual_processing:
            self._process_visual_once()

        with self._visual_lock:
            bundle = self._visual_bundle
            visual_error = self._visual_error

        if bundle is None:
            if visual_error is not None:
                raise RuntimeError("Визуальный канал завершился с ошибкой.") from visual_error
            raise RuntimeError("Визуальный канал ещё не подготовил первый кадр.")

        face_snapshot = bundle.face
        attention_snapshot = bundle.attention
        ocular_metrics = bundle.ocular
        visual_age_sec = max(0.0, captured_at - bundle.captured_at)
        typing_metrics = self.typing_sensor.snapshot()

        try:
            input_snapshot = self.input_sensor.snapshot()
            input_idle_sec: float | None = input_snapshot.idle_seconds
        except InputActivityError:
            input_idle_sec = None

        face_detected = face_snapshot.face_detected
        gaze_on_screen = attention_snapshot.gaze_on_screen
        data_quality = face_snapshot.data_quality
        if visual_age_sec > 1.0:
            face_detected = None
            gaze_on_screen = None
            data_quality = 0.0

        signals = PresenceSignals(
            face_detected=face_detected,
            input_idle_sec=input_idle_sec,
            gaze_on_screen=gaze_on_screen,
            data_quality=data_quality,
            screen_locked=self._screen_locked(),
            manual_break=self._manual_break,
        )
        decision = self.state_manager.update(signals, now=captured_at)
        typing_baseline = self._observe_typing_baseline(
            metrics=typing_metrics,
            state=decision.state,
            captured_at=captured_at,
            captured_datetime=captured_datetime,
        )

        diagnostic_frame = face_snapshot.frame if self.include_diagnostic_frame else None
        diagnostic_face_box = face_snapshot.face_box if self.include_diagnostic_frame else None

        return SessionSnapshot(
            captured_at=captured_at,
            captured_datetime=captured_datetime,
            state=decision.state,
            state_label=STATE_LABELS[decision.state],
            state_reason=decision.reason,
            seconds_in_state=decision.seconds_in_state,
            camera_status=self._camera_status(face_snapshot),
            attention_status=self._attention_status(attention_snapshot),
            ocular_status=self._ocular_status(ocular_metrics),
            face_detected=face_detected,
            gaze_on_screen=gaze_on_screen,
            input_idle_sec=input_idle_sec,
            fatigue_analysis_allowed=decision.fatigue_analysis_allowed,
            baseline_update_allowed=decision.baseline_update_allowed,
            typing_metrics=typing_metrics,
            typing_baseline=typing_baseline,
            ocular_metrics=ocular_metrics,
            visual_age_sec=visual_age_sec,
            frame=diagnostic_frame,
            face_box=diagnostic_face_box,
        )

    def _visual_loop(self) -> None:
        period = 1.0 / self.target_visual_fps
        while not self._visual_stop.is_set():
            started_at = time.monotonic()
            try:
                self._process_visual_once()
            except BaseException as error:
                with self._visual_lock:
                    self._visual_error = error
                # Не вращаемся бесконечно на полной скорости при ошибке камеры.
                self._visual_stop.wait(0.10)
                continue
            elapsed = time.monotonic() - started_at
            remaining = period - elapsed
            if remaining > 0:
                self._visual_stop.wait(remaining)

    def _process_visual_once(self) -> None:
        captured_at = time.monotonic()
        if not self._vision_available:
            self._publish_visionless_bundle(captured_at)
            return

        face = self.face_sensor.snapshot()

        frame_usable = (
            face.frame is not None
            and face.reason not in {
                "too_dark",
                "too_bright",
                "image_quality_too_low",
                "camera_frame_unavailable",
            }
        )

        if frame_usable:
            attention = self.attention_analyzer.analyze(
                face.frame,
                data_quality=face.data_quality,
            )
            landmarks_available = attention.face_landmarks_detected
            if face.reason == "frame_ready":
                face = replace(
                    face,
                    face_detected=landmarks_available,
                    reason=(
                        "face_detected"
                        if landmarks_available
                        else "face_not_detected"
                    ),
                    face_box=attention.face_box,
                )
            ocular = self.ocular_tracker.update(
                attention.left_eye_opening_ratio,
                attention.right_eye_opening_ratio,
                face_detected=landmarks_available,
                data_quality=face.data_quality,
                left_eye_quality=attention.left_eye_quality,
                right_eye_quality=attention.right_eye_quality,
                head_pose_quality=attention.head_pose_quality,
                captured_at=captured_at,
            )
        else:
            attention = self._empty_attention_snapshot(reason="face_not_available")
            ocular = self.ocular_tracker.update(
                None,
                None,
                face_detected=False,
                data_quality=face.data_quality,
                left_eye_quality=0.0,
                right_eye_quality=0.0,
                head_pose_quality=0.0,
                captured_at=captured_at,
            )

        bundle = _VisualBundle(
            captured_at=captured_at,
            face=face,
            attention=attention,
            ocular=ocular,
        )
        with self._visual_lock:
            self._visual_bundle = bundle
            self._visual_error = None

    def _publish_visionless_bundle(self, captured_at: float) -> None:
        """Собрать снимок без камеры.

        Basic mode: остальные каналы (клавиатура, Windows idle, самооценка)
        продолжают работать, а визуальный канал честно сообщает, что данных
        нет. `face_detected=None` означает «неизвестно», а не «лица нет»:
        отсутствие камеры не должно выглядеть как отсутствие человека.
        """

        face = FacePresenceSnapshot(
            face_detected=None,
            data_quality=0.0,
            detection_confidence=0.0,
            brightness=0.0,
            sharpness=0.0,
            reason="camera_unavailable",
            frame=None,
            face_box=None,
        )
        attention = self._empty_attention_snapshot(reason="face_not_available")
        ocular = self.ocular_tracker.update(
            None,
            None,
            face_detected=False,
            data_quality=0.0,
            left_eye_quality=0.0,
            right_eye_quality=0.0,
            head_pose_quality=0.0,
            captured_at=captured_at,
        )
        bundle = _VisualBundle(
            captured_at=captured_at,
            face=face,
            attention=attention,
            ocular=ocular,
        )
        with self._visual_lock:
            self._visual_bundle = bundle
            self._visual_error = None

    def _screen_locked(self) -> bool:
        """Заблокирован ли экран сейчас.

        Неизвестное состояние трактуется как «не заблокирован»: иначе на
        неподдерживаемой системе монитор навсегда ушёл бы в AWAY.
        """

        return self.screen_lock_sensor.is_locked() is True

    def _observe_typing_baseline(
        self,
        *,
        metrics: TypingMetrics,
        state: UserState,
        captured_at: float,
        captured_datetime: datetime,
    ) -> TypingBaselineResult | None:
        service = self.typing_baseline_service
        if service is None or self._workday is None:
            return None
        minutes_since_start = max(0.0, (captured_at - self._workday_started_at) / 60.0)
        return service.observe(
            metrics,
            state=state,
            workday=self._workday,
            minutes_since_workday_start=minutes_since_start,
            captured_at=captured_datetime,
            now_monotonic=captured_at,
        )

    def _roll_workday_if_needed(self, captured_datetime: datetime, captured_at: float) -> None:
        current_day = captured_datetime.date()
        if self._workday != current_day:
            self._workday = current_day
            self._workday_started_at = captured_at

    @staticmethod
    def _empty_attention_snapshot(*, reason: str) -> VisualAttentionSnapshot:
        return VisualAttentionSnapshot(
            gaze_on_screen=None,
            face_landmarks_detected=False,
            horizontal_gaze_ratio=None,
            head_turn_ratio=None,
            head_vertical_ratio=None,
            reason=reason,
        )

    @staticmethod
    def _camera_status(snapshot: FacePresenceSnapshot) -> str:
        return CAMERA_STATUS_LABELS.get(snapshot.reason, "Состояние камеры неизвестно")

    @staticmethod
    def _attention_status(snapshot: VisualAttentionSnapshot) -> str:
        return ATTENTION_STATUS_LABELS.get(snapshot.reason, "Направление внимания неизвестно")

    @staticmethod
    def _ocular_status(metrics: OcularMetrics) -> str:
        return OCULAR_STATUS_LABELS.get(metrics.reason, "Состояние глаз неизвестно")

    def __enter__(self) -> SessionMonitor:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
