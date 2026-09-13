from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum
from time import monotonic


class EyeSignalState(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    CLOSED = "closed"
    UNCERTAIN = "uncertain"


class OcularEventType(str, Enum):
    BILATERAL_BLINK = "bilateral_blink"
    INCOMPLETE_BILATERAL_BLINK = "incomplete_bilateral_blink"
    LEFT_WINK = "left_wink"
    RIGHT_WINK = "right_wink"
    LONG_BILATERAL_CLOSURE = "long_bilateral_closure"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class OcularMetrics:
    """Обезличенные показатели глаз за скользящее окно."""

    observation_sec: float
    elapsed_window_sec: float
    data_ready: bool
    calibrated: bool

    left_eye_opening_ratio: float | None
    right_eye_opening_ratio: float | None
    normalized_left_eye_opening: float | None
    normalized_right_eye_opening: float | None
    left_open_eye_reference: float | None
    right_open_eye_reference: float | None
    left_eye_state: EyeSignalState
    right_eye_state: EyeSignalState

    eyes_closed: bool | None
    current_closure_duration_sec: float
    perclos: float | None

    blink_count: int
    incomplete_blink_count: int
    left_wink_count: int
    right_wink_count: int
    uncertain_event_count: int

    blink_rate_per_min: float | None
    mean_blink_duration_ms: float | None
    longest_blink_duration_ms: float | None

    long_closure_count: int
    long_closure_rate_per_min: float | None
    severe_closure_detected: bool

    processed_fps: float | None
    valid_sample_rate_hz: float | None
    signal_coverage: float
    processed_sample_count: int
    valid_sample_count: int

    quality: float
    reason: str

    @property
    def normalized_eye_opening(self) -> float | None:
        """Среднее только для диагностики; решения принимаются по двум глазам."""
        values = [
            value
            for value in (
                self.normalized_left_eye_opening,
                self.normalized_right_eye_opening,
            )
            if value is not None
        ]
        return statistics.fmean(values) if values else None

    @property
    def bilateral_blink_count(self) -> int:
        """Все подтверждённые двусторонние моргательные события.

        Сюда входят как события с уверенно зафиксированным глубоким
        закрытием, так и события с более поверхностным минимумом.
        Второй класс не доказывает неполное моргание: камера могла
        просто не захватить самый нижний момент закрытия век.
        """
        return self.blink_count + self.incomplete_blink_count

    @property
    def open_eye_reference(self) -> float | None:
        values = [
            value
            for value in (
                self.left_open_eye_reference,
                self.right_open_eye_reference,
            )
            if value is not None
        ]
        return statistics.fmean(values) if values else None

    @property
    def eye_opening_ratio(self) -> float | None:
        values = [
            value
            for value in (
                self.left_eye_opening_ratio,
                self.right_eye_opening_ratio,
            )
            if value is not None
        ]
        return statistics.fmean(values) if values else None

    def feature_dict(self) -> dict[str, float | None]:
        bilateral_events = self.bilateral_blink_count
        incomplete_ratio = (
            self.incomplete_blink_count / bilateral_events
            if bilateral_events > 0
            else None
        )
        return {
            "ocular_perclos": self.perclos,
            "ocular_blink_duration_ms": self.mean_blink_duration_ms,
            "ocular_blink_rate_per_min": self.blink_rate_per_min,
            "ocular_incomplete_blink_ratio": incomplete_ratio,
            "ocular_long_closure_rate_per_min": self.long_closure_rate_per_min,
            "ocular_eye_opening_ratio": self.normalized_eye_opening,
        }


@dataclass(frozen=True, slots=True)
class OcularEventRecord:
    """Публичная обезличенная запись одного события глаз.

    Содержит только время и нормализованную геометрию. Кадры, изображение
    лица и какие-либо идентифицирующие сведения здесь отсутствуют.
    """

    started_at: float
    ended_at: float
    duration_sec: float
    event_type: OcularEventType
    min_left_normalized: float | None
    min_right_normalized: float | None
    sync_offset_sec: float | None

    @property
    def center_at(self) -> float:
        return (self.started_at + self.ended_at) / 2.0


@dataclass(frozen=True, slots=True)
class _TimelineInterval:
    end_at: float
    duration_sec: float
    valid: bool
    bilateral_closed: bool


@dataclass(frozen=True, slots=True)
class _EyeEvent:
    started_at: float
    ended_at: float
    duration_sec: float
    event_type: OcularEventType
    min_left_normalized: float | None = None
    min_right_normalized: float | None = None
    sync_offset_sec: float | None = None


@dataclass(slots=True)
class _CandidateEvent:
    started_at: float
    last_valid_at: float
    start_left: float
    start_right: float
    min_left: float
    min_right: float
    min_left_at: float
    min_right_at: float
    first_bilateral_closed_at: float | None = None
    last_bilateral_closed_at: float | None = None
    reopen_confirmations: int = 0


class EyeActivityTracker:
    """
    Высокочастотный двусторонний анализ глаз.

    Подмигивание не считается морганием. PERCLOS увеличивается только
    тогда, когда оба глаза надёжно закрыты. При недостаточном качестве
    система возвращает неопределённость, а не угадывает состояние.
    """

    def __init__(
        self,
        *,
        window_sec: float = 60.0,
        calibration_min_samples: int = 60,
        calibration_min_sec: float = 3.0,
        min_observation_sec: float = 10.0,
        min_eye_quality: float = 0.55,
        min_head_pose_quality: float = 0.35,
        partial_ratio_threshold: float = 0.80,
        closed_ratio_threshold: float = 0.55,
        full_blink_ratio_threshold: float = 0.75,
        open_ratio_threshold: float = 0.82,
        min_blink_sec: float = 0.06,
        max_blink_sec: float = 0.70,
        long_closure_sec: float = 1.00,
        severe_closure_sec: float = 2.00,
        bilateral_sync_tolerance_sec: float = 0.12,
        max_sample_gap_sec: float = 0.20,
        candidate_gap_tolerance_sec: float = 0.12,
        reopen_confirmation_samples: int = 2,
        min_reliable_fps: float = 15.0,
        min_signal_coverage: float = 0.70,
        reference_max_samples: int = 1800,
        smoothing_alpha: float = 0.75,
        update_reference_after_calibration: bool = True,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec должен быть больше нуля.")
        if calibration_min_samples < 5:
            raise ValueError("calibration_min_samples должен быть не меньше 5.")
        if calibration_min_sec <= 0 or min_observation_sec <= 0:
            raise ValueError("Временные параметры должны быть больше нуля.")
        if not 0.0 <= min_eye_quality <= 1.0:
            raise ValueError("min_eye_quality должен быть в диапазоне 0..1.")
        if not 0.0 <= min_head_pose_quality <= 1.0:
            raise ValueError("min_head_pose_quality должен быть в диапазоне 0..1.")
        if not 0.0 < closed_ratio_threshold < partial_ratio_threshold < 1.0:
            raise ValueError("Пороги закрытия глаз заданы неверно.")
        if not (
            closed_ratio_threshold
            <= full_blink_ratio_threshold
            < partial_ratio_threshold
        ):
            raise ValueError(
                "full_blink_ratio_threshold должен находиться между "
                "closed_ratio_threshold и partial_ratio_threshold."
            )
        if not partial_ratio_threshold <= open_ratio_threshold <= 1.2:
            raise ValueError("open_ratio_threshold задан неверно.")
        if not 0.0 < min_blink_sec < max_blink_sec < long_closure_sec:
            raise ValueError("Длительности моргания заданы неверно.")
        if severe_closure_sec <= long_closure_sec:
            raise ValueError("severe_closure_sec должен быть больше long_closure_sec.")
        if bilateral_sync_tolerance_sec <= 0:
            raise ValueError("bilateral_sync_tolerance_sec должен быть больше нуля.")
        if max_sample_gap_sec <= 0 or candidate_gap_tolerance_sec <= 0:
            raise ValueError("Допустимые разрывы должны быть больше нуля.")
        if reopen_confirmation_samples < 1:
            raise ValueError("Нужно хотя бы одно подтверждение открытия.")
        if min_reliable_fps <= 0:
            raise ValueError("min_reliable_fps должен быть больше нуля.")
        if not 0.0 <= min_signal_coverage <= 1.0:
            raise ValueError("min_signal_coverage должен быть в диапазоне 0..1.")
        if reference_max_samples < calibration_min_samples:
            raise ValueError("reference_max_samples слишком мал.")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha должен быть в диапазоне (0, 1].")

        self.window_sec = float(window_sec)
        self.calibration_min_samples = int(calibration_min_samples)
        self.calibration_min_sec = float(calibration_min_sec)
        self.min_observation_sec = float(min_observation_sec)
        self.min_eye_quality = float(min_eye_quality)
        self.min_head_pose_quality = float(min_head_pose_quality)
        self.partial_ratio_threshold = float(partial_ratio_threshold)
        self.closed_ratio_threshold = float(closed_ratio_threshold)
        self.full_blink_ratio_threshold = float(full_blink_ratio_threshold)
        self.open_ratio_threshold = float(open_ratio_threshold)
        self.min_blink_sec = float(min_blink_sec)
        self.max_blink_sec = float(max_blink_sec)
        self.long_closure_sec = float(long_closure_sec)
        self.severe_closure_sec = float(severe_closure_sec)
        self.bilateral_sync_tolerance_sec = float(bilateral_sync_tolerance_sec)
        self.max_sample_gap_sec = float(max_sample_gap_sec)
        self.candidate_gap_tolerance_sec = float(candidate_gap_tolerance_sec)
        self.reopen_confirmation_samples = int(reopen_confirmation_samples)
        self.min_reliable_fps = float(min_reliable_fps)
        self.min_signal_coverage = float(min_signal_coverage)
        self.smoothing_alpha = float(smoothing_alpha)
        self.update_reference_after_calibration = bool(
            update_reference_after_calibration
        )

        self._left_reference_samples: deque[tuple[float, float]] = deque(
            maxlen=reference_max_samples
        )
        self._right_reference_samples: deque[tuple[float, float]] = deque(
            maxlen=reference_max_samples
        )
        self._timeline: deque[_TimelineInterval] = deque()
        self._events: deque[_EyeEvent] = deque()
        self._processed_timestamps: deque[float] = deque()
        self._valid_timestamps: deque[float] = deque()

        self._previous_sample_at: float | None = None
        self._previous_sample_valid = False
        self._previous_bilateral_closed = False
        self._bilateral_closure_started_at: float | None = None
        self._candidate: _CandidateEvent | None = None

        self._smoothed_left: float | None = None
        self._smoothed_right: float | None = None
        self._last_left_ratio: float | None = None
        self._last_right_ratio: float | None = None
        self._last_normalized_left: float | None = None
        self._last_normalized_right: float | None = None
        self._last_left_state = EyeSignalState.UNCERTAIN
        self._last_right_state = EyeSignalState.UNCERTAIN
        self._last_channel_quality = 0.0
        self._last_reason = "not_started"

    @property
    def calibrated(self) -> bool:
        return self._reference_ready(self._left_reference_samples) and self._reference_ready(
            self._right_reference_samples
        )

    @property
    def left_open_eye_reference(self) -> float | None:
        return self._reference_value(self._left_reference_samples)

    @property
    def right_open_eye_reference(self) -> float | None:
        return self._reference_value(self._right_reference_samples)

    def update(
        self,
        left_eye_opening_ratio: float | None,
        right_eye_opening_ratio: float | None,
        *,
        face_detected: bool | None,
        data_quality: float,
        left_eye_quality: float,
        right_eye_quality: float,
        head_pose_quality: float,
        captured_at: float | None = None,
    ) -> OcularMetrics:
        now = monotonic() if captured_at is None else float(captured_at)
        self._validate_update_values(
            now=now,
            data_quality=data_quality,
            left_eye_quality=left_eye_quality,
            right_eye_quality=right_eye_quality,
            head_pose_quality=head_pose_quality,
        )
        self._record_timeline_before_current(now)
        self._processed_timestamps.append(now)

        invalid_reason = self._invalid_reason(
            left_eye_opening_ratio=left_eye_opening_ratio,
            right_eye_opening_ratio=right_eye_opening_ratio,
            face_detected=face_detected,
            data_quality=data_quality,
            left_eye_quality=left_eye_quality,
            right_eye_quality=right_eye_quality,
            head_pose_quality=head_pose_quality,
        )

        if invalid_reason is not None:
            self._handle_invalid_sample(now, invalid_reason)
            self._prune(now)
            return self.snapshot(captured_at=now)

        assert left_eye_opening_ratio is not None
        assert right_eye_opening_ratio is not None
        left_raw = float(left_eye_opening_ratio)
        right_raw = float(right_eye_opening_ratio)
        self._last_left_ratio = left_raw
        self._last_right_ratio = right_raw
        self._last_channel_quality = min(
            float(data_quality),
            float(left_eye_quality),
            float(right_eye_quality),
            float(head_pose_quality),
        )

        left_reference_before = self.left_open_eye_reference
        right_reference_before = self.right_open_eye_reference
        self._maybe_update_references(
            captured_at=now,
            left_ratio=left_raw,
            right_ratio=right_raw,
            left_reference=left_reference_before,
            right_reference=right_reference_before,
        )
        left_reference = self.left_open_eye_reference
        right_reference = self.right_open_eye_reference

        if left_reference is None or right_reference is None:
            self._mark_current_sample(now, valid=False, bilateral_closed=False)
            self._last_normalized_left = None
            self._last_normalized_right = None
            self._last_left_state = EyeSignalState.UNCERTAIN
            self._last_right_state = EyeSignalState.UNCERTAIN
            self._last_reason = "calibrating_open_eyes"
            self._prune(now)
            return self.snapshot(captured_at=now)

        left_normalized_raw = left_raw / left_reference
        right_normalized_raw = right_raw / right_reference
        left_normalized = self._smooth_left(left_normalized_raw)
        right_normalized = self._smooth_right(right_normalized_raw)
        self._last_normalized_left = left_normalized
        self._last_normalized_right = right_normalized

        left_state = self._state_for(left_normalized)
        right_state = self._state_for(right_normalized)
        self._last_left_state = left_state
        self._last_right_state = right_state
        bilateral_closed = (
            left_state is EyeSignalState.CLOSED
            and right_state is EyeSignalState.CLOSED
        )

        self._valid_timestamps.append(now)
        self._update_bilateral_closure(now, bilateral_closed)
        self._mark_current_sample(now, valid=True, bilateral_closed=bilateral_closed)
        self._update_candidate(
            now=now,
            left=left_normalized,
            right=right_normalized,
            bilateral_closed=bilateral_closed,
        )

        if bilateral_closed:
            self._last_reason = "bilateral_closure"
        elif self._candidate is not None:
            self._last_reason = "blink_candidate"
        else:
            self._last_reason = "eyes_open"

        self._prune(now)
        return self.snapshot(captured_at=now)

    def snapshot(self, *, captured_at: float | None = None) -> OcularMetrics:
        now = monotonic() if captured_at is None else float(captured_at)
        if not math.isfinite(now):
            raise ValueError("captured_at должен быть конечным числом.")
        self._prune(now)

        valid_observation_sec = sum(
            item.duration_sec for item in self._timeline if item.valid
        )
        total_observation_sec = sum(item.duration_sec for item in self._timeline)
        closed_sec = sum(
            item.duration_sec
            for item in self._timeline
            if item.valid and item.bilateral_closed
        )
        perclos = (
            closed_sec / valid_observation_sec
            if valid_observation_sec > 0
            else None
        )

        blink_events = [
            event
            for event in self._events
            if event.event_type is OcularEventType.BILATERAL_BLINK
        ]
        incomplete_events = [
            event
            for event in self._events
            if event.event_type is OcularEventType.INCOMPLETE_BILATERAL_BLINK
        ]
        left_winks = [
            event for event in self._events if event.event_type is OcularEventType.LEFT_WINK
        ]
        right_winks = [
            event for event in self._events if event.event_type is OcularEventType.RIGHT_WINK
        ]
        uncertain_events = [
            event for event in self._events if event.event_type is OcularEventType.UNCERTAIN
        ]
        long_events = [
            event
            for event in self._events
            if event.event_type is OcularEventType.LONG_BILATERAL_CLOSURE
        ]

        bilateral_blink_events = blink_events + incomplete_events
        blink_durations = [event.duration_sec for event in bilateral_blink_events]
        current_closure_duration = self._current_closure_duration(now)
        current_long = current_closure_duration >= self.long_closure_sec
        severe = current_closure_duration >= self.severe_closure_sec
        long_closure_count = len(long_events) + int(current_long)

        blink_rate = (
            len(bilateral_blink_events) * 60.0 / valid_observation_sec
            if valid_observation_sec > 0
            else None
        )
        long_rate = (
            long_closure_count * 60.0 / valid_observation_sec
            if valid_observation_sec > 0
            else None
        )
        mean_duration_ms = (
            statistics.fmean(blink_durations) * 1000.0
            if blink_durations
            else None
        )
        longest_duration_ms = (
            max(blink_durations) * 1000.0 if blink_durations else None
        )

        processed_fps = self._rate(self._processed_timestamps)
        valid_fps = self._rate(self._valid_timestamps)
        coverage = (
            valid_observation_sec / total_observation_sec
            if total_observation_sec > 0
            else 0.0
        )
        fps_quality = (
            min(1.0, processed_fps / self.min_reliable_fps)
            if processed_fps is not None
            else 0.0
        )
        quality = min(self._last_channel_quality, coverage, fps_quality)

        data_ready = (
            self.calibrated
            and valid_observation_sec >= self.min_observation_sec
            and coverage >= self.min_signal_coverage
            and processed_fps is not None
            and processed_fps >= self.min_reliable_fps
        )

        reason = self._last_reason
        if self.calibrated and processed_fps is not None and processed_fps < self.min_reliable_fps:
            reason = "processing_fps_too_low"
        elif self.calibrated and total_observation_sec > 0 and coverage < self.min_signal_coverage:
            reason = "eye_signal_coverage_too_low"

        eyes_closed: bool | None
        if (
            self._last_left_state is EyeSignalState.UNCERTAIN
            or self._last_right_state is EyeSignalState.UNCERTAIN
        ):
            eyes_closed = None
        else:
            eyes_closed = (
                self._last_left_state is EyeSignalState.CLOSED
                and self._last_right_state is EyeSignalState.CLOSED
            )

        return OcularMetrics(
            observation_sec=valid_observation_sec,
            elapsed_window_sec=total_observation_sec,
            data_ready=data_ready,
            calibrated=self.calibrated,
            left_eye_opening_ratio=self._last_left_ratio,
            right_eye_opening_ratio=self._last_right_ratio,
            normalized_left_eye_opening=self._last_normalized_left,
            normalized_right_eye_opening=self._last_normalized_right,
            left_open_eye_reference=self.left_open_eye_reference,
            right_open_eye_reference=self.right_open_eye_reference,
            left_eye_state=self._last_left_state,
            right_eye_state=self._last_right_state,
            eyes_closed=eyes_closed,
            current_closure_duration_sec=current_closure_duration,
            perclos=perclos,
            blink_count=len(blink_events),
            incomplete_blink_count=len(incomplete_events),
            left_wink_count=len(left_winks),
            right_wink_count=len(right_winks),
            uncertain_event_count=len(uncertain_events),
            blink_rate_per_min=blink_rate,
            mean_blink_duration_ms=mean_duration_ms,
            longest_blink_duration_ms=longest_duration_ms,
            long_closure_count=long_closure_count,
            long_closure_rate_per_min=long_rate,
            severe_closure_detected=severe,
            processed_fps=processed_fps,
            valid_sample_rate_hz=valid_fps,
            signal_coverage=coverage,
            processed_sample_count=len(self._processed_timestamps),
            valid_sample_count=len(self._valid_timestamps),
            quality=quality,
            reason=reason,
        )

    def event_history(self) -> tuple[OcularEventRecord, ...]:
        """Возвращает события текущего скользящего окна для валидации.

        Метод не раскрывает кадры и не меняет состояние трекера.
        """
        return tuple(
            OcularEventRecord(
                started_at=event.started_at,
                ended_at=event.ended_at,
                duration_sec=event.duration_sec,
                event_type=event.event_type,
                min_left_normalized=event.min_left_normalized,
                min_right_normalized=event.min_right_normalized,
                sync_offset_sec=event.sync_offset_sec,
            )
            for event in self._events
        )

    def reset(self, *, keep_calibration: bool = True) -> None:
        self._timeline.clear()
        self._events.clear()
        self._processed_timestamps.clear()
        self._valid_timestamps.clear()
        self._previous_sample_at = None
        self._previous_sample_valid = False
        self._previous_bilateral_closed = False
        self._bilateral_closure_started_at = None
        self._candidate = None
        self._smoothed_left = None
        self._smoothed_right = None
        self._last_left_ratio = None
        self._last_right_ratio = None
        self._last_normalized_left = None
        self._last_normalized_right = None
        self._last_left_state = EyeSignalState.UNCERTAIN
        self._last_right_state = EyeSignalState.UNCERTAIN
        self._last_channel_quality = 0.0
        self._last_reason = "reset"
        if not keep_calibration:
            self._left_reference_samples.clear()
            self._right_reference_samples.clear()

    def _reference_ready(self, samples: deque[tuple[float, float]]) -> bool:
        if len(samples) < self.calibration_min_samples:
            return False
        return samples[-1][0] - samples[0][0] >= self.calibration_min_sec

    def _reference_value(self, samples: deque[tuple[float, float]]) -> float | None:
        if not self._reference_ready(samples):
            return None

        # Обычное открытое положение должно соответствовать примерно 1.0.
        # Раньше использовалась медиана верхних 40% значений, то есть
        # фактически около 80-го процентиля. Из-за этого эталон получался
        # завышенным, а нормально открытые глаза отображались как 0.7–0.8
        # и ошибочно попадали в состояние PARTIAL.
        #
        # Обычная медиана устойчива к нескольким кадрам моргания: закрытые
        # глаза занимают малую долю калибровочного интервала и не способны
        # сдвинуть медиану, пока человек большую часть времени смотрит
        # с открытыми глазами.
        values = [value for _, value in samples]
        return float(statistics.median(values)) if values else None

    def _maybe_update_references(
        self,
        *,
        captured_at: float,
        left_ratio: float,
        right_ratio: float,
        left_reference: float | None,
        right_reference: float | None,
    ) -> None:
        if left_reference is None or right_reference is None:
            self._left_reference_samples.append((captured_at, left_ratio))
            self._right_reference_samples.append((captured_at, right_ratio))
            return
        if not self.update_reference_after_calibration:
            return
        left_normalized = left_ratio / left_reference
        right_normalized = right_ratio / right_reference
        if (
            left_normalized >= self.open_ratio_threshold
            and right_normalized >= self.open_ratio_threshold
            and self._candidate is None
        ):
            self._left_reference_samples.append((captured_at, left_ratio))
            self._right_reference_samples.append((captured_at, right_ratio))

    def _smooth_left(self, value: float) -> float:
        self._smoothed_left = self._smooth(self._smoothed_left, value)
        return self._smoothed_left

    def _smooth_right(self, value: float) -> float:
        self._smoothed_right = self._smooth(self._smoothed_right, value)
        return self._smoothed_right

    def _smooth(self, previous: float | None, value: float) -> float:
        if previous is None:
            return value
        return self.smoothing_alpha * value + (1.0 - self.smoothing_alpha) * previous

    def _state_for(self, normalized: float) -> EyeSignalState:
        if normalized <= self.closed_ratio_threshold:
            return EyeSignalState.CLOSED
        if normalized >= self.open_ratio_threshold:
            return EyeSignalState.OPEN
        return EyeSignalState.PARTIAL

    def _update_candidate(
        self,
        *,
        now: float,
        left: float,
        right: float,
        bilateral_closed: bool,
    ) -> None:
        candidate = self._candidate
        both_open = (
            left >= self.open_ratio_threshold and right >= self.open_ratio_threshold
        )
        any_dip = (
            left <= self.partial_ratio_threshold or right <= self.partial_ratio_threshold
        )

        if candidate is None:
            if any_dip:
                self._candidate = _CandidateEvent(
                    started_at=now,
                    last_valid_at=now,
                    start_left=left,
                    start_right=right,
                    min_left=left,
                    min_right=right,
                    min_left_at=now,
                    min_right_at=now,
                    first_bilateral_closed_at=now if bilateral_closed else None,
                    last_bilateral_closed_at=now if bilateral_closed else None,
                )
            return

        candidate.last_valid_at = now
        if left < candidate.min_left:
            candidate.min_left = left
            candidate.min_left_at = now
        if right < candidate.min_right:
            candidate.min_right = right
            candidate.min_right_at = now
        if bilateral_closed:
            if candidate.first_bilateral_closed_at is None:
                candidate.first_bilateral_closed_at = now
            candidate.last_bilateral_closed_at = now

        if both_open:
            candidate.reopen_confirmations += 1
            if candidate.reopen_confirmations >= self.reopen_confirmation_samples:
                self._finalize_candidate(now)
            return

        candidate.reopen_confirmations = 0
        if now - candidate.started_at > max(self.long_closure_sec * 4.0, 4.0):
            self._record_event(now, now - candidate.started_at, OcularEventType.UNCERTAIN)
            self._candidate = None

    def _finalize_candidate(self, ended_at: float) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        duration = max(0.0, ended_at - candidate.started_at)
        sync = abs(candidate.min_left_at - candidate.min_right_at)
        both_dipped = (
            candidate.min_left <= self.partial_ratio_threshold
            and candidate.min_right <= self.partial_ratio_threshold
        )
        # Для PERCLOS и длительного закрытия сохраняется строгий порог
        # closed_ratio_threshold. Для обычного быстрого моргания используется
        # более мягкий full_blink_ratio_threshold: камера нередко не успевает
        # захватить кадр с полностью сомкнутыми веками, хотя временная форма
        # и синхронность обоих глаз подтверждают обычное моргание.
        both_full_blink = (
            candidate.min_left <= self.full_blink_ratio_threshold
            and candidate.min_right <= self.full_blink_ratio_threshold
            and sync <= self.bilateral_sync_tolerance_sec
        )
        both_strictly_closed = (
            candidate.min_left <= self.closed_ratio_threshold
            and candidate.min_right <= self.closed_ratio_threshold
            and sync <= self.bilateral_sync_tolerance_sec
        )
        left_wink = (
            candidate.min_left <= self.full_blink_ratio_threshold
            and candidate.min_right > self.partial_ratio_threshold
        )
        right_wink = (
            candidate.min_right <= self.full_blink_ratio_threshold
            and candidate.min_left > self.partial_ratio_threshold
        )

        bilateral_closed_duration = 0.0
        if (
            candidate.first_bilateral_closed_at is not None
            and candidate.last_bilateral_closed_at is not None
        ):
            bilateral_closed_duration = max(
                0.0,
                candidate.last_bilateral_closed_at
                - candidate.first_bilateral_closed_at,
            )

        if both_strictly_closed and (
            duration >= self.long_closure_sec
            or bilateral_closed_duration >= self.long_closure_sec
        ):
            event_type = OcularEventType.LONG_BILATERAL_CLOSURE
        elif (
            both_full_blink
            and self.min_blink_sec <= duration <= self.max_blink_sec
        ):
            event_type = OcularEventType.BILATERAL_BLINK
        elif (
            both_dipped
            and sync <= self.bilateral_sync_tolerance_sec
            and self.min_blink_sec <= duration <= self.max_blink_sec
        ):
            event_type = OcularEventType.INCOMPLETE_BILATERAL_BLINK
        elif left_wink and self.min_blink_sec <= duration <= self.max_blink_sec:
            event_type = OcularEventType.LEFT_WINK
        elif right_wink and self.min_blink_sec <= duration <= self.max_blink_sec:
            event_type = OcularEventType.RIGHT_WINK
        else:
            event_type = OcularEventType.UNCERTAIN

        self._record_event(
            ended_at,
            duration,
            event_type,
            min_left_normalized=candidate.min_left,
            min_right_normalized=candidate.min_right,
            sync_offset_sec=sync,
        )
        self._candidate = None

    def _record_event(
        self,
        ended_at: float,
        duration_sec: float,
        event_type: OcularEventType,
        *,
        min_left_normalized: float | None = None,
        min_right_normalized: float | None = None,
        sync_offset_sec: float | None = None,
    ) -> None:
        duration = max(0.0, float(duration_sec))
        self._events.append(
            _EyeEvent(
                started_at=float(ended_at) - duration,
                ended_at=float(ended_at),
                duration_sec=duration,
                event_type=event_type,
                min_left_normalized=min_left_normalized,
                min_right_normalized=min_right_normalized,
                sync_offset_sec=sync_offset_sec,
            )
        )

    def _update_bilateral_closure(self, now: float, bilateral_closed: bool) -> None:
        if bilateral_closed and not self._previous_bilateral_closed:
            self._bilateral_closure_started_at = now
        elif not bilateral_closed and self._previous_bilateral_closed:
            self._bilateral_closure_started_at = None

    def _current_closure_duration(self, now: float) -> float:
        if not self._previous_bilateral_closed:
            return 0.0
        if self._bilateral_closure_started_at is None:
            return 0.0
        return max(0.0, now - self._bilateral_closure_started_at)

    def _record_timeline_before_current(self, now: float) -> None:
        previous_at = self._previous_sample_at
        if previous_at is None:
            return
        delta = now - previous_at
        if delta <= 0 or delta > self.max_sample_gap_sec:
            return
        self._timeline.append(
            _TimelineInterval(
                end_at=now,
                duration_sec=delta,
                valid=self._previous_sample_valid,
                bilateral_closed=(
                    self._previous_sample_valid and self._previous_bilateral_closed
                ),
            )
        )

    def _mark_current_sample(self, now: float, *, valid: bool, bilateral_closed: bool) -> None:
        self._previous_sample_at = now
        self._previous_sample_valid = valid
        self._previous_bilateral_closed = bilateral_closed if valid else False

    def _handle_invalid_sample(self, now: float, reason: str) -> None:
        candidate = self._candidate
        if candidate is not None and now - candidate.last_valid_at > self.candidate_gap_tolerance_sec:
            self._record_event(now, now - candidate.started_at, OcularEventType.UNCERTAIN)
            self._candidate = None
        self._mark_current_sample(now, valid=False, bilateral_closed=False)
        self._bilateral_closure_started_at = None
        self._smoothed_left = None
        self._smoothed_right = None
        self._last_left_ratio = None
        self._last_right_ratio = None
        self._last_normalized_left = None
        self._last_normalized_right = None
        self._last_left_state = EyeSignalState.UNCERTAIN
        self._last_right_state = EyeSignalState.UNCERTAIN
        self._last_channel_quality = 0.0
        self._last_reason = reason

    def _invalid_reason(
        self,
        *,
        left_eye_opening_ratio: float | None,
        right_eye_opening_ratio: float | None,
        face_detected: bool | None,
        data_quality: float,
        left_eye_quality: float,
        right_eye_quality: float,
        head_pose_quality: float,
    ) -> str | None:
        if face_detected is not True:
            return "face_not_available"
        if data_quality < self.min_eye_quality:
            return "eye_data_quality_too_low"
        if head_pose_quality < self.min_head_pose_quality:
            return "head_pose_too_extreme"
        if left_eye_quality < self.min_eye_quality:
            return "left_eye_quality_too_low"
        if right_eye_quality < self.min_eye_quality:
            return "right_eye_quality_too_low"
        if left_eye_opening_ratio is None or right_eye_opening_ratio is None:
            return "eye_geometry_unavailable"
        for value in (left_eye_opening_ratio, right_eye_opening_ratio):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                return "eye_geometry_unreliable"
        return None

    def _validate_update_values(
        self,
        *,
        now: float,
        data_quality: float,
        left_eye_quality: float,
        right_eye_quality: float,
        head_pose_quality: float,
    ) -> None:
        if not math.isfinite(now):
            raise ValueError("captured_at должен быть конечным числом.")
        for name, value in (
            ("data_quality", data_quality),
            ("left_eye_quality", left_eye_quality),
            ("right_eye_quality", right_eye_quality),
            ("head_pose_quality", head_pose_quality),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} должен быть в диапазоне 0..1.")

    @staticmethod
    def _rate(timestamps: deque[float]) -> float | None:
        if len(timestamps) < 2:
            return None
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            return None
        return (len(timestamps) - 1) / elapsed

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._timeline and self._timeline[0].end_at < cutoff:
            self._timeline.popleft()
        while self._events and self._events[0].ended_at < cutoff:
            self._events.popleft()
        while self._processed_timestamps and self._processed_timestamps[0] < cutoff:
            self._processed_timestamps.popleft()
        while self._valid_timestamps and self._valid_timestamps[0] < cutoff:
            self._valid_timestamps.popleft()
