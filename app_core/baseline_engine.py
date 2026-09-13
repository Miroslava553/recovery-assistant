from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Mapping


class WorkContext(StrEnum):
    """Контекст, внутри которого строится отдельная личная норма."""

    TYPING = "typing"
    READING = "reading"
    MIXED = "mixed"


class FeatureChannel(StrEnum):
    """Источник обезличенного признака."""

    TYPING = "typing"
    MOUSE = "mouse"
    OCULAR = "ocular"
    POSTURE = "posture"
    GENERIC = "generic"


# Это не нормы человека и не пороги усталости.
# Это только минимальный численный масштаб, который не позволяет
# robust z-score становиться огромным при почти нулевом MAD.
DEFAULT_MIN_SCALES: dict[str, float] = {
    # Клавиатура
    "typing_keys_per_minute": 10.0,
    "typing_mean_interval_sec": 0.03,
    "typing_median_interval_sec": 0.03,
    "typing_rhythm_cv": 0.03,
    "typing_correction_ratio": 0.02,
    "typing_long_pause_rate_per_min": 0.20,
    "typing_burst_rate_per_min": 0.20,
    # Мышь
    "mouse_speed_px_sec": 25.0,
    "mouse_jerk": 0.05,
    "mouse_micro_pause_ratio": 0.02,
    "mouse_click_interval_cv": 0.03,
    "mouse_path_entropy": 0.05,
    # Глаза
    "ocular_perclos": 0.015,
    "ocular_blink_duration_ms": 15.0,
    "ocular_blink_rate_per_min": 1.0,
    "ocular_incomplete_blink_ratio": 0.02,
    "ocular_long_closure_rate_per_min": 0.10,
    "ocular_eye_opening_ratio": 0.02,
    # Голова и поза
    "posture_head_slump_ratio": 0.02,
    "posture_neck_flexion_ratio": 0.02,
    "posture_head_nod_rate_per_min": 0.10,
}


@dataclass(frozen=True, slots=True)
class BaselineObservation:
    """
    Одно обезличенное окно, которое может попасть в личную норму.

    features не содержит текст, названия клавиш, кадры или координаты
    отдельных действий. Только агрегированные числовые показатели.
    """

    workday: date
    minutes_since_workday_start: float
    context: WorkContext
    features: Mapping[str, float | int | None]

    user_present: bool = True
    returning: bool = False
    severe_ocular_event: bool = False

    self_report_fatigue_0_10: float | None = None
    self_report_sleepiness_kss_1_9: float | None = None

    channel_quality: Mapping[FeatureChannel | str, float | None] = field(
        default_factory=dict
    )
    liveness_score: float | None = None

    def __post_init__(self) -> None:
        if self.minutes_since_workday_start < 0:
            raise ValueError(
                "minutes_since_workday_start не может быть отрицательным."
            )

        if not isinstance(self.context, WorkContext):
            raise TypeError("context должен быть значением WorkContext.")

        if self.self_report_fatigue_0_10 is not None:
            if not 0.0 <= self.self_report_fatigue_0_10 <= 10.0:
                raise ValueError(
                    "self_report_fatigue_0_10 должен быть в диапазоне 0..10."
                )

        if self.self_report_sleepiness_kss_1_9 is not None:
            if not 1.0 <= self.self_report_sleepiness_kss_1_9 <= 9.0:
                raise ValueError(
                    "self_report_sleepiness_kss_1_9 должен быть в диапазоне 1..9."
                )

        if self.liveness_score is not None:
            if not 0.0 <= self.liveness_score <= 1.0:
                raise ValueError("liveness_score должен быть в диапазоне 0..1.")

        for raw_channel, quality in self.channel_quality.items():
            try:
                FeatureChannel(str(raw_channel))
            except ValueError as error:
                raise ValueError(
                    f"Неизвестный канал качества: {raw_channel!r}."
                ) from error

            if quality is None:
                continue
            if not 0.0 <= float(quality) <= 1.0:
                raise ValueError(
                    f"Качество канала {raw_channel!r} должно быть в диапазоне 0..1."
                )


@dataclass(frozen=True, slots=True)
class BaselineSample:
    workday: date
    value: float


@dataclass(frozen=True, slots=True)
class BaselineStats:
    context: WorkContext
    feature_name: str
    median: float
    mad: float
    q25: float
    q75: float
    robust_scale: float
    sample_count: int
    day_count: int
    ready: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    available: bool
    ready: bool
    feature_name: str
    context: WorkContext
    actual_value: float | None
    baseline_median: float | None
    robust_z: float | None
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class BaselineUpdateResult:
    accepted: bool
    initial_calibration_complete: bool
    calibration_day_count: int
    accepted_features: tuple[str, ...]
    rejected_features: Mapping[str, str]
    reason: str


class PersonalBaselineEngine:
    """
    Строит личную норму отдельно для каждого контекста и признака.

    Важные правила:
    - общей нормы скорости печати для всех людей нет;
    - первые 5 рабочих дней являются периодом адаптации;
    - для первичной нормы используются только первые 2 часа рабочего дня;
    - уставшие, невалидные и сомнительные окна в baseline не попадают;
    - после калибровки обновление нормы возможно только явно, через
      allow_refresh=True;
    - отсутствующий канал не заменяется нулём.
    """

    def __init__(
        self,
        *,
        calibration_days: int = 5,
        baseline_window_minutes: float = 120.0,
        min_windows_per_feature: int = 30,
        min_samples_for_stats: int = 5,
        max_windows_per_feature: int = 500,
        min_channel_quality: float = 0.80,
        min_liveness_score: float = 0.80,
        max_baseline_fatigue_0_10: float = 3.0,
        max_baseline_sleepiness_kss: float = 5.0,
        min_scales: Mapping[str, float] | None = None,
    ) -> None:
        if calibration_days < 1:
            raise ValueError("calibration_days должен быть не меньше 1.")
        if baseline_window_minutes <= 0:
            raise ValueError("baseline_window_minutes должен быть больше нуля.")
        if min_windows_per_feature < 1:
            raise ValueError("min_windows_per_feature должен быть не меньше 1.")
        if min_samples_for_stats < 2:
            raise ValueError("min_samples_for_stats должен быть не меньше 2.")
        if min_samples_for_stats > min_windows_per_feature:
            raise ValueError(
                "min_samples_for_stats не должен превышать min_windows_per_feature."
            )
        if max_windows_per_feature < min_windows_per_feature:
            raise ValueError(
                "max_windows_per_feature не должен быть меньше "
                "min_windows_per_feature."
            )
        if not 0.0 <= min_channel_quality <= 1.0:
            raise ValueError("min_channel_quality должен быть в диапазоне 0..1.")
        if not 0.0 <= min_liveness_score <= 1.0:
            raise ValueError("min_liveness_score должен быть в диапазоне 0..1.")

        self.calibration_days = int(calibration_days)
        self.baseline_window_minutes = float(baseline_window_minutes)
        self.min_windows_per_feature = int(min_windows_per_feature)
        self.min_samples_for_stats = int(min_samples_for_stats)
        self.max_windows_per_feature = int(max_windows_per_feature)
        self.min_channel_quality = float(min_channel_quality)
        self.min_liveness_score = float(min_liveness_score)
        self.max_baseline_fatigue_0_10 = float(max_baseline_fatigue_0_10)
        self.max_baseline_sleepiness_kss = float(max_baseline_sleepiness_kss)

        self._min_scales = dict(DEFAULT_MIN_SCALES)
        if min_scales is not None:
            for feature_name, value in min_scales.items():
                clean_name = self._clean_feature_name(feature_name)
                numeric_value = float(value)
                if not math.isfinite(numeric_value) or numeric_value <= 0:
                    raise ValueError(
                        f"Минимальный масштаб {clean_name!r} должен быть "
                        "конечным положительным числом."
                    )
                self._min_scales[clean_name] = numeric_value

        self._samples: dict[
            WorkContext,
            dict[str, deque[BaselineSample]],
        ] = defaultdict(dict)
        self._accepted_workdays: set[date] = set()

    @property
    def calibration_day_count(self) -> int:
        return len(self._accepted_workdays)

    @property
    def initial_calibration_complete(self) -> bool:
        return self.calibration_day_count >= self.calibration_days

    @property
    def calibration_progress(self) -> float:
        return min(1.0, self.calibration_day_count / self.calibration_days)

    def observe(
        self,
        observation: BaselineObservation,
        *,
        allow_refresh: bool = False,
    ) -> BaselineUpdateResult:
        general_rejection = self._general_rejection_reason(
            observation,
            allow_refresh=allow_refresh,
        )
        if general_rejection is not None:
            return BaselineUpdateResult(
                accepted=False,
                initial_calibration_complete=self.initial_calibration_complete,
                calibration_day_count=self.calibration_day_count,
                accepted_features=(),
                rejected_features={},
                reason=general_rejection,
            )

        accepted_features: list[str] = []
        rejected_features: dict[str, str] = {}

        for raw_name, raw_value in observation.features.items():
            try:
                feature_name = self._clean_feature_name(raw_name)
            except (TypeError, ValueError) as error:
                rejected_features[str(raw_name)] = str(error)
                continue

            value, value_error = self._validated_feature_value(raw_value)
            if value_error is not None:
                rejected_features[feature_name] = value_error
                continue

            assert value is not None
            channel = self.feature_channel(feature_name)
            channel_error = self._channel_rejection_reason(
                observation,
                channel,
            )
            if channel_error is not None:
                rejected_features[feature_name] = channel_error
                continue

            feature_samples = self._samples[observation.context].get(feature_name)
            if feature_samples is None:
                feature_samples = deque(maxlen=self.max_windows_per_feature)
                self._samples[observation.context][feature_name] = feature_samples

            feature_samples.append(
                BaselineSample(
                    workday=observation.workday,
                    value=value,
                )
            )
            accepted_features.append(feature_name)

        if accepted_features:
            self._accepted_workdays.add(observation.workday)

        accepted = bool(accepted_features)
        if accepted:
            reason = "Приняты качественные обезличенные признаки."
        elif rejected_features:
            reason = "Ни один признак не прошёл проверку качества."
        else:
            reason = "В окне отсутствуют пригодные числовые признаки."

        return BaselineUpdateResult(
            accepted=accepted,
            initial_calibration_complete=self.initial_calibration_complete,
            calibration_day_count=self.calibration_day_count,
            accepted_features=tuple(sorted(accepted_features)),
            rejected_features=rejected_features,
            reason=reason,
        )

    def stats(
        self,
        context: WorkContext,
        feature_name: str,
    ) -> BaselineStats | None:
        clean_name = self._clean_feature_name(feature_name)
        samples = self._samples.get(context, {}).get(clean_name)
        if samples is None or len(samples) < self.min_samples_for_stats:
            return None

        values = [sample.value for sample in samples]
        median_value = statistics.median(values)
        abs_deviations = [abs(value - median_value) for value in values]
        mad_value = statistics.median(abs_deviations)
        q25_value = self._percentile(values, 0.25)
        q75_value = self._percentile(values, 0.75)

        mad_scale = 1.4826 * mad_value
        iqr_scale = max(0.0, q75_value - q25_value) / 1.349
        minimum_scale = self._minimum_scale(clean_name, median_value)
        robust_scale = max(mad_scale, iqr_scale, minimum_scale)

        day_count = len({sample.workday for sample in samples})
        ready = (
            len(samples) >= self.min_windows_per_feature
            and day_count >= self.calibration_days
        )

        sample_confidence = min(
            1.0,
            len(samples) / self.min_windows_per_feature,
        )
        day_confidence = min(
            1.0,
            day_count / self.calibration_days,
        )
        confidence = 0.5 * sample_confidence + 0.5 * day_confidence

        return BaselineStats(
            context=context,
            feature_name=clean_name,
            median=median_value,
            mad=mad_value,
            q25=q25_value,
            q75=q75_value,
            robust_scale=robust_scale,
            sample_count=len(samples),
            day_count=day_count,
            ready=ready,
            confidence=confidence,
        )

    def compare(
        self,
        context: WorkContext,
        feature_name: str,
        actual_value: float | int | None,
        *,
        fallback_to_mixed: bool = True,
    ) -> BaselineComparison:
        clean_name = self._clean_feature_name(feature_name)
        value, value_error = self._validated_feature_value(actual_value)
        if value_error is not None:
            return BaselineComparison(
                available=False,
                ready=False,
                feature_name=clean_name,
                context=context,
                actual_value=None,
                baseline_median=None,
                robust_z=None,
                confidence=0.0,
                reason=value_error,
            )

        stats = self.stats(context, clean_name)
        used_context = context

        if stats is None and fallback_to_mixed and context is not WorkContext.MIXED:
            stats = self.stats(WorkContext.MIXED, clean_name)
            used_context = WorkContext.MIXED

        if stats is None:
            return BaselineComparison(
                available=False,
                ready=False,
                feature_name=clean_name,
                context=used_context,
                actual_value=value,
                baseline_median=None,
                robust_z=None,
                confidence=0.0,
                reason="Для признака пока недостаточно личных данных.",
            )

        assert value is not None
        robust_z = (value - stats.median) / stats.robust_scale

        if stats.ready:
            reason = "Сравнение выполнено с готовой личной нормой."
        else:
            reason = (
                "Сравнение предварительное: личная норма ещё калибруется."
            )

        return BaselineComparison(
            available=True,
            ready=stats.ready,
            feature_name=clean_name,
            context=used_context,
            actual_value=value,
            baseline_median=stats.median,
            robust_z=robust_z,
            confidence=stats.confidence,
            reason=reason,
        )

    def ready_features(self, context: WorkContext) -> tuple[str, ...]:
        result: list[str] = []
        for feature_name in self._samples.get(context, {}):
            stats = self.stats(context, feature_name)
            if stats is not None and stats.ready:
                result.append(feature_name)
        return tuple(sorted(result))

    def sample_count(self, context: WorkContext, feature_name: str) -> int:
        clean_name = self._clean_feature_name(feature_name)
        samples = self._samples.get(context, {}).get(clean_name)
        return 0 if samples is None else len(samples)

    @staticmethod
    def feature_channel(feature_name: str) -> FeatureChannel:
        clean_name = PersonalBaselineEngine._clean_feature_name(feature_name)
        if clean_name.startswith("typing_"):
            return FeatureChannel.TYPING
        if clean_name.startswith("mouse_"):
            return FeatureChannel.MOUSE
        if clean_name.startswith("ocular_"):
            return FeatureChannel.OCULAR
        if clean_name.startswith("posture_") or clean_name.startswith("head_"):
            return FeatureChannel.POSTURE
        return FeatureChannel.GENERIC

    def _general_rejection_reason(
        self,
        observation: BaselineObservation,
        *,
        allow_refresh: bool,
    ) -> str | None:
        if not observation.user_present:
            return "Пользователь отсутствует за рабочим местом."
        if observation.returning:
            return "Защитный период после возвращения не используется для нормы."
        if observation.severe_ocular_event:
            return "Окно содержит выраженный глазной признак сонливости."
        if (
            observation.self_report_fatigue_0_10 is not None
            and observation.self_report_fatigue_0_10
            > self.max_baseline_fatigue_0_10
        ):
            return "Самооценка указывает на заметную усталость."
        if (
            observation.self_report_sleepiness_kss_1_9 is not None
            and observation.self_report_sleepiness_kss_1_9
            > self.max_baseline_sleepiness_kss
        ):
            return "Самооценка указывает на заметную сонливость."
        if (
            observation.minutes_since_workday_start
            > self.baseline_window_minutes
        ):
            return "Окно находится за пределами первых двух часов рабочего дня."
        if self.initial_calibration_complete and not allow_refresh:
            return (
                "Первичная калибровка завершена; обновление нормы требует "
                "allow_refresh=True."
            )
        return None

    def _channel_rejection_reason(
        self,
        observation: BaselineObservation,
        channel: FeatureChannel,
    ) -> str | None:
        quality = self._channel_quality(observation, channel)
        if quality is None:
            return f"Нет оценки качества канала {channel.value}."
        if quality < self.min_channel_quality:
            return f"Низкое качество канала {channel.value}."

        if channel in {FeatureChannel.OCULAR, FeatureChannel.POSTURE}:
            if observation.liveness_score is None:
                return "Для визуального канала нет оценки liveness."
            if observation.liveness_score < self.min_liveness_score:
                return "Liveness визуального канала не подтверждён."

        return None

    @staticmethod
    def _channel_quality(
        observation: BaselineObservation,
        channel: FeatureChannel,
    ) -> float | None:
        for raw_channel, quality in observation.channel_quality.items():
            if FeatureChannel(str(raw_channel)) is channel:
                return None if quality is None else float(quality)
        return None

    def _minimum_scale(self, feature_name: str, median_value: float) -> float:
        explicit = self._min_scales.get(feature_name)
        if explicit is not None:
            return explicit
        return max(abs(median_value) * 0.03, 1e-6)

    @staticmethod
    def _clean_feature_name(raw_name: object) -> str:
        if not isinstance(raw_name, str):
            raise TypeError("Название признака должно быть строкой.")
        clean_name = raw_name.strip().lower()
        if not clean_name:
            raise ValueError("Название признака не может быть пустым.")
        return clean_name

    @staticmethod
    def _validated_feature_value(
        raw_value: float | int | None,
    ) -> tuple[float | None, str | None]:
        if raw_value is None:
            return None, "Значение отсутствует."
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            return None, "Значение должно быть числом."
        value = float(raw_value)
        if not math.isfinite(value):
            return None, "Значение должно быть конечным числом."
        return value, None

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            raise ValueError("Нельзя вычислить процентиль пустого набора.")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction должен быть в диапазоне 0..1.")

        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (len(sorted_values) - 1) * fraction
        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        weight = position - lower_index
        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        return lower_value + (upper_value - lower_value) * weight