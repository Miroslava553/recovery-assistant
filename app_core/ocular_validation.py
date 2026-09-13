from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from app_core.ocular_activity import OcularEventType


class ValidationDataError(RuntimeError):
    """Валидационная сессия неполна или внутренне противоречива."""


class ValidationLabelKind(str, Enum):
    FULL_BLINK = "full_blink"
    INCOMPLETE_BLINK = "incomplete_blink"
    LEFT_WINK = "left_wink"
    RIGHT_WINK = "right_wink"

    @property
    def is_bilateral(self) -> bool:
        return self in {
            ValidationLabelKind.FULL_BLINK,
            ValidationLabelKind.INCOMPLETE_BLINK,
        }


class EyeSampleFailureReason(str, Enum):
    """Причина, по которой кадр нельзя использовать для анализа глаз."""

    FACE_NOT_DETECTED = "face_not_detected"
    IMAGE_QUALITY_LOW_OR_INVALID = "image_quality_low_or_invalid"
    LEFT_EYE_QUALITY_LOW_OR_INVALID = "left_eye_quality_low_or_invalid"
    RIGHT_EYE_QUALITY_LOW_OR_INVALID = "right_eye_quality_low_or_invalid"
    HEAD_POSE_QUALITY_LOW_OR_INVALID = "head_pose_quality_low_or_invalid"
    LEFT_EYE_SIGNAL_MISSING_OR_INVALID = "left_eye_signal_missing_or_invalid"
    RIGHT_EYE_SIGNAL_MISSING_OR_INVALID = "right_eye_signal_missing_or_invalid"


@dataclass(frozen=True, slots=True)
class EyeSampleQualityThresholds:
    min_data_quality: float = 0.55
    min_eye_quality: float = 0.55
    min_head_pose_quality: float = 0.35

    def __post_init__(self) -> None:
        for name, value in (
            ("min_data_quality", self.min_data_quality),
            ("min_eye_quality", self.min_eye_quality),
            ("min_head_pose_quality", self.min_head_pose_quality),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} должен быть конечным числом от 0 до 1.")


@dataclass(frozen=True, slots=True)
class EyeSampleQualityAssessment:
    valid: bool
    failed_reasons: tuple[EyeSampleFailureReason, ...]
    primary_reason: EyeSampleFailureReason | None


@dataclass(frozen=True, slots=True)
class HeadPoseReference:
    """Персональная нейтральная поза головы из калибровочной части сессии.

    Это не оценка личности и не биометрический шаблон. Сохраняются только
    два безразмерных геометрических отношения и устойчивый разброс вокруг
    них. Они нужны, чтобы не сравнивать всех людей с одной условной
    вертикальной пропорцией лица.
    """

    neutral_turn_ratio: float
    neutral_vertical_ratio: float
    turn_mad: float
    vertical_mad: float
    sample_count: int
    calibration_duration_sec: float

    def __post_init__(self) -> None:
        numeric = (
            self.neutral_turn_ratio,
            self.neutral_vertical_ratio,
            self.turn_mad,
            self.vertical_mad,
            self.calibration_duration_sec,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Параметры HeadPoseReference должны быть конечными.")
        if self.turn_mad < 0.0 or self.vertical_mad < 0.0:
            raise ValueError("MAD не может быть отрицательным.")
        if self.sample_count < 1:
            raise ValueError("Для HeadPoseReference нужен хотя бы один кадр.")
        if self.calibration_duration_sec <= 0.0:
            raise ValueError("Длительность калибровки должна быть положительной.")


@dataclass(frozen=True, slots=True)
class ValidationSample:
    frame_index: int
    timestamp_sec: float
    left_eye_opening_raw: float | None
    right_eye_opening_raw: float | None
    data_quality: float
    left_eye_quality: float
    right_eye_quality: float
    head_pose_quality: float
    face_detected: bool
    valid: bool
    brightness: float | None = None
    sharpness: float | None = None
    horizontal_gaze_ratio: float | None = None
    head_turn_ratio: float | None = None
    head_vertical_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class NormalizedValidationSample:
    frame_index: int
    timestamp_sec: float
    left_eye_opening_raw: float | None
    right_eye_opening_raw: float | None
    left_eye_opening_normalized: float | None
    right_eye_opening_normalized: float | None
    data_quality: float
    left_eye_quality: float
    right_eye_quality: float
    head_pose_quality: float
    face_detected: bool
    valid: bool
    brightness: float | None = None
    sharpness: float | None = None
    horizontal_gaze_ratio: float | None = None
    head_turn_ratio: float | None = None
    head_vertical_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class ManualBlinkLabel:
    frame_index: int
    timestamp_sec: float
    kind: ValidationLabelKind


@dataclass(frozen=True, slots=True)
class ReferenceEstimate:
    left_open_eye_reference: float
    right_open_eye_reference: float
    left_relative_mad: float
    right_relative_mad: float
    valid_sample_count: int
    calibration_duration_sec: float


@dataclass(frozen=True, slots=True)
class OfflineBlinkConfig:
    """Параметры только для офлайн-проверки записанного сигнала.

    Чем меньше нормализованное раскрытие, тем сильнее закрыт глаз.
    Порядок порогов обязан сохраняться: full < partial < open.
    """

    full_blink_ratio_threshold: float = 0.82
    partial_ratio_threshold: float = 0.90
    open_ratio_threshold: float = 0.94
    smoothing_alpha: float = 0.80
    min_blink_sec: float = 0.03
    max_blink_sec: float = 0.70
    bilateral_sync_tolerance_sec: float = 0.12
    max_sample_gap_sec: float = 0.20
    candidate_gap_tolerance_sec: float = 0.14
    reopen_confirmation_samples: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.full_blink_ratio_threshold < self.partial_ratio_threshold:
            raise ValueError("full_blink_ratio_threshold должен быть меньше partial.")
        if not self.partial_ratio_threshold < self.open_ratio_threshold <= 1.20:
            raise ValueError("Нужен порядок full < partial < open.")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha должен быть в диапазоне (0, 1].")
        if not 0.0 < self.min_blink_sec < self.max_blink_sec:
            raise ValueError("Длительности моргания заданы неверно.")
        if self.bilateral_sync_tolerance_sec <= 0:
            raise ValueError("bilateral_sync_tolerance_sec должен быть больше нуля.")
        if self.max_sample_gap_sec <= 0 or self.candidate_gap_tolerance_sec <= 0:
            raise ValueError("Допустимые разрывы должны быть больше нуля.")
        if self.reopen_confirmation_samples < 1:
            raise ValueError("Нужно хотя бы одно подтверждение открытия.")


@dataclass(frozen=True, slots=True)
class OfflineBlinkEvent:
    started_at: float
    ended_at: float
    center_at: float
    duration_sec: float
    event_type: OcularEventType
    min_left_normalized: float
    min_right_normalized: float
    min_left_at: float
    min_right_at: float
    sync_offset_sec: float

    @property
    def bilateral_depth(self) -> float:
        """Глубина, которую гарантированно достигли оба глаза.

        Используется максимум двух минимумов: если один глаз закрылся слабее,
        именно он ограничивает уверенность в двустороннем закрытии.
        """
        return max(self.min_left_normalized, self.min_right_normalized)


@dataclass(frozen=True, slots=True)
class MatchedBlink:
    label: ManualBlinkLabel
    event: OfflineBlinkEvent
    timing_error_sec: float


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    true_bilateral_count: int
    predicted_bilateral_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_absolute_timing_error_ms: float | None
    classification_accuracy: float | None
    matched_count: int


@dataclass(frozen=True, slots=True)
class ThresholdSearchResult:
    config: OfflineBlinkConfig
    metrics: ValidationMetrics
    classification_threshold_validated: bool
    classification_balanced_accuracy: float | None
    evaluated_config_count: int


@dataclass(slots=True)
class _OfflineCandidate:
    started_at: float
    last_valid_at: float
    min_left: float
    min_right: float
    min_left_at: float
    min_right_at: float
    reopen_confirmations: int = 0


SAMPLE_FIELDNAMES = (
    "frame_index",
    "timestamp_sec",
    "left_eye_opening_raw",
    "right_eye_opening_raw",
    "data_quality",
    "left_eye_quality",
    "right_eye_quality",
    "head_pose_quality",
    "face_detected",
    "valid",
    "brightness",
    "sharpness",
    "horizontal_gaze_ratio",
    "head_turn_ratio",
    "head_vertical_ratio",
)

REQUIRED_SAMPLE_FIELDNAMES = (
    "frame_index",
    "timestamp_sec",
    "left_eye_opening_raw",
    "right_eye_opening_raw",
    "data_quality",
    "left_eye_quality",
    "right_eye_quality",
    "head_pose_quality",
    "face_detected",
    "valid",
)


def estimate_head_pose_reference(
    samples: Sequence[ValidationSample],
    *,
    calibration_duration_sec: float,
    min_candidate_count: int = 30,
    min_data_quality: float = 0.55,
    min_eye_quality: float = 0.55,
) -> HeadPoseReference:
    """Оценивает личную нейтральную позу по первым секундам записи.

    В выборку не попадают кадры без лица, с плохим изображением, слабой
    геометрией глаз или вероятным морганием. Универсальная точка вроде
    ``head_vertical_ratio == 0.52`` намеренно не используется: этот
    показатель меняется от формы лица, высоты камеры и перспективы.
    """
    if calibration_duration_sec <= 0.0:
        raise ValueError("calibration_duration_sec должен быть больше нуля.")
    if min_candidate_count < 5:
        raise ValueError("min_candidate_count должен быть не меньше 5.")

    preliminary: list[ValidationSample] = []
    for sample in samples:
        if sample.timestamp_sec < 0.0 or sample.timestamp_sec > calibration_duration_sec:
            continue
        if not sample.face_detected:
            continue
        if not _finite_at_least(sample.data_quality, min_data_quality):
            continue
        if not _finite_at_least(sample.left_eye_quality, min_eye_quality):
            continue
        if not _finite_at_least(sample.right_eye_quality, min_eye_quality):
            continue
        if not _valid_eye_opening(sample.left_eye_opening_raw):
            continue
        if not _valid_eye_opening(sample.right_eye_opening_raw):
            continue
        if not _finite_number(sample.head_turn_ratio):
            continue
        if not _finite_number(sample.head_vertical_ratio):
            continue
        preliminary.append(sample)

    if len(preliminary) < min_candidate_count:
        raise ValidationDataError(
            "Недостаточно качественных кадров калибровки для личной нормы "
            f"положения головы: {len(preliminary)}, требуется не меньше "
            f"{min_candidate_count}."
        )

    left_median = statistics.median(
        float(sample.left_eye_opening_raw) for sample in preliminary
        if sample.left_eye_opening_raw is not None
    )
    right_median = statistics.median(
        float(sample.right_eye_opening_raw) for sample in preliminary
        if sample.right_eye_opening_raw is not None
    )

    # Убираем случайные моргания во время калибровки. Граница мягкая: она
    # должна исключить глубокий провал, но сохранить обычную вариативность
    # открытых глаз.
    filtered = [
        sample
        for sample in preliminary
        if float(sample.left_eye_opening_raw) >= 0.65 * left_median
        and float(sample.right_eye_opening_raw) >= 0.65 * right_median
    ]
    if len(filtered) < min_candidate_count:
        raise ValidationDataError(
            "После исключения вероятных морганий осталось слишком мало "
            f"кадров калибровки: {len(filtered)}."
        )

    turn_values = [float(sample.head_turn_ratio) for sample in filtered]
    vertical_values = [float(sample.head_vertical_ratio) for sample in filtered]
    neutral_turn = float(statistics.median(turn_values))
    neutral_vertical = float(statistics.median(vertical_values))
    turn_mad = _median_absolute_deviation(turn_values, neutral_turn)
    vertical_mad = _median_absolute_deviation(vertical_values, neutral_vertical)

    return HeadPoseReference(
        neutral_turn_ratio=neutral_turn,
        neutral_vertical_ratio=neutral_vertical,
        turn_mad=turn_mad,
        vertical_mad=vertical_mad,
        sample_count=len(filtered),
        calibration_duration_sec=float(calibration_duration_sec),
    )


def calibrated_head_pose_quality(
    *,
    head_turn_ratio: float | None,
    head_vertical_ratio: float | None,
    reference: HeadPoseReference,
) -> float:
    """Качество ракурса относительно личной нейтральной позы.

    Небольшие естественные движения получают качество 1.0. Затем качество
    плавно падает и становится нулевым только при сильном отклонении, когда
    перспектива действительно мешает измерять веки. Разброс калибровки
    учитывается, но не может сделать модель бесконтрольно мягкой.
    """
    if not _finite_number(head_turn_ratio) or not _finite_number(head_vertical_ratio):
        return 0.0

    turn_deviation = abs(float(head_turn_ratio) - reference.neutral_turn_ratio)
    vertical_deviation = abs(
        float(head_vertical_ratio) - reference.neutral_vertical_ratio
    )

    turn_soft = 0.06 + min(0.04, 3.0 * reference.turn_mad)
    turn_hard = 0.24 + min(0.08, 6.0 * reference.turn_mad)
    vertical_soft = 0.08 + min(0.05, 3.0 * reference.vertical_mad)
    vertical_hard = 0.30 + min(0.10, 6.0 * reference.vertical_mad)

    turn_quality = _plateau_quality(
        turn_deviation,
        full_quality_until=turn_soft,
        zero_quality_at=turn_hard,
    )
    vertical_quality = _plateau_quality(
        vertical_deviation,
        full_quality_until=vertical_soft,
        zero_quality_at=vertical_hard,
    )
    return float(min(turn_quality, vertical_quality))


def apply_personal_head_pose_quality(
    samples: Sequence[ValidationSample],
    *,
    reference: HeadPoseReference,
    thresholds: EyeSampleQualityThresholds | None = None,
) -> list[ValidationSample]:
    """Возвращает копии кадров с персонально пересчитанным pose quality.

    Поле ``valid`` пересчитывается теми же прозрачными правилами качества,
    поэтому старые сессии можно анализировать повторно без изменения видео
    и без потери исходных head_turn/head_vertical значений.
    """
    active_thresholds = thresholds or EyeSampleQualityThresholds()
    result: list[ValidationSample] = []
    for sample in samples:
        pose_quality = calibrated_head_pose_quality(
            head_turn_ratio=sample.head_turn_ratio,
            head_vertical_ratio=sample.head_vertical_ratio,
            reference=reference,
        )
        assessment = assess_eye_sample_quality(
            face_detected=sample.face_detected,
            left_eye_opening_raw=sample.left_eye_opening_raw,
            right_eye_opening_raw=sample.right_eye_opening_raw,
            data_quality=sample.data_quality,
            left_eye_quality=sample.left_eye_quality,
            right_eye_quality=sample.right_eye_quality,
            head_pose_quality=pose_quality,
            thresholds=active_thresholds,
        )
        result.append(
            replace(
                sample,
                head_pose_quality=pose_quality,
                valid=assessment.valid,
            )
        )
    return result


def _median_absolute_deviation(values: Sequence[float], center: float) -> float:
    if not values:
        return 0.0
    return float(statistics.median(abs(value - center) for value in values))


def _plateau_quality(
    deviation: float,
    *,
    full_quality_until: float,
    zero_quality_at: float,
) -> float:
    if zero_quality_at <= full_quality_until:
        raise ValueError("zero_quality_at должен быть больше full_quality_until.")
    if deviation <= full_quality_until:
        return 1.0
    if deviation >= zero_quality_at:
        return 0.0
    return float(1.0 - (deviation - full_quality_until) / (zero_quality_at - full_quality_until))


def _finite_number(value: float | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def assess_eye_sample_quality(
    *,
    face_detected: bool,
    left_eye_opening_raw: float | None,
    right_eye_opening_raw: float | None,
    data_quality: float,
    left_eye_quality: float,
    right_eye_quality: float,
    head_pose_quality: float,
    thresholds: EyeSampleQualityThresholds | None = None,
    min_data_quality: float = 0.55,
    min_eye_quality: float = 0.55,
    min_head_pose_quality: float = 0.35,
) -> EyeSampleQualityAssessment:
    """Проверяет все критерии и сохраняет каждую причину браковки.

    Параметр ``thresholds`` предпочтителен. Три отдельных порога сохранены
    для обратной совместимости с прежними вызовами. Причины не являются
    взаимоисключающими: один кадр может одновременно иметь плохое освещение
    и неудобный ракурс головы.
    """
    active_thresholds = thresholds or EyeSampleQualityThresholds(
        min_data_quality=min_data_quality,
        min_eye_quality=min_eye_quality,
        min_head_pose_quality=min_head_pose_quality,
    )

    failed: list[EyeSampleFailureReason] = []
    if not face_detected:
        failed.append(EyeSampleFailureReason.FACE_NOT_DETECTED)

    if not _finite_at_least(data_quality, active_thresholds.min_data_quality):
        failed.append(EyeSampleFailureReason.IMAGE_QUALITY_LOW_OR_INVALID)
    if not _finite_at_least(left_eye_quality, active_thresholds.min_eye_quality):
        failed.append(EyeSampleFailureReason.LEFT_EYE_QUALITY_LOW_OR_INVALID)
    if not _finite_at_least(right_eye_quality, active_thresholds.min_eye_quality):
        failed.append(EyeSampleFailureReason.RIGHT_EYE_QUALITY_LOW_OR_INVALID)
    if not _finite_at_least(
        head_pose_quality,
        active_thresholds.min_head_pose_quality,
    ):
        failed.append(EyeSampleFailureReason.HEAD_POSE_QUALITY_LOW_OR_INVALID)

    if not _valid_eye_opening(left_eye_opening_raw):
        failed.append(EyeSampleFailureReason.LEFT_EYE_SIGNAL_MISSING_OR_INVALID)
    if not _valid_eye_opening(right_eye_opening_raw):
        failed.append(EyeSampleFailureReason.RIGHT_EYE_SIGNAL_MISSING_OR_INVALID)

    failed_tuple = tuple(failed)
    return EyeSampleQualityAssessment(
        valid=not failed_tuple,
        failed_reasons=failed_tuple,
        primary_reason=failed_tuple[0] if failed_tuple else None,
    )


def is_valid_eye_sample(
    *,
    face_detected: bool,
    left_eye_opening_raw: float | None,
    right_eye_opening_raw: float | None,
    data_quality: float,
    left_eye_quality: float,
    right_eye_quality: float,
    head_pose_quality: float,
    min_data_quality: float = 0.55,
    min_eye_quality: float = 0.55,
    min_head_pose_quality: float = 0.35,
) -> bool:
    return assess_eye_sample_quality(
        face_detected=face_detected,
        left_eye_opening_raw=left_eye_opening_raw,
        right_eye_opening_raw=right_eye_opening_raw,
        data_quality=data_quality,
        left_eye_quality=left_eye_quality,
        right_eye_quality=right_eye_quality,
        head_pose_quality=head_pose_quality,
        min_data_quality=min_data_quality,
        min_eye_quality=min_eye_quality,
        min_head_pose_quality=min_head_pose_quality,
    ).valid


def summarize_eye_sample_quality(
    samples: Sequence[ValidationSample],
    *,
    thresholds: EyeSampleQualityThresholds | None = None,
    start_after_sec: float = 0.0,
) -> dict[str, object]:
    """Возвращает прозрачную статистику причин непригодности.

    ``reason_counts`` и ``reason_percent_of_frames`` могут суммироваться
    больше чем до 100%, потому что один кадр способен провалить несколько
    критериев одновременно. ``primary_reason_counts`` взаимоисключающие и
    предназначены только для определения доминирующей причины.
    """
    if start_after_sec < 0.0:
        raise ValueError("start_after_sec не может быть отрицательным.")
    active_thresholds = thresholds or EyeSampleQualityThresholds()
    selected = [sample for sample in samples if sample.timestamp_sec >= start_after_sec]

    reason_counts = {reason.value: 0 for reason in EyeSampleFailureReason}
    primary_counts = {reason.value: 0 for reason in EyeSampleFailureReason}
    valid_count = 0

    for sample in selected:
        assessment = assess_eye_sample_quality(
            face_detected=sample.face_detected,
            left_eye_opening_raw=sample.left_eye_opening_raw,
            right_eye_opening_raw=sample.right_eye_opening_raw,
            data_quality=sample.data_quality,
            left_eye_quality=sample.left_eye_quality,
            right_eye_quality=sample.right_eye_quality,
            head_pose_quality=sample.head_pose_quality,
            thresholds=active_thresholds,
        )
        if assessment.valid:
            valid_count += 1
        for reason in assessment.failed_reasons:
            reason_counts[reason.value] += 1
        if assessment.primary_reason is not None:
            primary_counts[assessment.primary_reason.value] += 1

    total_count = len(selected)
    invalid_count = total_count - valid_count
    percent_frames = {
        reason: (count / total_count if total_count else 0.0)
        for reason, count in reason_counts.items()
    }
    percent_invalid = {
        reason: (count / invalid_count if invalid_count else 0.0)
        for reason, count in reason_counts.items()
    }
    dominant_reason = None
    if invalid_count:
        dominant_reason = max(
            primary_counts,
            key=lambda reason: primary_counts[reason],
        )
        if primary_counts[dominant_reason] == 0:
            dominant_reason = None

    return {
        "start_after_sec": start_after_sec,
        "thresholds": asdict(active_thresholds),
        "total_frame_count": total_count,
        "valid_frame_count": valid_count,
        "invalid_frame_count": invalid_count,
        "frame_coverage": valid_count / total_count if total_count else 0.0,
        "reason_counts": reason_counts,
        "reason_percent_of_frames": percent_frames,
        "reason_percent_of_invalid_frames": percent_invalid,
        "primary_reason_counts": primary_counts,
        "dominant_primary_reason": dominant_reason,
        "reason_counts_can_overlap": True,
    }


def _finite_at_least(value: float, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= threshold


def _valid_eye_opening(value: float | None) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def save_samples(path: str | Path, samples: Sequence[ValidationSample]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDNAMES)
        writer.writeheader()
        for sample in samples:
            row = asdict(sample)
            row["face_detected"] = int(sample.face_detected)
            row["valid"] = int(sample.valid)
            writer.writerow(row)
    temporary.replace(destination)


def load_samples(path: str | Path) -> list[ValidationSample]:
    source = Path(path)
    if not source.exists():
        raise ValidationDataError(f"Не найден файл сигналов: {source}")
    samples: list[ValidationSample] = []
    with source.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(REQUIRED_SAMPLE_FIELDNAMES) - set(reader.fieldnames or ())
        if missing:
            raise ValidationDataError(
                "В signals.csv отсутствуют столбцы: " + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                samples.append(
                    ValidationSample(
                        frame_index=int(row["frame_index"]),
                        timestamp_sec=float(row["timestamp_sec"]),
                        left_eye_opening_raw=_parse_optional_float(
                            row["left_eye_opening_raw"]
                        ),
                        right_eye_opening_raw=_parse_optional_float(
                            row["right_eye_opening_raw"]
                        ),
                        data_quality=float(row["data_quality"]),
                        left_eye_quality=float(row["left_eye_quality"]),
                        right_eye_quality=float(row["right_eye_quality"]),
                        head_pose_quality=float(row["head_pose_quality"]),
                        face_detected=_parse_bool(row["face_detected"]),
                        valid=_parse_bool(row["valid"]),
                        brightness=_parse_optional_float(row.get("brightness")),
                        sharpness=_parse_optional_float(row.get("sharpness")),
                        horizontal_gaze_ratio=_parse_optional_float(
                            row.get("horizontal_gaze_ratio")
                        ),
                        head_turn_ratio=_parse_optional_float(
                            row.get("head_turn_ratio")
                        ),
                        head_vertical_ratio=_parse_optional_float(
                            row.get("head_vertical_ratio")
                        ),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValidationDataError(
                    f"Некорректная строка {row_number} в signals.csv."
                ) from error
    _validate_sample_sequence(samples)
    return samples


def save_labels(path: str | Path, labels: Sequence[ManualBlinkLabel]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "labels": [
            {
                "frame_index": label.frame_index,
                "timestamp_sec": label.timestamp_sec,
                "kind": label.kind.value,
            }
            for label in sorted(labels, key=lambda item: item.timestamp_sec)
        ],
    }
    _atomic_json_write(destination, payload)


def load_labels(path: str | Path) -> list[ManualBlinkLabel]:
    source = Path(path)
    if not source.exists():
        return []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationDataError(f"Не удалось прочитать разметку: {source}") from error
    labels: list[ManualBlinkLabel] = []
    for item in payload.get("labels", []):
        try:
            labels.append(
                ManualBlinkLabel(
                    frame_index=int(item["frame_index"]),
                    timestamp_sec=float(item["timestamp_sec"]),
                    kind=ValidationLabelKind(item["kind"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationDataError("labels.json содержит некорректную запись.") from error
    labels.sort(key=lambda item: item.timestamp_sec)
    return labels


def estimate_open_eye_reference(
    samples: Sequence[ValidationSample],
    *,
    calibration_sec: float,
    min_valid_samples: int = 30,
    min_time_span_sec: float = 2.0,
) -> ReferenceEstimate:
    if calibration_sec <= 0:
        raise ValueError("calibration_sec должен быть больше нуля.")
    calibration_samples = [
        sample
        for sample in samples
        if sample.timestamp_sec <= calibration_sec
        and sample.valid
        and sample.left_eye_opening_raw is not None
        and sample.right_eye_opening_raw is not None
    ]
    if len(calibration_samples) < min_valid_samples:
        raise ValidationDataError(
            "Для калибровки недостаточно пригодных кадров: "
            f"{len(calibration_samples)} из требуемых {min_valid_samples}."
        )
    span = calibration_samples[-1].timestamp_sec - calibration_samples[0].timestamp_sec
    if span < min_time_span_sec:
        raise ValidationDataError(
            f"Калибровка покрывает только {span:.2f} с; требуется не менее "
            f"{min_time_span_sec:.2f} с."
        )

    left_values = [float(sample.left_eye_opening_raw) for sample in calibration_samples]
    right_values = [float(sample.right_eye_opening_raw) for sample in calibration_samples]
    left_reference = float(statistics.median(left_values))
    right_reference = float(statistics.median(right_values))
    if left_reference <= 0.0 or right_reference <= 0.0:
        raise ValidationDataError("Получен неположительный эталон раскрытия глаз.")

    left_mad = _median_absolute_deviation(left_values, left_reference)
    right_mad = _median_absolute_deviation(right_values, right_reference)
    return ReferenceEstimate(
        left_open_eye_reference=left_reference,
        right_open_eye_reference=right_reference,
        left_relative_mad=left_mad / left_reference,
        right_relative_mad=right_mad / right_reference,
        valid_sample_count=len(calibration_samples),
        calibration_duration_sec=span,
    )


def normalize_samples(
    samples: Sequence[ValidationSample],
    reference: ReferenceEstimate,
) -> list[NormalizedValidationSample]:
    normalized: list[NormalizedValidationSample] = []
    for sample in samples:
        left_normalized = (
            float(sample.left_eye_opening_raw) / reference.left_open_eye_reference
            if sample.valid and sample.left_eye_opening_raw is not None
            else None
        )
        right_normalized = (
            float(sample.right_eye_opening_raw) / reference.right_open_eye_reference
            if sample.valid and sample.right_eye_opening_raw is not None
            else None
        )
        normalized.append(
            NormalizedValidationSample(
                frame_index=sample.frame_index,
                timestamp_sec=sample.timestamp_sec,
                left_eye_opening_raw=sample.left_eye_opening_raw,
                right_eye_opening_raw=sample.right_eye_opening_raw,
                left_eye_opening_normalized=left_normalized,
                right_eye_opening_normalized=right_normalized,
                data_quality=sample.data_quality,
                left_eye_quality=sample.left_eye_quality,
                right_eye_quality=sample.right_eye_quality,
                head_pose_quality=sample.head_pose_quality,
                face_detected=sample.face_detected,
                valid=sample.valid and left_normalized is not None and right_normalized is not None,
                brightness=sample.brightness,
                sharpness=sample.sharpness,
                horizontal_gaze_ratio=sample.horizontal_gaze_ratio,
                head_turn_ratio=sample.head_turn_ratio,
                head_vertical_ratio=sample.head_vertical_ratio,
            )
        )
    return normalized


def detect_offline_blinks(
    samples: Sequence[NormalizedValidationSample],
    config: OfflineBlinkConfig,
    *,
    start_after_sec: float = 0.0,
) -> list[OfflineBlinkEvent]:
    events: list[OfflineBlinkEvent] = []
    candidate: _OfflineCandidate | None = None
    smoothed_left: float | None = None
    smoothed_right: float | None = None
    previous_timestamp: float | None = None

    for sample in samples:
        timestamp = sample.timestamp_sec
        if timestamp < start_after_sec:
            continue

        if previous_timestamp is not None:
            gap = timestamp - previous_timestamp
            if gap <= 0.0:
                previous_timestamp = timestamp
                continue
            if gap > config.max_sample_gap_sec:
                candidate = None
                smoothed_left = None
                smoothed_right = None
        previous_timestamp = timestamp

        if (
            not sample.valid
            or sample.left_eye_opening_normalized is None
            or sample.right_eye_opening_normalized is None
        ):
            if (
                candidate is not None
                and timestamp - candidate.last_valid_at
                > config.candidate_gap_tolerance_sec
            ):
                candidate = None
            smoothed_left = None
            smoothed_right = None
            continue

        left_value = float(sample.left_eye_opening_normalized)
        right_value = float(sample.right_eye_opening_normalized)
        smoothed_left = _ema(smoothed_left, left_value, config.smoothing_alpha)
        smoothed_right = _ema(smoothed_right, right_value, config.smoothing_alpha)

        any_dip = (
            smoothed_left <= config.partial_ratio_threshold
            or smoothed_right <= config.partial_ratio_threshold
        )
        both_open = (
            smoothed_left >= config.open_ratio_threshold
            and smoothed_right >= config.open_ratio_threshold
        )

        if candidate is None:
            if any_dip:
                candidate = _OfflineCandidate(
                    started_at=timestamp,
                    last_valid_at=timestamp,
                    min_left=smoothed_left,
                    min_right=smoothed_right,
                    min_left_at=timestamp,
                    min_right_at=timestamp,
                )
            continue

        candidate.last_valid_at = timestamp
        if smoothed_left < candidate.min_left:
            candidate.min_left = smoothed_left
            candidate.min_left_at = timestamp
        if smoothed_right < candidate.min_right:
            candidate.min_right = smoothed_right
            candidate.min_right_at = timestamp

        if both_open:
            candidate.reopen_confirmations += 1
            if candidate.reopen_confirmations >= config.reopen_confirmation_samples:
                event = _finalize_offline_candidate(candidate, timestamp, config)
                if event is not None:
                    events.append(event)
                candidate = None
            continue

        candidate.reopen_confirmations = 0
        if timestamp - candidate.started_at > max(config.max_blink_sec * 3.0, 2.0):
            candidate = None

    return events


def match_bilateral_events(
    events: Sequence[OfflineBlinkEvent],
    labels: Sequence[ManualBlinkLabel],
    *,
    tolerance_sec: float = 0.25,
) -> tuple[list[MatchedBlink], list[OfflineBlinkEvent], list[ManualBlinkLabel]]:
    if tolerance_sec <= 0:
        raise ValueError("tolerance_sec должен быть больше нуля.")
    predicted = [
        event
        for event in events
        if event.event_type
        in {
            OcularEventType.BILATERAL_BLINK,
            OcularEventType.INCOMPLETE_BILATERAL_BLINK,
        }
    ]
    truth = [label for label in labels if label.kind.is_bilateral]

    possible_pairs: list[tuple[float, int, int]] = []
    for event_index, event in enumerate(predicted):
        for label_index, label in enumerate(truth):
            error = abs(event.center_at - label.timestamp_sec)
            if error <= tolerance_sec:
                possible_pairs.append((error, event_index, label_index))
    possible_pairs.sort(key=lambda item: item[0])

    used_events: set[int] = set()
    used_labels: set[int] = set()
    matched: list[MatchedBlink] = []
    for error, event_index, label_index in possible_pairs:
        if event_index in used_events or label_index in used_labels:
            continue
        used_events.add(event_index)
        used_labels.add(label_index)
        event = predicted[event_index]
        label = truth[label_index]
        matched.append(
            MatchedBlink(
                label=label,
                event=event,
                timing_error_sec=event.center_at - label.timestamp_sec,
            )
        )

    false_positives = [
        event for index, event in enumerate(predicted) if index not in used_events
    ]
    missed = [label for index, label in enumerate(truth) if index not in used_labels]
    matched.sort(key=lambda item: item.label.timestamp_sec)
    return matched, false_positives, missed


def calculate_validation_metrics(
    events: Sequence[OfflineBlinkEvent],
    labels: Sequence[ManualBlinkLabel],
    *,
    tolerance_sec: float = 0.25,
) -> ValidationMetrics:
    matched, false_positives, missed = match_bilateral_events(
        events,
        labels,
        tolerance_sec=tolerance_sec,
    )
    true_count = sum(label.kind.is_bilateral for label in labels)
    predicted_count = sum(
        event.event_type
        in {
            OcularEventType.BILATERAL_BLINK,
            OcularEventType.INCOMPLETE_BILATERAL_BLINK,
        }
        for event in events
    )
    true_positive_count = len(matched)
    false_positive_count = len(false_positives)
    false_negative_count = len(missed)

    precision = (
        true_positive_count / predicted_count if predicted_count > 0 else None
    )
    recall = true_positive_count / true_count if true_count > 0 else None
    f1 = _f1(precision, recall)
    timing_error_ms = (
        statistics.fmean(abs(item.timing_error_sec) for item in matched) * 1000.0
        if matched
        else None
    )

    classification_results: list[bool] = []
    for item in matched:
        predicted_kind = (
            ValidationLabelKind.FULL_BLINK
            if item.event.event_type is OcularEventType.BILATERAL_BLINK
            else ValidationLabelKind.INCOMPLETE_BLINK
        )
        classification_results.append(predicted_kind is item.label.kind)
    classification_accuracy = (
        sum(classification_results) / len(classification_results)
        if classification_results
        else None
    )

    return ValidationMetrics(
        true_bilateral_count=true_count,
        predicted_bilateral_count=predicted_count,
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_absolute_timing_error_ms=timing_error_ms,
        classification_accuracy=classification_accuracy,
        matched_count=true_positive_count,
    )


def optimize_offline_thresholds(
    samples: Sequence[NormalizedValidationSample],
    labels: Sequence[ManualBlinkLabel],
    *,
    calibration_sec: float,
    default_config: OfflineBlinkConfig | None = None,
    tolerance_sec: float = 0.25,
    min_bilateral_labels: int = 5,
) -> ThresholdSearchResult:
    truth = [
        label
        for label in labels
        if label.kind.is_bilateral and label.timestamp_sec > calibration_sec
    ]
    if len(truth) < min_bilateral_labels:
        raise ValidationDataError(
            "Для подбора порогов нужно разметить не менее "
            f"{min_bilateral_labels} двусторонних морганий. Сейчас: {len(truth)}."
        )

    baseline = default_config or OfflineBlinkConfig()
    best_config: OfflineBlinkConfig | None = None
    best_metrics: ValidationMetrics | None = None
    best_events: list[OfflineBlinkEvent] = []
    best_score: tuple[float, float, float, int, float] | None = None
    evaluated_count = 0

    partial_values = _float_range(0.64, 0.96, 0.02)
    open_margins = (0.03, 0.05, 0.07)
    smoothing_values = (0.65, 0.80, 1.0)
    sync_values = (0.08, 0.12, 0.16)

    for partial in partial_values:
        for open_margin in open_margins:
            open_threshold = min(1.08, partial + open_margin)
            if open_threshold <= partial:
                continue
            provisional_full = min(baseline.full_blink_ratio_threshold, partial - 0.01)
            provisional_full = max(0.40, provisional_full)
            if provisional_full >= partial:
                continue
            for smoothing_alpha in smoothing_values:
                for sync_tolerance in sync_values:
                    config = OfflineBlinkConfig(
                        full_blink_ratio_threshold=provisional_full,
                        partial_ratio_threshold=partial,
                        open_ratio_threshold=open_threshold,
                        smoothing_alpha=smoothing_alpha,
                        min_blink_sec=baseline.min_blink_sec,
                        max_blink_sec=baseline.max_blink_sec,
                        bilateral_sync_tolerance_sec=sync_tolerance,
                        max_sample_gap_sec=baseline.max_sample_gap_sec,
                        candidate_gap_tolerance_sec=baseline.candidate_gap_tolerance_sec,
                        reopen_confirmation_samples=baseline.reopen_confirmation_samples,
                    )
                    events = detect_offline_blinks(
                        samples,
                        config,
                        start_after_sec=calibration_sec,
                    )
                    metrics = calculate_validation_metrics(
                        events,
                        truth,
                        tolerance_sec=tolerance_sec,
                    )
                    evaluated_count += 1
                    f1 = metrics.f1 if metrics.f1 is not None else -1.0
                    precision = metrics.precision if metrics.precision is not None else -1.0
                    recall = metrics.recall if metrics.recall is not None else -1.0
                    score = (
                        f1,
                        precision,
                        recall,
                        -metrics.false_positive_count,
                        -abs(partial - baseline.partial_ratio_threshold),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_config = config
                        best_metrics = metrics
                        best_events = events

    if best_config is None or best_metrics is None:
        raise ValidationDataError("Не удалось подобрать допустимую конфигурацию.")

    matched, _, _ = match_bilateral_events(
        best_events,
        truth,
        tolerance_sec=tolerance_sec,
    )
    full_labels = [item for item in matched if item.label.kind is ValidationLabelKind.FULL_BLINK]
    incomplete_labels = [
        item for item in matched if item.label.kind is ValidationLabelKind.INCOMPLETE_BLINK
    ]
    classification_validated = len(full_labels) >= 2 and len(incomplete_labels) >= 2
    balanced_accuracy: float | None = None

    if classification_validated:
        best_full_threshold = best_config.full_blink_ratio_threshold
        best_classification_score: tuple[float, float, float] | None = None
        upper = best_config.partial_ratio_threshold - 0.01
        for threshold in _float_range(0.40, upper, 0.01):
            full_correct = sum(
                item.event.bilateral_depth <= threshold for item in full_labels
            )
            incomplete_correct = sum(
                item.event.bilateral_depth > threshold for item in incomplete_labels
            )
            full_recall = full_correct / len(full_labels)
            incomplete_recall = incomplete_correct / len(incomplete_labels)
            current_balanced = (full_recall + incomplete_recall) / 2.0
            current_accuracy = (
                full_correct + incomplete_correct
            ) / (len(full_labels) + len(incomplete_labels))
            score = (
                current_balanced,
                current_accuracy,
                -abs(threshold - baseline.full_blink_ratio_threshold),
            )
            if best_classification_score is None or score > best_classification_score:
                best_classification_score = score
                best_full_threshold = threshold
                balanced_accuracy = current_balanced

        best_config = OfflineBlinkConfig(
            full_blink_ratio_threshold=best_full_threshold,
            partial_ratio_threshold=best_config.partial_ratio_threshold,
            open_ratio_threshold=best_config.open_ratio_threshold,
            smoothing_alpha=best_config.smoothing_alpha,
            min_blink_sec=best_config.min_blink_sec,
            max_blink_sec=best_config.max_blink_sec,
            bilateral_sync_tolerance_sec=best_config.bilateral_sync_tolerance_sec,
            max_sample_gap_sec=best_config.max_sample_gap_sec,
            candidate_gap_tolerance_sec=best_config.candidate_gap_tolerance_sec,
            reopen_confirmation_samples=best_config.reopen_confirmation_samples,
        )
        best_events = detect_offline_blinks(
            samples,
            best_config,
            start_after_sec=calibration_sec,
        )
        best_metrics = calculate_validation_metrics(
            best_events,
            truth,
            tolerance_sec=tolerance_sec,
        )

    return ThresholdSearchResult(
        config=best_config,
        metrics=best_metrics,
        classification_threshold_validated=classification_validated,
        classification_balanced_accuracy=balanced_accuracy,
        evaluated_config_count=evaluated_count,
    )


def calculate_time_weighted_coverage(
    samples: Sequence[NormalizedValidationSample],
    *,
    start_after_sec: float = 0.0,
    max_interval_sec: float = 0.20,
) -> float:
    """Доля времени с пригодным двусторонним сигналом.

    Интервалы крупнее max_interval_sec не приписываются предыдущему кадру:
    это защищает оценку от зависшей камеры или пропуска кадров.
    """
    if max_interval_sec <= 0.0:
        raise ValueError("max_interval_sec должен быть больше нуля.")
    valid_time = 0.0
    total_time = 0.0
    previous: NormalizedValidationSample | None = None
    for sample in samples:
        if sample.timestamp_sec < start_after_sec:
            previous = sample
            continue
        if previous is None:
            previous = sample
            continue
        interval_start = max(previous.timestamp_sec, start_after_sec)
        duration = sample.timestamp_sec - interval_start
        if 0.0 < duration <= max_interval_sec:
            total_time += duration
            if previous.valid:
                valid_time += duration
        previous = sample
    return valid_time / total_time if total_time > 0.0 else 0.0


def save_normalized_samples(
    path: str | Path,
    samples: Sequence[NormalizedValidationSample],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(asdict(samples[0]).keys()) if samples else (
        "frame_index",
        "timestamp_sec",
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = asdict(sample)
            row["face_detected"] = int(sample.face_detected)
            row["valid"] = int(sample.valid)
            writer.writerow(row)
    temporary.replace(destination)


def save_events(path: str | Path, events: Sequence[OfflineBlinkEvent]) -> None:
    payload = {
        "schema_version": 1,
        "events": [
            {
                **asdict(event),
                "event_type": event.event_type.value,
            }
            for event in events
        ],
    }
    _atomic_json_write(Path(path), payload)


def build_report_payload(
    *,
    reference: ReferenceEstimate,
    default_config: OfflineBlinkConfig,
    default_metrics: ValidationMetrics,
    optimized: ThresholdSearchResult,
    labels: Sequence[ManualBlinkLabel],
    samples: Sequence[NormalizedValidationSample],
    calibration_sec: float,
) -> dict[str, object]:
    total_duration = samples[-1].timestamp_sec if samples else 0.0
    signal_coverage = calculate_time_weighted_coverage(
        samples,
        start_after_sec=calibration_sec,
    )
    processed_fps = _sample_rate(samples)
    return {
        "schema_version": 1,
        "session": {
            "duration_sec": total_duration,
            "calibration_sec": calibration_sec,
            "sample_count": len(samples),
            "processed_fps": processed_fps,
            "signal_coverage": signal_coverage,
            "manual_label_count": len(labels),
            "manual_bilateral_label_count": sum(label.kind.is_bilateral for label in labels),
        },
        "reference": asdict(reference),
        "default": {
            "config": asdict(default_config),
            "metrics": asdict(default_metrics),
        },
        "optimized": {
            "config": asdict(optimized.config),
            "metrics": asdict(optimized.metrics),
            "classification_threshold_validated": optimized.classification_threshold_validated,
            "classification_balanced_accuracy": optimized.classification_balanced_accuracy,
            "evaluated_config_count": optimized.evaluated_config_count,
        },
        "limitations": [
            "Подобранные пороги относятся только к этой диагностической сессии.",
            "Одна сессия не доказывает точность на других людях, камерах и освещении.",
            "Порог полного/неполного моргания считается проверенным только при наличии минимум двух размеченных событий каждого класса.",
            "Видео используется только для ручной разметки и может быть удалено после отчёта.",
        ],
    }


def save_report_files(
    session_dir: str | Path,
    payload: dict[str, object],
) -> None:
    directory = Path(session_dir)
    _atomic_json_write(directory / "report.json", payload)
    (directory / "report.txt").write_text(
        _format_report_text(payload),
        encoding="utf-8",
    )


def render_validation_plot(
    path: str | Path,
    samples: Sequence[NormalizedValidationSample],
    labels: Sequence[ManualBlinkLabel],
    events: Sequence[OfflineBlinkEvent],
    config: OfflineBlinkConfig,
    *,
    calibration_sec: float,
) -> None:
    if not samples:
        raise ValidationDataError("Нельзя построить график без сигналов.")
    width, height = 1600, 820
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 60, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    image = np.full((height, width, 3), 248, dtype=np.uint8)

    minimum_y, maximum_y = 0.20, 1.30
    maximum_time = max(samples[-1].timestamp_sec, 1.0)

    def x_for(timestamp: float) -> int:
        return margin_left + int(plot_width * timestamp / maximum_time)

    def y_for(value: float) -> int:
        clipped = max(minimum_y, min(maximum_y, value))
        fraction = (clipped - minimum_y) / (maximum_y - minimum_y)
        return margin_top + plot_height - int(plot_height * fraction)

    for index in range(0, 12):
        value = minimum_y + (maximum_y - minimum_y) * index / 11.0
        y = y_for(value)
        cv2.line(image, (margin_left, y), (width - margin_right, y), (220, 220, 220), 1)
        cv2.putText(
            image,
            f"{value:.2f}",
            (15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    for threshold, label, line_color in (
        (config.full_blink_ratio_threshold, "full", (170, 70, 70)),
        (config.partial_ratio_threshold, "partial", (80, 120, 190)),
        (config.open_ratio_threshold, "open", (80, 150, 80)),
    ):
        y = y_for(threshold)
        cv2.line(image, (margin_left, y), (width - margin_right, y), line_color, 2)
        cv2.putText(
            image,
            f"{label} {threshold:.2f}",
            (width - 220, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            line_color,
            1,
            cv2.LINE_AA,
        )

    calibration_x = x_for(calibration_sec)
    cv2.line(
        image,
        (calibration_x, margin_top),
        (calibration_x, margin_top + plot_height),
        (120, 120, 120),
        2,
    )
    cv2.putText(
        image,
        "calibration end",
        (max(margin_left, calibration_x - 70), margin_top - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    _draw_signal_line(image, samples, x_for, y_for, side="left", color=(30, 110, 220))
    _draw_signal_line(image, samples, x_for, y_for, side="right", color=(40, 170, 70))

    for label in labels:
        if not label.kind.is_bilateral:
            continue
        x = x_for(label.timestamp_sec)
        cv2.line(
            image,
            (x, margin_top),
            (x, margin_top + plot_height),
            (180, 40, 180),
            1,
        )
    for event in events:
        if event.event_type not in {
            OcularEventType.BILATERAL_BLINK,
            OcularEventType.INCOMPLETE_BILATERAL_BLINK,
        }:
            continue
        x = x_for(event.center_at)
        cv2.circle(image, (x, y_for(event.bilateral_depth)), 5, (20, 20, 20), -1)

    cv2.rectangle(
        image,
        (margin_left, margin_top),
        (margin_left + plot_width, margin_top + plot_height),
        (70, 70, 70),
        1,
    )
    cv2.putText(
        image,
        "Normalized eye opening: left=blue, right=green; purple=true labels; black=detected",
        (margin_left, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "time, sec",
        (width // 2 - 40, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Не удалось сохранить график: {destination}")


def _finalize_offline_candidate(
    candidate: _OfflineCandidate,
    ended_at: float,
    config: OfflineBlinkConfig,
) -> OfflineBlinkEvent | None:
    duration = max(0.0, ended_at - candidate.started_at)
    sync_offset = abs(candidate.min_left_at - candidate.min_right_at)
    both_dipped = (
        candidate.min_left <= config.partial_ratio_threshold
        and candidate.min_right <= config.partial_ratio_threshold
    )
    left_wink = (
        candidate.min_left <= config.full_blink_ratio_threshold
        and candidate.min_right > config.partial_ratio_threshold
    )
    right_wink = (
        candidate.min_right <= config.full_blink_ratio_threshold
        and candidate.min_left > config.partial_ratio_threshold
    )
    if not config.min_blink_sec <= duration <= config.max_blink_sec:
        return None

    if both_dipped and sync_offset <= config.bilateral_sync_tolerance_sec:
        event_type = (
            OcularEventType.BILATERAL_BLINK
            if candidate.min_left <= config.full_blink_ratio_threshold
            and candidate.min_right <= config.full_blink_ratio_threshold
            else OcularEventType.INCOMPLETE_BILATERAL_BLINK
        )
        center_at = (candidate.min_left_at + candidate.min_right_at) / 2.0
    elif left_wink:
        event_type = OcularEventType.LEFT_WINK
        center_at = candidate.min_left_at
    elif right_wink:
        event_type = OcularEventType.RIGHT_WINK
        center_at = candidate.min_right_at
    else:
        return None

    return OfflineBlinkEvent(
        started_at=candidate.started_at,
        ended_at=ended_at,
        center_at=center_at,
        duration_sec=duration,
        event_type=event_type,
        min_left_normalized=candidate.min_left,
        min_right_normalized=candidate.min_right,
        min_left_at=candidate.min_left_at,
        min_right_at=candidate.min_right_at,
        sync_offset_sec=sync_offset,
    )


def _draw_signal_line(image, samples, x_for, y_for, *, side: str, color) -> None:
    previous: tuple[int, int] | None = None
    previous_time: float | None = None
    for sample in samples:
        value = (
            sample.left_eye_opening_normalized
            if side == "left"
            else sample.right_eye_opening_normalized
        )
        if not sample.valid or value is None:
            previous = None
            previous_time = None
            continue
        current = (x_for(sample.timestamp_sec), y_for(float(value)))
        if (
            previous is not None
            and previous_time is not None
            and sample.timestamp_sec - previous_time <= 0.20
        ):
            cv2.line(image, previous, current, color, 2, cv2.LINE_AA)
        previous = current
        previous_time = sample.timestamp_sec


def _format_report_text(payload: dict[str, object]) -> str:
    session = payload["session"]
    reference = payload["reference"]
    default = payload["default"]
    optimized = payload["optimized"]
    assert isinstance(session, dict)
    assert isinstance(reference, dict)
    assert isinstance(default, dict)
    assert isinstance(optimized, dict)
    default_metrics = default["metrics"]
    optimized_metrics = optimized["metrics"]
    optimized_config = optimized["config"]
    assert isinstance(default_metrics, dict)
    assert isinstance(optimized_metrics, dict)
    assert isinstance(optimized_config, dict)

    def metric(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    lines = [
        "OCULAR VALIDATION REPORT",
        "=" * 72,
        f"Duration: {float(session['duration_sec']):.2f} sec",
        f"Processed FPS: {metric(session['processed_fps'])}",
        f"Signal coverage: {metric(session['signal_coverage'])}",
        f"Manual bilateral labels: {session['manual_bilateral_label_count']}",
        "",
        "OPEN-EYE REFERENCE",
        f"Left: {float(reference['left_open_eye_reference']):.6f}",
        f"Right: {float(reference['right_open_eye_reference']):.6f}",
        f"Left relative MAD: {float(reference['left_relative_mad']):.4f}",
        f"Right relative MAD: {float(reference['right_relative_mad']):.4f}",
        "",
        "DEFAULT CONFIG",
        f"Precision: {metric(default_metrics['precision'])}",
        f"Recall: {metric(default_metrics['recall'])}",
        f"F1: {metric(default_metrics['f1'])}",
        f"False positives: {default_metrics['false_positive_count']}",
        f"Missed: {default_metrics['false_negative_count']}",
        "",
        "SESSION-SPECIFIC OPTIMIZED CONFIG",
        f"full_blink_ratio_threshold={float(optimized_config['full_blink_ratio_threshold']):.2f}",
        f"partial_ratio_threshold={float(optimized_config['partial_ratio_threshold']):.2f}",
        f"open_ratio_threshold={float(optimized_config['open_ratio_threshold']):.2f}",
        f"smoothing_alpha={float(optimized_config['smoothing_alpha']):.2f}",
        f"bilateral_sync_tolerance_sec={float(optimized_config['bilateral_sync_tolerance_sec']):.2f}",
        f"Precision: {metric(optimized_metrics['precision'])}",
        f"Recall: {metric(optimized_metrics['recall'])}",
        f"F1: {metric(optimized_metrics['f1'])}",
        f"False positives: {optimized_metrics['false_positive_count']}",
        f"Missed: {optimized_metrics['false_negative_count']}",
        f"Full/incomplete threshold validated: {optimized['classification_threshold_validated']}",
        "",
        "LIMITATIONS",
    ]
    limitations = payload.get("limitations", [])
    if isinstance(limitations, list):
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def _validate_sample_sequence(samples: Sequence[ValidationSample]) -> None:
    previous_frame = -1
    previous_time = -math.inf
    for sample in samples:
        if sample.frame_index <= previous_frame:
            raise ValidationDataError("frame_index должен строго возрастать.")
        if sample.timestamp_sec < previous_time:
            raise ValidationDataError("timestamp_sec не должен уменьшаться.")
        if not math.isfinite(sample.timestamp_sec) or sample.timestamp_sec < 0.0:
            raise ValidationDataError("timestamp_sec должен быть конечным и неотрицательным.")
        previous_frame = sample.frame_index
        previous_time = sample.timestamp_sec


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() in {"", "None", "null"}:
        return None
    return float(value)


def _parse_bool(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"Неизвестное логическое значение: {value!r}")


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _median_absolute_deviation(values: Sequence[float], center: float) -> float:
    if not values:
        return 0.0
    return float(statistics.median(abs(value - center) for value in values))


def _ema(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step должен быть больше нуля.")
    values: list[float] = []
    current = start
    while current <= stop + step / 10.0:
        values.append(round(current, 6))
        current += step
    return values


def _sample_rate(samples: Sequence[NormalizedValidationSample]) -> float | None:
    if len(samples) < 2:
        return None
    elapsed = samples[-1].timestamp_sec - samples[0].timestamp_sec
    if elapsed <= 0.0:
        return None
    return (len(samples) - 1) / elapsed
