from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Mapping

from app_core.baseline_engine import BaselineComparison, WorkContext
from app_core.session_monitor import SessionSnapshot
from app_core.state_machine import UserState
from app_core.typing_baseline import TypingBaselineService


class RecommendationKind(StrEnum):
    NONE = "none"
    EYE_REST = "eye_rest"
    MICROBREAK = "microbreak"
    RECOVERY_BREAK = "recovery_break"


class EvidenceSource(StrEnum):
    SESSION = "session"
    TYPING = "typing"
    OCULAR = "ocular"
    SELF_REPORT = "self_report"


class WorkloadLevel(IntEnum):
    """Пять пользовательских уровней наблюдаемой рабочей нагрузки.

    Это не проценты усталости и не медицинские категории. Каждый уровень
    существует потому, что меняет поведение ассистента: от наблюдения до
    приоритетного восстановления.
    """

    STABLE = 1
    EARLY = 2
    SUSTAINED = 3
    EXPRESSED = 4
    RECOVERY_PRIORITY = 5


class AssessmentReliability(StrEnum):
    LIMITED = "limited"
    MODERATE = "moderate"
    HIGH = "high"


WORKLOAD_TITLES: dict[WorkloadLevel, str] = {
    WorkloadLevel.STABLE: "Рабочее состояние устойчиво",
    WorkloadLevel.EARLY: "Появляются признаки нагрузки",
    WorkloadLevel.SUSTAINED: "Нагрузка становится устойчивой",
    WorkloadLevel.EXPRESSED: "Нагрузка выражена",
    WorkloadLevel.RECOVERY_PRIORITY: "Восстановление приоритетно",
}

WORKLOAD_SUMMARIES: dict[WorkloadLevel, str] = {
    WorkloadLevel.STABLE: (
        "Доступные показатели находятся в вашем обычном рабочем диапазоне."
    ),
    WorkloadLevel.EARLY: (
        "Появился ранний сигнал нагрузки; система проверяет его устойчивость и контекст."
    ),
    WorkloadLevel.SUSTAINED: (
        "Изменения сохраняются во времени или подтверждаются несколькими показателями."
    ),
    WorkloadLevel.EXPRESSED: (
        "Несколько признаков указывают на выраженное накопление рабочей нагрузки."
    ),
    WorkloadLevel.RECOVERY_PRIORITY: (
        "Текущая совокупность признаков делает восстановительный перерыв приоритетным."
    ),
}

RELIABILITY_LABELS: dict[AssessmentReliability, str] = {
    AssessmentReliability.LIMITED: "Оценка ограничена",
    AssessmentReliability.MODERATE: "Надёжность оценки средняя",
    AssessmentReliability.HIGH: "Надёжность оценки высокая",
}


@dataclass(frozen=True, slots=True)
class TypingDeviation:
    feature_name: str
    actual_value: float | None
    baseline_median: float | None
    robust_z: float | None
    ready: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class RecoverySignals:
    captured_at: float
    state: UserState
    fatigue_analysis_allowed: bool

    typing_data_ready: bool
    typing_baseline_ready: bool
    typing_calibration_progress: float
    typing_deviations: Mapping[str, TypingDeviation]

    ocular_calibrated: bool
    ocular_data_ready: bool
    ocular_quality: float
    ocular_signal_coverage: float
    ocular_blink_rate_per_min: float | None
    ocular_perclos: float | None
    ocular_long_closure_count: int
    ocular_severe_closure_detected: bool


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    code: str
    source: EvidenceSource
    severity: int
    title: str
    detail: str
    decision_ready: bool

    def __post_init__(self) -> None:
        if self.severity not in {1, 2, 3}:
            raise ValueError("severity должен быть 1, 2 или 3.")


@dataclass(frozen=True, slots=True)
class RecoveryRecommendation:
    recommendation_id: int
    kind: RecommendationKind
    title: str
    message: str
    suggested_break_sec: float
    reason_codes: tuple[str, ...]
    created_at: float


@dataclass(frozen=True, slots=True)
class RecoveryAssessment:
    captured_at: float
    state: UserState
    status_label: str
    continuous_work_sec: float
    current_break_sec: float
    typing_calibration_progress: float
    evidence: tuple[RecoveryEvidence, ...]
    recommendation: RecoveryRecommendation | None
    recommendation_is_new: bool
    decision_basis: str
    ocular_used_for_decision: bool

    workload_level: WorkloadLevel = WorkloadLevel.STABLE
    workload_title: str = WORKLOAD_TITLES[WorkloadLevel.STABLE]
    workload_summary: str = WORKLOAD_SUMMARIES[WorkloadLevel.STABLE]
    reliability: AssessmentReliability = AssessmentReliability.LIMITED
    reliability_label: str = RELIABILITY_LABELS[AssessmentReliability.LIMITED]
    reliability_detail: str = "Доступен только ограниченный набор независимых каналов."
    primary_reasons: tuple[str, ...] = ()
    contributing_sources: tuple[EvidenceSource, ...] = ()
    self_report_fatigue_sp: int | None = None
    self_report_sleepiness_kss: int | None = None


class AdaptiveRecoveryEngine:
    """Выбирает восстановительное действие по наблюдаемым сигналам.

    Движок не вычисляет «процент усталости». Он объединяет независимые
    источники, учитывает их устойчивость и качество, а затем переводит это
    в один из пяти функциональных уровней рабочей нагрузки.

    Глазной канал по умолчанию остаётся диагностическим и не участвует в
    решениях до завершения отдельной размеченной валидации.
    """

    TYPING_FEATURES = (
        "typing_keys_per_minute",
        "typing_rhythm_cv",
        "typing_correction_ratio",
        "typing_long_pause_rate_per_min",
    )

    def __init__(
        self,
        *,
        eye_rest_after_sec: float = 45.0 * 60.0,
        microbreak_after_sec: float = 70.0 * 60.0,
        recovery_break_after_sec: float = 105.0 * 60.0,
        meaningful_break_sec: float = 180.0,
        alert_cooldown_sec: float = 25.0 * 60.0,
        typing_evaluation_interval_sec: float = 10.0,
        typing_persistence_samples: int = 3,
        typing_required_positive_samples: int = 2,
        typing_z_threshold: float = 1.5,
        ocular_decisions_enabled: bool = False,
        level_rise_confirmations: int = 3,
        level_drop_confirmations: int = 10,
    ) -> None:
        ordered = (eye_rest_after_sec, microbreak_after_sec, recovery_break_after_sec)
        if not (0 < ordered[0] < ordered[1] < ordered[2]):
            raise ValueError(
                "Пороги длительности должны идти: eye_rest < microbreak < recovery."
            )
        if meaningful_break_sec <= 0 or alert_cooldown_sec < 0:
            raise ValueError("Параметры перерыва заданы неверно.")
        if typing_evaluation_interval_sec <= 0:
            raise ValueError("Интервал проверки печати должен быть положительным.")
        if typing_persistence_samples < 1:
            raise ValueError("typing_persistence_samples должен быть не меньше 1.")
        if not 1 <= typing_required_positive_samples <= typing_persistence_samples:
            raise ValueError("Некорректное число подтверждающих окон печати.")
        if typing_z_threshold <= 0:
            raise ValueError("typing_z_threshold должен быть положительным.")
        if level_rise_confirmations < 1 or level_drop_confirmations < 1:
            raise ValueError("Подтверждения уровней должны быть положительными.")

        self.eye_rest_after_sec = float(eye_rest_after_sec)
        self.microbreak_after_sec = float(microbreak_after_sec)
        self.recovery_break_after_sec = float(recovery_break_after_sec)
        self.meaningful_break_sec = float(meaningful_break_sec)
        self.alert_cooldown_sec = float(alert_cooldown_sec)
        self.typing_evaluation_interval_sec = float(typing_evaluation_interval_sec)
        self.typing_persistence_samples = int(typing_persistence_samples)
        self.typing_required_positive_samples = int(typing_required_positive_samples)
        self.typing_z_threshold = float(typing_z_threshold)
        self.ocular_decisions_enabled = bool(ocular_decisions_enabled)
        self.level_rise_confirmations = int(level_rise_confirmations)
        self.level_drop_confirmations = int(level_drop_confirmations)

        self._last_update_at: float | None = None
        self._previous_state = UserState.UNKNOWN
        self._continuous_work_sec = 0.0
        self._current_break_sec = 0.0
        self._last_typing_evaluation_at: float | None = None
        self._typing_history: deque[frozenset[str]] = deque(
            maxlen=self.typing_persistence_samples
        )

        self._current_recommendation: RecoveryRecommendation | None = None
        self._cooldown_until = 0.0
        self._snoozed_until = 0.0
        self._next_recommendation_id = 1

        self._self_report_fatigue_sp: int | None = None
        self._self_report_sleepiness_kss: int | None = None
        self._self_report_at: float | None = None

        self._stable_workload_level = WorkloadLevel.STABLE
        self._pending_workload_level: WorkloadLevel | None = None
        self._pending_workload_count = 0
        self._last_raw_workload_level = WorkloadLevel.STABLE

    @property
    def continuous_work_sec(self) -> float:
        return self._continuous_work_sec

    @property
    def current_break_sec(self) -> float:
        return self._current_break_sec

    @property
    def current_recommendation(self) -> RecoveryRecommendation | None:
        return self._current_recommendation

    @property
    def workload_level(self) -> WorkloadLevel:
        return self._stable_workload_level

    def record_self_report(
        self,
        *,
        fatigue_sp_1_7: float,
        sleepiness_kss_1_9: float,
        captured_at: float,
    ) -> None:
        fatigue = float(fatigue_sp_1_7)
        sleepiness = float(sleepiness_kss_1_9)
        timestamp = float(captured_at)
        if not fatigue.is_integer() or not 1.0 <= fatigue <= 7.0:
            raise ValueError("fatigue_sp_1_7 должен быть целым числом 1..7.")
        if not sleepiness.is_integer() or not 1.0 <= sleepiness <= 9.0:
            raise ValueError("sleepiness_kss_1_9 должен быть целым числом 1..9.")
        if not math.isfinite(timestamp):
            raise ValueError("captured_at должен быть конечным числом.")
        self._self_report_fatigue_sp = int(fatigue)
        self._self_report_sleepiness_kss = int(sleepiness)
        self._self_report_at = timestamp

    def snooze(self, *, minutes: float, captured_at: float) -> None:
        if minutes <= 0:
            raise ValueError("Время отсрочки должно быть положительным.")
        self._snoozed_until = max(
            self._snoozed_until,
            float(captured_at) + float(minutes) * 60.0,
        )
        self._current_recommendation = None

    def resolve_recommendation(
        self,
        *,
        captured_at: float,
        longer_cooldown: bool = False,
    ) -> RecoveryRecommendation | None:
        recommendation = self._current_recommendation
        self._current_recommendation = None
        if recommendation is not None and longer_cooldown:
            self._cooldown_until = max(
                self._cooldown_until,
                float(captured_at) + 60.0 * 60.0,
            )
        return recommendation

    def diagnostics(self, *, captured_at: float) -> dict[str, object]:
        now = float(captured_at)
        report_age = None
        if self._self_report_at is not None:
            report_age = max(0.0, now - self._self_report_at)
        return {
            "stable_workload_level": int(self._stable_workload_level),
            "raw_workload_level": int(self._last_raw_workload_level),
            "pending_workload_level": (
                int(self._pending_workload_level)
                if self._pending_workload_level is not None
                else None
            ),
            "pending_workload_count": self._pending_workload_count,
            "continuous_work_sec": self._continuous_work_sec,
            "current_break_sec": self._current_break_sec,
            "cooldown_remaining_sec": max(0.0, self._cooldown_until - now),
            "snooze_remaining_sec": max(0.0, self._snoozed_until - now),
            "self_report_age_sec": report_age,
            "fatigue_sp_1_7": self._self_report_fatigue_sp,
            "sleepiness_kss_1_9": self._self_report_sleepiness_kss,
            "typing_history_samples": len(self._typing_history),
            "typing_persistent_codes": sorted(self._persistent_typing_codes()),
            "ocular_decisions_enabled": self.ocular_decisions_enabled,
        }

    def update(self, signals: RecoverySignals) -> RecoveryAssessment:
        now = float(signals.captured_at)
        if not math.isfinite(now):
            raise ValueError("captured_at должен быть конечным числом.")

        dt = self._elapsed_since_last_update(now)
        self._update_session_clock(signals.state, signals.fatigue_analysis_allowed, dt)
        self._update_typing_history(signals, now)

        evidence = self._build_evidence(signals, now)
        raw_level = self._derive_workload_level(signals, evidence)
        self._last_raw_workload_level = raw_level
        urgent = any(
            item.decision_ready
            and item.code in {"self_report_very_high", "severe_eye_closure"}
            for item in evidence
        )
        stable_level = self._stabilize_workload_level(
            signals=signals,
            target=raw_level,
            urgent=urgent,
        )

        recommendation, is_new = self._select_recommendation(
            signals=signals,
            evidence=evidence,
            stable_level=stable_level,
            now=now,
        )

        reliability, reliability_detail = self._assess_reliability(signals, now)
        primary_reasons = self._primary_reasons(signals, evidence)
        contributing_sources = tuple(
            sorted(
                {item.source for item in evidence if item.decision_ready},
                key=lambda item: item.value,
            )
        )
        status_label = self._status_label(signals, stable_level, recommendation)
        decision_basis = self._decision_basis(signals, evidence)

        self._previous_state = signals.state
        return RecoveryAssessment(
            captured_at=now,
            state=signals.state,
            status_label=status_label,
            continuous_work_sec=self._continuous_work_sec,
            current_break_sec=self._current_break_sec,
            typing_calibration_progress=signals.typing_calibration_progress,
            evidence=tuple(evidence),
            recommendation=recommendation,
            recommendation_is_new=is_new,
            decision_basis=decision_basis,
            ocular_used_for_decision=(
                self.ocular_decisions_enabled
                and any(
                    item.source is EvidenceSource.OCULAR and item.decision_ready
                    for item in evidence
                )
            ),
            workload_level=stable_level,
            workload_title=WORKLOAD_TITLES[stable_level],
            workload_summary=WORKLOAD_SUMMARIES[stable_level],
            reliability=reliability,
            reliability_label=RELIABILITY_LABELS[reliability],
            reliability_detail=reliability_detail,
            primary_reasons=primary_reasons,
            contributing_sources=contributing_sources,
            self_report_fatigue_sp=(
                self._self_report_fatigue_sp if self._recent_self_report(now) else None
            ),
            self_report_sleepiness_kss=(
                self._self_report_sleepiness_kss if self._recent_self_report(now) else None
            ),
        )

    def _elapsed_since_last_update(self, now: float) -> float:
        if self._last_update_at is None:
            self._last_update_at = now
            return 0.0
        if now < self._last_update_at:
            raise ValueError("Время обновлений должно возрастать.")
        dt = min(5.0, now - self._last_update_at)
        self._last_update_at = now
        return dt

    def _update_session_clock(
        self,
        state: UserState,
        fatigue_analysis_allowed: bool,
        dt: float,
    ) -> None:
        working = (
            fatigue_analysis_allowed
            and state in {UserState.ACTIVE_WORK, UserState.PASSIVE_WORK}
        )
        resting = state in {UserState.AWAY, UserState.BREAK}

        if working:
            if self._current_break_sec >= self.meaningful_break_sec:
                self._continuous_work_sec = 0.0
                self._typing_history.clear()
            self._current_break_sec = 0.0
            self._continuous_work_sec += dt
            return

        if resting:
            self._current_break_sec += dt
            if self._current_break_sec >= self.meaningful_break_sec:
                self._continuous_work_sec = 0.0
                self._current_recommendation = None
            return

    def _update_typing_history(self, signals: RecoverySignals, now: float) -> None:
        if not signals.typing_data_ready or not signals.typing_baseline_ready:
            return
        if (
            self._last_typing_evaluation_at is not None
            and now - self._last_typing_evaluation_at < self.typing_evaluation_interval_sec
        ):
            return
        self._last_typing_evaluation_at = now

        abnormal: set[str] = set()
        for feature_name, deviation in signals.typing_deviations.items():
            if not deviation.ready or deviation.robust_z is None:
                continue
            z = deviation.robust_z
            if feature_name == "typing_keys_per_minute":
                if z <= -self.typing_z_threshold:
                    abnormal.add("typing_slowdown")
            elif feature_name == "typing_rhythm_cv":
                if z >= self.typing_z_threshold:
                    abnormal.add("typing_rhythm_instability")
            elif feature_name == "typing_correction_ratio":
                if z >= self.typing_z_threshold:
                    abnormal.add("typing_more_corrections")
            elif feature_name == "typing_long_pause_rate_per_min":
                if z >= self.typing_z_threshold:
                    abnormal.add("typing_more_pauses")
        self._typing_history.append(frozenset(abnormal))

    def _persistent_typing_codes(self) -> set[str]:
        if len(self._typing_history) < self.typing_persistence_samples:
            return set()
        counts: Counter[str] = Counter()
        for sample in self._typing_history:
            counts.update(sample)
        return {
            code
            for code, count in counts.items()
            if count >= self.typing_required_positive_samples
        }

    def _build_evidence(
        self,
        signals: RecoverySignals,
        now: float,
    ) -> list[RecoveryEvidence]:
        evidence: list[RecoveryEvidence] = []

        if self._continuous_work_sec >= self.recovery_break_after_sec:
            evidence.append(
                RecoveryEvidence(
                    code="very_long_continuous_work",
                    source=EvidenceSource.SESSION,
                    severity=3,
                    title="Очень длинная непрерывная рабочая сессия",
                    detail="Полноценного перерыва не было дольше верхнего рабочего порога.",
                    decision_ready=True,
                )
            )
        elif self._continuous_work_sec >= self.microbreak_after_sec:
            evidence.append(
                RecoveryEvidence(
                    code="long_continuous_work",
                    source=EvidenceSource.SESSION,
                    severity=2,
                    title="Непрерывная работа продолжается длительное время",
                    detail="Короткие остановки не сформировали полноценный перерыв.",
                    decision_ready=True,
                )
            )
        elif self._continuous_work_sec >= self.eye_rest_after_sec:
            evidence.append(
                RecoveryEvidence(
                    code="screen_session_accumulating",
                    source=EvidenceSource.SESSION,
                    severity=1,
                    title="Продолжительность рабочей сессии выросла",
                    detail="Это временной сигнал нагрузки, а не вывод о физиологической усталости.",
                    decision_ready=True,
                )
            )

        typing_titles = {
            "typing_slowdown": (
                "Темп печати устойчиво снизился",
                "Снижение сравнивается только с вашей личной нормой.",
            ),
            "typing_rhythm_instability": (
                "Ритм печати стал менее устойчивым",
                "Изменение сохранялось в нескольких независимых проверках.",
            ),
            "typing_more_corrections": (
                "Доля исправлений выросла",
                "Система видит категорию исправления, но не введённый текст.",
            ),
            "typing_more_pauses": (
                "В печати стало больше длинных пауз",
                "Сигнал используется только вместе с рабочим контекстом.",
            ),
        }
        for code in sorted(self._persistent_typing_codes()):
            title, detail = typing_titles[code]
            evidence.append(
                RecoveryEvidence(
                    code=code,
                    source=EvidenceSource.TYPING,
                    severity=2,
                    title=title,
                    detail=detail,
                    decision_ready=True,
                )
            )

        if self._recent_self_report(now):
            fatigue = self._self_report_fatigue_sp
            sleepiness = self._self_report_sleepiness_kss
            assert fatigue is not None and sleepiness is not None
            severity = max(
                self._fatigue_sp_severity(fatigue),
                self._sleepiness_kss_severity(sleepiness),
            )
            if severity > 0:
                code = {
                    1: "self_report_early",
                    2: "self_report_moderate",
                    3: "self_report_strong",
                }[severity]
                if fatigue == 7 or sleepiness == 9:
                    code = "self_report_very_high"
                evidence.append(
                    RecoveryEvidence(
                        code=code,
                        source=EvidenceSource.SELF_REPORT,
                        severity=severity,
                        title="Самооценка указывает на изменение состояния",
                        detail=(
                            f"Усталость по Samn–Perelli: {fatigue}/7; "
                            f"сонливость по KSS: {sleepiness}/9."
                        ),
                        decision_ready=True,
                    )
                )

        if signals.ocular_calibrated and signals.ocular_data_ready:
            if self.ocular_decisions_enabled:
                if signals.ocular_severe_closure_detected:
                    evidence.append(
                        RecoveryEvidence(
                            code="severe_eye_closure",
                            source=EvidenceSource.OCULAR,
                            severity=3,
                            title="Зафиксировано длительное двустороннее закрытие глаз",
                            detail="Событие требует приоритетного восстановления.",
                            decision_ready=True,
                        )
                    )
            else:
                evidence.append(
                    RecoveryEvidence(
                        code="ocular_diagnostic_available",
                        source=EvidenceSource.OCULAR,
                        severity=1,
                        title="Глазной сигнал доступен для диагностики",
                        detail=(
                            "Он отображается в технической панели, но пока не влияет "
                            "на итоговый уровень и рекомендации."
                        ),
                        decision_ready=False,
                    )
                )

        return evidence

    @staticmethod
    def _fatigue_sp_severity(value: int) -> int:
        if value <= 3:
            return 0
        if value == 4:
            return 1
        if value == 5:
            return 2
        return 3

    @staticmethod
    def _sleepiness_kss_severity(value: int) -> int:
        if value <= 5:
            return 0
        if value == 6:
            return 1
        if value == 7:
            return 2
        return 3

    def _derive_workload_level(
        self,
        signals: RecoverySignals,
        evidence: list[RecoveryEvidence],
    ) -> WorkloadLevel:
        ready = [item for item in evidence if item.decision_ready]
        ready_codes = {item.code for item in ready}
        sources = {item.source for item in ready}
        max_severity = max((item.severity for item in ready), default=0)
        typing_count = sum(item.source is EvidenceSource.TYPING for item in ready)

        if "self_report_very_high" in ready_codes or "severe_eye_closure" in ready_codes:
            return WorkloadLevel.RECOVERY_PRIORITY

        # Выраженная нагрузка: один сильный субъективный сигнал либо согласование
        # не менее двух независимых каналов, где есть хотя бы умеренный признак.
        if "self_report_strong" in ready_codes:
            return WorkloadLevel.EXPRESSED
        if len(sources) >= 2 and max_severity >= 2:
            return WorkloadLevel.EXPRESSED
        if "very_long_continuous_work" in ready_codes:
            return WorkloadLevel.EXPRESSED

        # Устойчивый уровень: длительная сессия, два устойчивых изменения печати,
        # либо умеренная самооценка, подтверждённая ещё одним каналом.
        if "long_continuous_work" in ready_codes:
            return WorkloadLevel.SUSTAINED
        if typing_count >= 2:
            return WorkloadLevel.SUSTAINED
        if "self_report_moderate" in ready_codes and len(sources) >= 2:
            return WorkloadLevel.SUSTAINED

        if ready:
            return WorkloadLevel.EARLY
        return WorkloadLevel.STABLE

    def _stabilize_workload_level(
        self,
        *,
        signals: RecoverySignals,
        target: WorkloadLevel,
        urgent: bool,
    ) -> WorkloadLevel:
        if signals.state in {UserState.UNKNOWN, UserState.RETURNING}:
            self._pending_workload_level = None
            self._pending_workload_count = 0
            return self._stable_workload_level

        if signals.state in {UserState.AWAY, UserState.BREAK}:
            if self._current_break_sec >= self.meaningful_break_sec:
                self._stable_workload_level = WorkloadLevel.STABLE
                self._pending_workload_level = None
                self._pending_workload_count = 0
            return self._stable_workload_level

        if urgent:
            self._stable_workload_level = target
            self._pending_workload_level = None
            self._pending_workload_count = 0
            return target

        if target == self._stable_workload_level:
            self._pending_workload_level = None
            self._pending_workload_count = 0
            return self._stable_workload_level

        if target != self._pending_workload_level:
            self._pending_workload_level = target
            self._pending_workload_count = 1
        else:
            self._pending_workload_count += 1

        required = (
            self.level_rise_confirmations
            if target > self._stable_workload_level
            else self.level_drop_confirmations
        )
        if self._pending_workload_count >= required:
            self._stable_workload_level = target
            self._pending_workload_level = None
            self._pending_workload_count = 0
        return self._stable_workload_level

    def _select_recommendation(
        self,
        *,
        signals: RecoverySignals,
        evidence: list[RecoveryEvidence],
        stable_level: WorkloadLevel,
        now: float,
    ) -> tuple[RecoveryRecommendation | None, bool]:
        if self._current_recommendation is not None:
            return self._current_recommendation, False
        if not signals.fatigue_analysis_allowed:
            return None, False
        if signals.state not in {UserState.ACTIVE_WORK, UserState.PASSIVE_WORK}:
            return None, False
        if now < self._snoozed_until or now < self._cooldown_until:
            return None, False

        ready_codes = {item.code for item in evidence if item.decision_ready}
        ready_sources = {item.source for item in evidence if item.decision_ready}
        typing_count = sum(
            item.source is EvidenceSource.TYPING and item.decision_ready
            for item in evidence
        )

        kind = RecommendationKind.NONE
        title = ""
        message = ""
        break_sec = 0.0

        if stable_level is WorkloadLevel.RECOVERY_PRIORITY:
            kind = RecommendationKind.RECOVERY_BREAK
            title = "Восстановительный перерыв приоритетен"
            message = (
                "Завершите безопасную текущую операцию и прервите экранную работу примерно на 5 минут."
            )
            break_sec = 5.0 * 60.0
        elif stable_level is WorkloadLevel.EXPRESSED and (
            "self_report_strong" in ready_codes
            or len(ready_sources) >= 2
            or "very_long_continuous_work" in ready_codes
        ):
            kind = RecommendationKind.RECOVERY_BREAK
            title = "Рекомендуется восстановительный перерыв"
            message = (
                "Текущие признаки достаточно выражены. Завершите текущую операцию и сделайте перерыв около 5 минут."
            )
            break_sec = 5.0 * 60.0
        elif stable_level >= WorkloadLevel.SUSTAINED and (
            typing_count >= 2
            or "long_continuous_work" in ready_codes
            or "self_report_moderate" in ready_codes
        ):
            kind = RecommendationKind.MICROBREAK
            title = "Рекомендуется короткий перерыв"
            message = (
                "Изменения стали устойчивыми. После завершения текущего действия прервите работу примерно на 3 минуты."
            )
            break_sec = 3.0 * 60.0
        elif self._continuous_work_sec >= self.eye_rest_after_sec:
            kind = RecommendationKind.EYE_REST
            title = "Рекомендуется зрительная разгрузка"
            message = (
                "Рабочая сессия стала продолжительной. Уберите экран из поля зрения примерно на 90 секунд."
            )
            break_sec = 90.0

        if kind is RecommendationKind.NONE:
            return None, False

        recommendation = RecoveryRecommendation(
            recommendation_id=self._next_recommendation_id,
            kind=kind,
            title=title,
            message=message,
            suggested_break_sec=break_sec,
            reason_codes=tuple(sorted(ready_codes)),
            created_at=now,
        )
        self._next_recommendation_id += 1
        self._current_recommendation = recommendation
        self._cooldown_until = max(self._cooldown_until, now + self.alert_cooldown_sec)
        return recommendation, True

    def _assess_reliability(
        self,
        signals: RecoverySignals,
        now: float,
    ) -> tuple[AssessmentReliability, str]:
        if signals.state in {UserState.UNKNOWN, UserState.RETURNING}:
            return (
                AssessmentReliability.LIMITED,
                "Рабочий контекст пока недостаточно стабилен для уверенной интерпретации.",
            )

        usable_channels = 1  # длительность/контекст рабочей сессии
        details = ["контекст рабочей сессии"]
        if signals.typing_baseline_ready and signals.typing_data_ready:
            usable_channels += 1
            details.append("личная норма печати")
        if (
            signals.ocular_calibrated
            and signals.ocular_data_ready
            and signals.ocular_signal_coverage >= 0.85
            and signals.ocular_quality >= 0.55
        ):
            usable_channels += 1
            details.append("качественный глазной сигнал (пока диагностический)")
        if self._recent_self_report(now):
            usable_channels += 1
            details.append("актуальная самооценка")

        if usable_channels >= 3:
            reliability = AssessmentReliability.HIGH
        elif usable_channels == 2:
            reliability = AssessmentReliability.MODERATE
        else:
            reliability = AssessmentReliability.LIMITED
        return reliability, "Доступны: " + ", ".join(details) + "."

    def _primary_reasons(
        self,
        signals: RecoverySignals,
        evidence: list[RecoveryEvidence],
    ) -> tuple[str, ...]:
        ready = [item for item in evidence if item.decision_ready]
        if ready:
            ordered = sorted(ready, key=lambda item: (-item.severity, item.source.value))
            chosen: list[RecoveryEvidence] = []
            used_sources: set[EvidenceSource] = set()
            for item in ordered:
                if item.source not in used_sources:
                    chosen.append(item)
                    used_sources.add(item.source)
                if len(chosen) == 2:
                    break
            if len(chosen) < 2:
                for item in ordered:
                    if item not in chosen:
                        chosen.append(item)
                    if len(chosen) == 2:
                        break
            return tuple(item.title for item in chosen)

        reasons: list[str] = []
        if self._continuous_work_sec > 0:
            minutes = int(self._continuous_work_sec // 60)
            reasons.append(f"Непрерывная рабочая сессия: {minutes} мин")
        if signals.typing_baseline_ready:
            reasons.append("Устойчивых отклонений ритма работы сейчас нет")
        elif signals.typing_calibration_progress > 0:
            reasons.append(
                f"Личная норма ритма формируется: {signals.typing_calibration_progress:.0%}"
            )
        else:
            reasons.append("Личная норма ритма ещё формируется")
        return tuple(reasons[:2])

    def _status_label(
        self,
        signals: RecoverySignals,
        stable_level: WorkloadLevel,
        recommendation: RecoveryRecommendation | None,
    ) -> str:
        if signals.state in {UserState.AWAY, UserState.BREAK}:
            return "Перерыв"
        if signals.state is UserState.RETURNING:
            return "Возвращение к работе"
        if signals.state is UserState.UNKNOWN:
            return "Оценка временно ограничена"
        if recommendation is not None:
            return "Рекомендация активна"
        return WORKLOAD_TITLES[stable_level]

    def _decision_basis(
        self,
        signals: RecoverySignals,
        evidence: list[RecoveryEvidence],
    ) -> str:
        ready_sources = {item.source for item in evidence if item.decision_ready}
        names = {
            EvidenceSource.SESSION: "длительность рабочей сессии",
            EvidenceSource.TYPING: "личная норма печати",
            EvidenceSource.SELF_REPORT: "самооценка",
            EvidenceSource.OCULAR: "проверенный глазной сигнал",
        }
        if ready_sources:
            return " + ".join(names[source] for source in sorted(ready_sources, key=lambda s: s.value))
        if signals.typing_baseline_ready:
            return "личная норма готова; значимых устойчивых изменений сейчас нет"
        return "контекст рабочей сессии + формирование личной нормы"

    def _recent_self_report(self, now: float) -> bool:
        return (
            self._self_report_at is not None
            and now - self._self_report_at <= 2.0 * 60.0 * 60.0
        )


def build_recovery_signals(
    snapshot: SessionSnapshot,
    typing_baseline_service: TypingBaselineService,
) -> RecoverySignals:
    """Преобразует SessionSnapshot в безопасный набор сигналов движка."""

    metrics = snapshot.typing_metrics
    features = typing_baseline_service.features_from_metrics(metrics)
    deviations: dict[str, TypingDeviation] = {}

    for feature_name in AdaptiveRecoveryEngine.TYPING_FEATURES:
        comparison: BaselineComparison = typing_baseline_service.baseline.compare(
            WorkContext.TYPING,
            feature_name,
            features.get(feature_name),
            fallback_to_mixed=False,
        )
        deviations[feature_name] = TypingDeviation(
            feature_name=feature_name,
            actual_value=comparison.actual_value,
            baseline_median=comparison.baseline_median,
            robust_z=comparison.robust_z,
            ready=comparison.ready,
            confidence=comparison.confidence,
        )

    ocular = snapshot.ocular_metrics
    baseline = typing_baseline_service.baseline
    return RecoverySignals(
        captured_at=snapshot.captured_at,
        state=snapshot.state,
        fatigue_analysis_allowed=snapshot.fatigue_analysis_allowed,
        typing_data_ready=metrics.data_ready,
        typing_baseline_ready=baseline.initial_calibration_complete,
        typing_calibration_progress=baseline.calibration_progress,
        typing_deviations=deviations,
        ocular_calibrated=ocular.calibrated,
        ocular_data_ready=ocular.data_ready,
        ocular_quality=ocular.quality,
        ocular_signal_coverage=ocular.signal_coverage,
        ocular_blink_rate_per_min=ocular.blink_rate_per_min,
        ocular_perclos=ocular.perclos,
        ocular_long_closure_count=ocular.long_closure_count,
        ocular_severe_closure_detected=ocular.severe_closure_detected,
    )
