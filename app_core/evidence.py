"""Правила определения уровня наблюдаемой рабочей нагрузки.

Модуль отвечает на один вопрос: какой уровень соответствует текущему набору
признаков и почему именно этот.

Ключевое отличие от прежней реализации: правила заданы таблицей, а не
лесенкой `if`. Благодаря этому возвращается не только уровень, но и
идентификатор сработавшего правила вместе с готовой формулировкой причины.
Объяснение во вкладке «Решение» строится из этого же источника и не может
разойтись с фактическим решением.

Правила проверяются сверху вниз; побеждает первое подошедшее. Порядок строк
таблицы и есть приоритет.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum


class EvidenceSource(StrEnum):
    """Независимый канал наблюдения."""

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


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    """Один наблюдаемый признак с оценкой силы и готовности к решению."""

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
class EvidenceSummary:
    """Свёртка готовых к решению признаков.

    Правила смотрят только сюда и ничего не знают об устройстве
    `RecoveryEvidence`. Это позволяет тестировать таблицу правил без сборки
    полноценных сигналов.
    """

    codes: frozenset[str]
    sources: frozenset[EvidenceSource]
    max_severity: int
    typing_count: int
    has_any: bool

    @classmethod
    def from_evidence(cls, evidence: Iterable[RecoveryEvidence]) -> EvidenceSummary:
        ready = [item for item in evidence if item.decision_ready]
        return cls(
            codes=frozenset(item.code for item in ready),
            sources=frozenset(item.source for item in ready),
            max_severity=max((item.severity for item in ready), default=0),
            typing_count=sum(item.source is EvidenceSource.TYPING for item in ready),
            has_any=bool(ready),
        )


@dataclass(frozen=True, slots=True)
class LevelRule:
    """Одно правило таблицы.

    `urgent` означает, что уровень должен подниматься немедленно, без
    гистерезиса. Это свойство самого правила, а не отдельный расчёт.
    """

    rule_id: str
    level: WorkloadLevel
    condition: Callable[[EvidenceSummary], bool]
    explanation: str
    urgent: bool = False


@dataclass(frozen=True, slots=True)
class LevelDecision:
    """Результат прохода по таблице."""

    level: WorkloadLevel
    rule_id: str
    explanation: str
    urgent: bool


LEVEL_RULES: tuple[LevelRule, ...] = (
    # --- Уровень 5: восстановление приоритетно -------------------------------
    LevelRule(
        rule_id="very_high_self_report",
        level=WorkloadLevel.RECOVERY_PRIORITY,
        condition=lambda summary: "self_report_very_high" in summary.codes,
        explanation="Крайняя самооценка утомления или сонливости.",
        urgent=True,
    ),
    LevelRule(
        rule_id="severe_eye_closure",
        level=WorkloadLevel.RECOVERY_PRIORITY,
        condition=lambda summary: "severe_eye_closure" in summary.codes,
        explanation="Зафиксировано длительное закрытие глаз.",
        urgent=True,
    ),
    # --- Уровень 4: нагрузка выражена ----------------------------------------
    LevelRule(
        rule_id="strong_self_report",
        level=WorkloadLevel.EXPRESSED,
        condition=lambda summary: "self_report_strong" in summary.codes,
        explanation="Выраженная самооценка утомления или сонливости.",
    ),
    LevelRule(
        rule_id="two_independent_channels",
        level=WorkloadLevel.EXPRESSED,
        condition=lambda summary: len(summary.sources) >= 2 and summary.max_severity >= 2,
        explanation=(
            "Не менее двух независимых каналов показывают заметные изменения."
        ),
    ),
    LevelRule(
        rule_id="very_long_continuous_work",
        level=WorkloadLevel.EXPRESSED,
        condition=lambda summary: "very_long_continuous_work" in summary.codes,
        explanation="Очень длительная непрерывная рабочая сессия.",
    ),
    # --- Уровень 3: нагрузка становится устойчивой ---------------------------
    LevelRule(
        rule_id="long_continuous_work",
        level=WorkloadLevel.SUSTAINED,
        condition=lambda summary: "long_continuous_work" in summary.codes,
        explanation="Длительная непрерывная рабочая сессия.",
    ),
    LevelRule(
        rule_id="persistent_typing_anomalies",
        level=WorkloadLevel.SUSTAINED,
        condition=lambda summary: summary.typing_count >= 2,
        explanation=(
            "Не менее двух устойчивых отклонений ритма печати от личной нормы."
        ),
    ),
    LevelRule(
        rule_id="moderate_self_report_with_support",
        level=WorkloadLevel.SUSTAINED,
        condition=lambda summary: (
            "self_report_moderate" in summary.codes and len(summary.sources) >= 2
        ),
        explanation=(
            "Умеренная самооценка подтверждается ещё одним независимым каналом."
        ),
    ),
    # --- Уровень 2: появляются признаки нагрузки -----------------------------
    LevelRule(
        rule_id="any_ready_evidence",
        level=WorkloadLevel.EARLY,
        condition=lambda summary: summary.has_any,
        explanation="Появился ранний признак нагрузки, устойчивость пока не подтверждена.",
    ),
)

NO_EVIDENCE_DECISION = LevelDecision(
    level=WorkloadLevel.STABLE,
    rule_id="no_evidence",
    explanation="Доступные показатели находятся в обычном рабочем диапазоне.",
    urgent=False,
)


def derive_level(
    summary: EvidenceSummary,
    *,
    rules: Sequence[LevelRule] = LEVEL_RULES,
) -> LevelDecision:
    """Найти первое подошедшее правило.

    Если не подошло ни одно, состояние считается устойчивым.
    """

    for rule in rules:
        if rule.condition(summary):
            return LevelDecision(
                level=rule.level,
                rule_id=rule.rule_id,
                explanation=rule.explanation,
                urgent=rule.urgent,
            )
    return NO_EVIDENCE_DECISION


def derive_level_from_evidence(
    evidence: Iterable[RecoveryEvidence],
    *,
    rules: Sequence[LevelRule] = LEVEL_RULES,
) -> LevelDecision:
    """Удобная обёртка: свернуть признаки и сразу пройти по таблице."""

    return derive_level(EvidenceSummary.from_evidence(evidence), rules=rules)
