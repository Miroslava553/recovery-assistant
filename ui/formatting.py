"""Чистые функции подготовки текста для интерфейса.

Модуль намеренно не зависит от tkinter: эти функции проверяются тестами
без запуска графического окна.
"""

from __future__ import annotations

from app_core.recovery_engine import RecoveryAssessment
from app_core.state_machine import UserState


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_optional(value: object, *, digits: int = 2) -> str:
    if value is None:
        return "нет данных"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def self_report_description(
    labels: dict[int, tuple[str, str]],
    value: int,
) -> tuple[str, str]:
    if value not in labels:
        raise ValueError("Значение отсутствует в шкале.")
    return labels[value]


def neighbor_description(
    labels: dict[int, tuple[str, str]],
    value: int,
) -> str:
    lines: list[str] = []
    if value - 1 in labels:
        short, description = labels[value - 1]
        lines.append(f"Ниже: {value - 1} — {short}. {description}")
    if value + 1 in labels:
        short, description = labels[value + 1]
        lines.append(f"Выше: {value + 1} — {short}. {description}")
    return "\n".join(lines)


def user_facing_summary(
    assessment: RecoveryAssessment,
    *,
    demo_mode: bool,
) -> tuple[str, str]:
    """Короткое состояние для первого из трёх пользовательских блоков."""

    if assessment.state in {UserState.AWAY, UserState.BREAK}:
        return (
            "Восстановительный интервал идёт",
            "Рабочая нагрузка в этот период не накапливается.",
        )
    if assessment.state is UserState.RETURNING:
        return (
            "Оценка возобновляется",
            "После возвращения система ждёт стабилизации рабочего контекста.",
        )
    if assessment.state is UserState.UNKNOWN:
        return (
            "Оценка временно ограничена",
            "Рабочий контекст пока недостаточно определён для устойчивого вывода.",
        )
    return assessment.workload_title, assessment.workload_summary
