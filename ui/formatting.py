"""Чистые функции подготовки текста для интерфейса.

Модуль намеренно не зависит от tkinter: эти функции проверяются тестами
без запуска графического окна.
"""

from __future__ import annotations

from app_core.recovery_engine import RecoveryAssessment, WorkloadLevel
from app_core.state_machine import UserState


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    # Всегда часы:минуты:секунды. Формат «00:12» читается как двенадцать
    # минут, хотя означает двенадцать секунд.
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def format_assessment_age(seconds: float) -> str:
    """Возраст последней оценки в понятной пользователю форме."""

    age = max(0.0, float(seconds))
    if age < 5.0:
        return "сейчас"
    if age < 60.0:
        return f"{int(age)} с назад"
    minutes = int(age // 60)
    if minutes < 60:
        return f"{minutes} мин назад"
    hours = minutes // 60
    return f"{hours} ч {minutes % 60:02d} мин назад"


RECOMMENDED_ACTIONS: dict[WorkloadLevel, tuple[str, str]] = {
    WorkloadLevel.STABLE: (
        "Продолжайте работу в обычном режиме",
        "Отдельное восстановительное действие сейчас не требуется.",
    ),
    WorkloadLevel.EARLY: (
        "Пока достаточно наблюдения",
        "Ранний сигнал ещё не требует отдельного восстановительного действия.",
    ),
    WorkloadLevel.SUSTAINED: (
        "Уместна профилактическая разгрузка",
        "Изменения стали устойчивыми. Короткий перерыв после текущего действия обоснован.",
    ),
    WorkloadLevel.EXPRESSED: (
        "Восстановительный перерыв обоснован",
        "Признаки выражены. Завершите текущую операцию и прервите экранную работу.",
    ),
    WorkloadLevel.RECOVERY_PRIORITY: (
        "Восстановление приоритетно",
        "Продолжение работы в прежнем темпе сейчас менее уместно, чем перерыв.",
    ),
}


def recommended_action(assessment: RecoveryAssessment) -> tuple[str, str]:
    """Что сейчас рекомендуется, независимо от показа уведомления.

    Разделение принципиальное: cooldown подавляет повторное всплывающее
    уведомление, но не должен превращать рекомендацию на главном экране в
    противоположный совет. Прежняя реализация при уровне «Нагрузка выражена»
    и отсутствии активного toast показывала «Продолжайте работу в обычном
    режиме», что прямо противоречило состоянию.
    """

    if assessment.state in {UserState.AWAY, UserState.BREAK}:
        return (
            "Продолжайте текущий перерыв",
            f"Текущая продолжительность: {format_duration(assessment.current_break_sec)}.",
        )
    if assessment.state is UserState.RETURNING:
        return (
            "Возвращайтесь к работе в спокойном темпе",
            "Оценка возобновится после стабилизации рабочего контекста.",
        )
    if assessment.state is UserState.UNKNOWN:
        return (
            "Рекомендация временно недоступна",
            "Рабочий контекст пока недостаточно определён для устойчивого вывода.",
        )
    return RECOMMENDED_ACTIONS[assessment.workload_level]

