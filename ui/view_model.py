"""Данные, которые показывает главный экран.

Экран ничего не вычисляет сам: он получает готовый `MainScreenView` и
рисует его. Благодаря этому внешний вид можно крутить на выдуманных данных
(`preview_ui.py`), не запуская камеру и датчики, а логика остаётся в
`app_core` и проверяется тестами отдельно.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ReasonRow:
    """Одна строка блока «Что на это повлияло»."""

    channel: str  # session | typing | self_report | ocular
    channel_label: str
    text: str
    dim: bool = False


@dataclass(frozen=True, slots=True)
class DetailBlock:
    """Один блок раскрытой панели «Подробнее»."""

    title: str
    text: str
    channel: str = "session"
    dim: bool = False


@dataclass(frozen=True, slots=True)
class MainScreenView:
    monitoring: bool = True
    monitoring_text: str = "Мониторинг включён"

    # таймер: во время перерыва карточка меняет заголовок и смысл
    work_label: str = "Непрерывная работа"
    work_time_text: str = "00:00:00"
    work_progress: float = 0.0  # 0..1 до следующего порога
    next_threshold_text: str = "до зрительной разгрузки ещё 45 мин"

    # сигнал камеры
    signal_value_text: str = "—"
    signal_percent: int | None = None
    signal_bars: int = 0  # 0..5
    signal_tone: str = "off"  # good | warn | bad | off
    signal_hint: str = "камера не используется"

    # уровень нагрузки
    level: int = 1
    level_title: str = "Рабочее состояние устойчиво"

    reasons: tuple[ReasonRow, ...] = ()

    # рекомендация
    action_title: str = "Продолжайте работу в обычном режиме"
    action_text: str = "Отдельное восстановительное действие сейчас не требуется."
    # Кнопки «Отложить» и «Неуместно» относятся к конкретной рекомендации и
    # без неё бессмысленны. Кнопка перерыва доступна всегда: человек вправе
    # прерваться, ничего не дожидаясь.
    show_action_buttons: bool = False
    break_active: bool = False

    updated_text: str = "обновлено сейчас"

    details: tuple[DetailBlock, ...] = ()
