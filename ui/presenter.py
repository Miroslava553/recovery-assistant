"""Перевод показаний движка в то, что рисует главный экран.

Модуль намеренно не знает ни про tkinter, ни про CustomTkinter: здесь только
преобразование данных. Благодаря этому всё, что видит человек на экране,
проверяется обычными тестами, без запуска окна.
"""

from __future__ import annotations

from app_core.recovery_engine import (
    RecoveryAssessment,
    RecoverySignals,
    WorkloadLevel,
)
from app_core.state_machine import UserState
from ui.formatting import format_assessment_age, format_duration, recommended_action
from ui.view_model import DetailBlock, MainScreenView, ReasonRow

CHANNEL_LABELS: dict[str, str] = {
    "session": "сессия",
    "typing": "печать",
    "self_report": "самооценка",
    "ocular": "глаза",
}

LEVEL_TITLES: dict[WorkloadLevel, str] = {
    WorkloadLevel.STABLE: "Рабочее состояние устойчиво",
    WorkloadLevel.EARLY: "Появляются признаки нагрузки",
    WorkloadLevel.SUSTAINED: "Нагрузка становится устойчивой",
    WorkloadLevel.EXPRESSED: "Нагрузка выражена",
    WorkloadLevel.RECOVERY_PRIORITY: "Восстановление приоритетно",
}

# Понятные подсказки вместо технических кодов отбраковки кадра.
CAMERA_HINTS: dict[str, str] = {
    "face_detected": "лицо видно, взгляд к экрану",
    "face_not_detected": "лицо не в кадре",
    "too_dark": "мало света — попробуйте включить лампу",
    "too_bright": "слишком яркий свет — уберите источник из-за спины",
    "image_quality_too_low": "изображение нечёткое — протрите камеру",
    "camera_frame_unavailable": "камера не передаёт изображение",
    "camera_unavailable": "камера занята другим приложением",
}


def _next_threshold_text(
    continuous_work_sec: float,
    *,
    eye_rest_after_sec: float,
    microbreak_after_sec: float,
    recovery_break_after_sec: float,
) -> tuple[float, str]:
    """Доля пути до ближайшего порога и подпись под таймером."""

    previous = 0.0
    for threshold in (eye_rest_after_sec, microbreak_after_sec, recovery_break_after_sec):
        if continuous_work_sec < threshold:
            span = threshold - previous
            done = continuous_work_sec - previous
            fraction = done / span if span > 0 else 0.0
            minutes = max(1, int((threshold - continuous_work_sec) // 60) + 1)
            return fraction, f"следующая проверка через {minutes} мин"
        previous = threshold
    return 1.0, "перерыв рекомендован"


def _camera_view(snapshot, *, vision_available: bool) -> tuple[str, int | None, int, str, str]:
    """Значение, процент, число сегментов, тон и подсказка для карточки камеры."""

    if not vision_available:
        return "нет", None, 0, "off", CAMERA_HINTS["camera_unavailable"]

    coverage = float(getattr(snapshot.ocular_metrics, "signal_coverage", 0.0) or 0.0)
    percent = int(round(coverage * 100))
    bars = max(0, min(5, int(round(coverage * 5))))

    if snapshot.face_detected is None:
        return "нет", None, 0, "off", "визуальный канал недоступен"

    if snapshot.face_detected is False:
        hint = CAMERA_HINTS.get(snapshot.camera_status, "лицо не в кадре")
        return f"{percent}%", percent, max(bars, 1), "bad", hint

    if coverage >= 0.85:
        tone = "good"
    elif coverage >= 0.5:
        tone = "warn"
    else:
        tone = "bad"

    if tone == "good":
        hint = "лицо видно, взгляд к экрану"
    else:
        hint = CAMERA_HINTS.get(
            snapshot.camera_status,
            "часть кадров отбраковывается — сядьте лицом к камере",
        )
    return f"{percent}%", percent, max(bars, 1), tone, hint


def _reasons(assessment: RecoveryAssessment) -> tuple[ReasonRow, ...]:
    rows: list[ReasonRow] = []
    for item in assessment.evidence:
        if not item.decision_ready:
            continue
        rows.append(
            ReasonRow(
                channel=str(item.source),
                channel_label=CHANNEL_LABELS.get(str(item.source), str(item.source)),
                text=item.title,
            )
        )
        if len(rows) == 4:
            break
    if rows:
        return tuple(rows)

    return (
        ReasonRow(
            channel="session",
            channel_label="сессия",
            text="Устойчивых признаков нагрузки сейчас нет",
            dim=True,
        ),
    )


def _details(
    assessment: RecoveryAssessment,
    signals: RecoverySignals,
    *,
    vision_available: bool,
) -> tuple[DetailBlock, ...]:
    """Объяснение вывода обычным языком.

    Здесь же честно перечисляется, каких каналов не хватает. Именно это
    заменяет прежнее слово «надёжность»: человеку полезнее знать, что можно
    сделать, чем услышать абстрактную оценку качества данных.
    """

    blocks: list[DetailBlock] = []

    minutes = int(assessment.continuous_work_sec // 60)
    blocks.append(
        DetailBlock(
            title="Длительность работы",
            text=(
                f"Вы работаете без полноценного перерыва {format_duration(assessment.continuous_work_sec)}. "
                "Перерывом считается пауза от трёх минут: короткие остановки в счёт не идут."
                if minutes >= 1
                else "Рабочая сессия только началась."
            ),
            channel="session",
        )
    )

    if signals.typing_baseline_ready:
        typing_codes = [
            item.title
            for item in assessment.evidence
            if str(item.source) == "typing" and item.decision_ready
        ]
        if typing_codes:
            blocks.append(
                DetailBlock(
                    title="Ритм печати",
                    text=(
                        "Отклонение от вашей личной нормы повторилось в нескольких "
                        "проверках подряд. Одиночное изменение программа бы не засчитала."
                    ),
                    channel="typing",
                )
            )
        else:
            blocks.append(
                DetailBlock(
                    title="Ритм печати",
                    text="Темп и доля исправлений держатся в пределах вашей личной нормы.",
                    channel="typing",
                )
            )
    else:
        progress = int(signals.typing_calibration_progress * 100)
        blocks.append(
            DetailBlock(
                title="Личная норма печати ещё формируется",
                text=(
                    f"Готовность {progress}%. Пока нормы нет, сравнивать не с чем, "
                    "и отклонения ритма не учитываются."
                ),
                dim=True,
            )
        )

    if assessment.self_report_fatigue_sp is None:
        blocks.append(
            DetailBlock(
                title="Самооценка не участвует",
                text=(
                    "Вы давно не отмечали, как себя чувствуете. Если отметите, "
                    "вывод будет опираться на более широкое основание."
                ),
                dim=True,
            )
        )
    else:
        blocks.append(
            DetailBlock(
                title="Ваша самооценка",
                text=(
                    f"Усталость {assessment.self_report_fatigue_sp} из 7, "
                    f"сонливость {assessment.self_report_sleepiness_kss} из 9. "
                    "Это самый весомый сигнал, потому что он приходит напрямую от вас."
                ),
                channel="self_report",
            )
        )

    if not vision_available:
        blocks.append(
            DetailBlock(
                title="Камера недоступна",
                text=(
                    "Визуальный канал отключён, остальные продолжают работать: "
                    "длительность сессии, ритм печати и ваша самооценка."
                ),
                dim=True,
            )
        )
    elif not assessment.ocular_used_for_decision:
        blocks.append(
            DetailBlock(
                title="Глаза только измеряются",
                text=(
                    "Моргания и закрытия век записываются, но на вывод не влияют: "
                    "точность их распознавания пока не проверена на размеченных данных."
                ),
                dim=True,
            )
        )

    return tuple(blocks[:5])


def _break_view(
    assessment: RecoveryAssessment,
    *,
    now: float,
    monitoring: bool,
    meaningful_break_sec: float,
) -> MainScreenView:
    """Во время перерыва экран показывает перерыв, а не рабочие показатели.

    Признаки нагрузки в это время не оцениваются: человек не работает, и
    любые числа с камеры или клавиатуры описывали бы не рабочее поведение.
    Поэтому карточки честно сообщают, что измерение приостановлено.
    """

    elapsed = assessment.current_break_sec
    counted = elapsed >= meaningful_break_sec
    if counted:
        note = "Перерыв засчитан: рабочая сессия начнётся заново."
        progress = 1.0
    else:
        left = max(1, int((meaningful_break_sec - elapsed) // 60) + 1)
        note = f"засчитается через {left} мин"
        progress = elapsed / meaningful_break_sec if meaningful_break_sec else 0.0

    return MainScreenView(
        monitoring=monitoring,
        monitoring_text="Идёт перерыв",
        work_label="Перерыв",
        work_time_text=format_duration(elapsed),
        work_progress=min(1.0, progress),
        next_threshold_text=note if not counted else "перерыв засчитан",
        signal_value_text="пауза",
        signal_percent=None,
        signal_bars=0,
        signal_tone="off",
        signal_hint="во время перерыва данные не оцениваются",
        level=int(assessment.workload_level),
        level_title="Вы начали перерыв",
        reasons=(
            ReasonRow(
                channel="session",
                channel_label="перерыв",
                text="Оценка рабочего состояния приостановлена",
                dim=True,
            ),
        ),
        action_title="Перерыв идёт",
        action_text=note if not counted else "Перерыв засчитан. Можно возвращаться к работе.",
        show_action_buttons=False,
        break_active=True,
        updated_text="обновлено " + format_assessment_age(now - assessment.captured_at),
        details=(
            DetailBlock(
                title="Что происходит во время перерыва",
                text=(
                    "Программа не оценивает признаки нагрузки: вы не работаете, и "
                    "показания камеры и клавиатуры сейчас не описывают рабочее поведение."
                ),
                dim=True,
            ),
            DetailBlock(
                title="Почему счётчик работы обнулится",
                text=(
                    f"Перерывом считается пауза от {int(meaningful_break_sec // 60) or 1} мин. "
                    "После неё непрерывная работа начинается заново — в этом и смысл перерыва. "
                    "Более короткие остановки счётчик не сбрасывают."
                ),
                dim=True,
            ),
        ),
    )


def build_main_screen_view(
    snapshot,
    signals: RecoverySignals,
    assessment: RecoveryAssessment,
    *,
    now: float,
    monitoring: bool = True,
    vision_available: bool = True,
    break_active: bool = False,
    meaningful_break_sec: float = 180.0,
    returning_duration_sec: float = 60.0,
    eye_rest_after_sec: float,
    microbreak_after_sec: float,
    recovery_break_after_sec: float,
) -> MainScreenView:
    if assessment.state is UserState.RETURNING:
        # Защитный период после перерыва: счётчики намеренно стоят. Без
        # объяснения это выглядит как зависшая программа.
        seconds_left = max(0, int(returning_duration_sec - getattr(snapshot, "seconds_in_state", 0.0)))
        return MainScreenView(
            monitoring=monitoring,
            monitoring_text="Возвращение к работе",
            work_label="Отсчёт приостановлен",
            work_time_text=format_duration(assessment.continuous_work_sec),
            work_progress=0.0,
            next_threshold_text=(
                f"счётчик возобновится через {seconds_left} с"
                if seconds_left
                else "счётчик вот-вот возобновится"
            ),
            signal_value_text="пауза",
            signal_percent=None,
            signal_bars=0,
            signal_tone="off",
            signal_hint="оценка возобновится после возвращения к работе",
            level=int(assessment.workload_level),
            level_title="Возвращение к работе",
            reasons=(
                ReasonRow(
                    channel="session",
                    channel_label="контекст",
                    text="Рабочее состояние ещё не устоялось",
                    dim=True,
                ),
            ),
            action_title="Возвращайтесь к работе в спокойном темпе",
            action_text="Оценка возобновится, когда рабочий контекст станет устойчивым.",
            show_action_buttons=False,
            updated_text="обновлено " + format_assessment_age(now - assessment.captured_at),
            details=(
                DetailBlock(
                    title="Почему счётчики стоят",
                    text=(
                        "Сразу после перерыва действует короткий защитный период. "
                        "Пока он идёт, программа не начисляет ни рабочее время, ни "
                        "время перерыва: рабочее поведение ещё не установилось, и "
                        "любые выводы были бы преждевременными."
                    ),
                    dim=True,
                ),
            ),
        )

    on_break = break_active or assessment.state in {UserState.BREAK, UserState.AWAY}
    if on_break:
        return _break_view(
            assessment,
            now=now,
            monitoring=monitoring,
            meaningful_break_sec=meaningful_break_sec,
        )

    progress, threshold_text = _next_threshold_text(
        assessment.continuous_work_sec,
        eye_rest_after_sec=eye_rest_after_sec,
        microbreak_after_sec=microbreak_after_sec,
        recovery_break_after_sec=recovery_break_after_sec,
    )
    value, percent, bars, tone, hint = _camera_view(
        snapshot, vision_available=vision_available
    )
    action_title, action_text = recommended_action(assessment)
    recommendation = assessment.recommendation

    if monitoring:
        monitoring_text = (
            "Мониторинг включён" if vision_available else "Мониторинг включён (без камеры)"
        )
    else:
        monitoring_text = "Мониторинг приостановлен"

    if recommendation is not None:
        action_title = recommendation.title
        action_text = recommendation.message

    return MainScreenView(
        monitoring=monitoring,
        monitoring_text=monitoring_text,
        work_time_text=format_duration(assessment.continuous_work_sec),
        work_progress=progress,
        next_threshold_text=threshold_text,
        signal_value_text=value,
        signal_percent=percent,
        signal_bars=bars,
        signal_tone=tone,
        signal_hint=hint,
        level=int(assessment.workload_level),
        level_title=LEVEL_TITLES[assessment.workload_level],
        reasons=_reasons(assessment),
        action_title=action_title,
        action_text=action_text,
        show_action_buttons=recommendation is not None,
        break_active=break_active,
        updated_text="обновлено " + format_assessment_age(now - assessment.captured_at),
        details=_details(assessment, signals, vision_available=vision_available),
    )
