"""Просмотр нового интерфейса на выдуманных данных.

Запуск:

    python preview_ui.py

Камера, клавиатура и база данных не используются. Скрипт нужен, чтобы
крутить внешний вид быстро и не трогать рабочую программу: снизу есть
переключатель сценариев, показывающий все пять уровней нагрузки и разные
состояния камеры, а также кнопки для проверки двух отдельных окон —
самооценки и уведомления о рекомендации.

Требуется:

    pip install customtkinter
"""

from __future__ import annotations

import customtkinter as ctk

from ui import theme
from ui.main_screen import MainScreen
from ui.self_report import SelfReportDialog
from ui.toast import RecommendationToast
from ui.view_model import DetailBlock, MainScreenView, ReasonRow

SCENARIOS: dict[str, MainScreenView] = {
    "1 · спокойно": MainScreenView(
        work_time_text="00:12:04",
        work_progress=0.09,
        next_threshold_text="до зрительной разгрузки ещё 33 мин",
        signal_value_text="98%",
        signal_percent=98,
        signal_bars=5,
        signal_tone="good",
        signal_hint="лицо видно, взгляд к экрану",
        level=1,
        level_title="Рабочее состояние устойчиво",
        reasons=(
            ReasonRow("session", "сессия", "Работа началась недавно"),
            ReasonRow("typing", "печать", "Ритм соответствует вашей норме"),
        ),
        action_title="Продолжайте работу в обычном режиме",
        action_text="Отдельное восстановительное действие сейчас не требуется.",
        updated_text="обновлено сейчас",
        details=(
            DetailBlock(
                "Длительность работы",
                "Вы работаете 12 минут. До первого порога ещё далеко.",
                channel="session",
            ),
            DetailBlock(
                "Ритм печати",
                "Темп и доля исправлений держатся в пределах вашей личной нормы.",
                channel="typing",
            ),
            DetailBlock(
                "Самооценка не участвует",
                "Вы давно не отмечали, как себя чувствуете. Если отметите, вывод "
                "будет опираться на более широкое основание.",
                dim=True,
            ),
        ),
    ),
    "2 · ранние признаки": MainScreenView(
        work_time_text="00:48:31",
        work_progress=0.35,
        next_threshold_text="до микроперерыва ещё 21 мин",
        signal_value_text="91%",
        signal_percent=91,
        signal_bars=4,
        signal_tone="good",
        signal_hint="лицо видно, взгляд к экрану",
        level=2,
        level_title="Появляются признаки нагрузки",
        reasons=(
            ReasonRow("session", "сессия", "Продолжительность сессии выросла"),
        ),
        action_title="Пока достаточно наблюдения",
        action_text="Ранний сигнал ещё не требует отдельного действия.",
        updated_text="обновлено сейчас",
        details=(
            DetailBlock(
                "Длительность работы",
                "Вы работаете без полноценного перерыва 48 минут. Это первый "
                "временной признак, но сам по себе он мало о чём говорит.",
                channel="session",
            ),
            DetailBlock(
                "Ритм печати",
                "Устойчивых отклонений от вашей нормы сейчас нет.",
                channel="typing",
            ),
        ),
    ),
    "3 · устойчиво": MainScreenView(
        work_time_text="01:03:17",
        work_progress=0.52,
        next_threshold_text="до восстановительного перерыва ещё 42 мин",
        signal_value_text="72%",
        signal_percent=72,
        signal_bars=3,
        signal_tone="warn",
        signal_hint="голова часто повёрнута — сядьте лицом к камере",
        level=3,
        level_title="Нагрузка становится устойчивой",
        reasons=(
            ReasonRow("session", "сессия", "Длительная непрерывная работа"),
            ReasonRow("typing", "печать", "Выросла доля исправлений"),
        ),
        action_title="Уместна профилактическая разгрузка",
        action_text="Короткий перерыв после текущего действия обоснован.",
        show_action_buttons=True,
        updated_text="обновлено сейчас",
        details=(
            DetailBlock(
                "Длительность работы",
                "Час без полноценного перерыва. Короткие остановки в счёт не "
                "идут: перерывом считается пауза от трёх минут.",
                channel="session",
            ),
            DetailBlock(
                "Ритм печати",
                "Доля исправлений выше вашей обычной, и это повторилось в "
                "нескольких проверках подряд.",
                channel="typing",
            ),
            DetailBlock(
                "Сигнал камеры ослаб",
                "Часть кадров отбрасывается из-за поворота головы. На вывод "
                "это сейчас не влияет, но качество измерений ниже.",
                dim=True,
            ),
        ),
    ),
    "4 · выражена": MainScreenView(
        work_time_text="01:17:43",
        work_progress=0.62,
        next_threshold_text="до восстановительного перерыва ещё 22 мин",
        signal_value_text="94%",
        signal_percent=94,
        signal_bars=4,
        signal_tone="good",
        signal_hint="лицо видно, взгляд к экрану",
        level=4,
        level_title="Нагрузка выражена",
        reasons=(
            ReasonRow("session", "сессия", "Очень длинная непрерывная работа"),
            ReasonRow("typing", "печать", "Темп печати устойчиво снизился"),
        ),
        action_title="Восстановительный перерыв обоснован",
        action_text="Завершите текущую операцию и прервите экранную работу.",
        show_action_buttons=True,
        updated_text="обновлено сейчас",
        details=(
            DetailBlock(
                "Длительность работы",
                "Вы работаете без полноценного перерыва 1 час 18 минут. Это "
                "дольше интервала, после которого разгрузка обычно уместна.",
                channel="session",
            ),
            DetailBlock(
                "Ритм печати",
                "Ваш темп ниже обычной для вас нормы, причём не однократно, а "
                "в нескольких проверках подряд. Одиночное замедление "
                "программа бы не засчитала.",
                channel="typing",
            ),
            DetailBlock(
                "Самооценка не участвует",
                "Вы давно не отмечали, как себя чувствуете. Если отметите, "
                "вывод будет опираться на более широкое основание.",
                dim=True,
            ),
            DetailBlock(
                "Глаза только измеряются",
                "Моргания и закрытия век записываются, но на вывод не влияют: "
                "точность их распознавания пока не проверена на размеченных "
                "данных.",
                dim=True,
            ),
        ),
    ),
    "5 · приоритет": MainScreenView(
        work_time_text="02:11:56",
        work_progress=1.0,
        next_threshold_text="все временные пороги пройдены",
        signal_value_text="89%",
        signal_percent=89,
        signal_bars=4,
        signal_tone="good",
        signal_hint="лицо видно, взгляд к экрану",
        level=5,
        level_title="Восстановление приоритетно",
        reasons=(
            ReasonRow("self_report", "самооценка", "Вы отметили сильную усталость"),
            ReasonRow("session", "сессия", "Очень длинная непрерывная работа"),
            ReasonRow("typing", "печать", "Темп печати устойчиво снизился"),
        ),
        action_title="Восстановление приоритетно",
        action_text="Продолжение работы в прежнем темпе сейчас менее уместно, чем перерыв.",
        show_action_buttons=True,
        updated_text="обновлено сейчас",
        details=(
            DetailBlock(
                "Ваша самооценка",
                "Пятнадцать минут назад вы отметили усталость 6 из 7. Это "
                "самый весомый из доступных сигналов, потому что он приходит "
                "напрямую от вас.",
                channel="self_report",
            ),
            DetailBlock(
                "Длительность работы",
                "Больше двух часов без полноценного перерыва.",
                channel="session",
            ),
            DetailBlock(
                "Ритм печати",
                "Темп снизился относительно вашей нормы и держится ниже неё.",
                channel="typing",
            ),
        ),
    ),
    "камера темно": MainScreenView(
        work_time_text="00:34:02",
        work_progress=0.25,
        next_threshold_text="до зрительной разгрузки ещё 11 мин",
        signal_value_text="21%",
        signal_percent=21,
        signal_bars=1,
        signal_tone="bad",
        signal_hint="мало света — попробуйте включить лампу",
        level=2,
        level_title="Появляются признаки нагрузки",
        reasons=(
            ReasonRow("session", "сессия", "Продолжительность сессии выросла"),
        ),
        action_title="Пока достаточно наблюдения",
        action_text="Ранний сигнал ещё не требует отдельного действия.",
        updated_text="обновлено сейчас",
        details=(
            DetailBlock(
                "Сигнал камеры слабый",
                "Большая часть кадров отбраковывается из-за недостатка света. "
                "Визуальный канал сейчас почти не даёт информации.",
                dim=True,
            ),
        ),
    ),
    "без камеры": MainScreenView(
        monitoring_text="Мониторинг включён (без камеры)",
        work_time_text="00:52:19",
        work_progress=0.41,
        next_threshold_text="до микроперерыва ещё 18 мин",
        signal_value_text="нет",
        signal_percent=None,
        signal_bars=0,
        signal_tone="off",
        signal_hint="камера занята другим приложением",
        level=2,
        level_title="Появляются признаки нагрузки",
        reasons=(
            ReasonRow("session", "сессия", "Продолжительность сессии выросла"),
            ReasonRow("ocular", "глаза", "Канал недоступен", dim=True),
        ),
        action_title="Пока достаточно наблюдения",
        action_text="Ранний сигнал ещё не требует отдельного действия.",
        updated_text="обновлено сейчас",
        details=(
            DetailBlock(
                "Камера недоступна",
                "Визуальный канал отключён, остальные продолжают работать: "
                "длительность сессии, ритм печати и ваша самооценка.",
                dim=True,
            ),
        ),
    ),
}


def main() -> None:
    ctk.set_appearance_mode("dark")

    root = ctk.CTk()
    root.title("Ассистент восстановления — просмотр интерфейса")
    root.geometry("760x900")
    root.minsize(620, 700)
    root.configure(fg_color=theme.BG)

    holder: dict[str, MainScreen] = {}

    def screen_camera_toggle() -> None:
        holder["screen"].toggle_camera()

    screen = MainScreen(
        root,
        on_start_break=lambda: print("нажато: начать перерыв"),
        on_snooze=lambda: print("нажато: отложить"),
        on_dismiss=lambda: print("нажато: неуместно"),
        on_self_report=lambda: open_self_report(),
        on_technical=lambda: print("нажато: технические показатели"),
        on_monitoring_click=screen_camera_toggle,
    )
    holder["screen"] = screen
    screen.pack(fill="both", expand=True, padx=10, pady=(10, 0))

    # Просмотр интерфейса работает на выдуманных данных: сенсоров нет, и
    # кадру взяться неоткуда. Без подписи пустой квадрат читается как
    # сломанная камера.
    screen.set_camera_placeholder(
        "в просмотре интерфейса камера не подключена\n"
        "здесь будет изображение с вашей камеры"
    )

    def open_self_report() -> None:
        SelfReportDialog(
            root,
            initial_fatigue=None,
            initial_sleepiness=None,
            on_save=lambda f, s: print(f"самооценка сохранена: SP={f}, KSS={s}"),
        )

    def open_toast(demo: bool) -> None:
        RecommendationToast(
            root,
            title="Пора сделать перерыв",
            message=(
                "Работа идёт без остановки больше полутора часов, и ритм печати "
                "заметно изменился относительно вашей обычной нормы."
            ),
            duration_text="00:10:00",
            demo_mode=demo,
            on_accept=lambda: print("уведомление: начать перерыв"),
            on_snooze=lambda: print("уведомление: отложить"),
            on_irrelevant=lambda: print("уведомление: неуместно"),
            on_close=lambda: print("уведомление закрыто"),
        )

    windows = ctk.CTkFrame(root, fg_color="transparent")
    windows.pack(fill="x", padx=10, pady=(10, 0))

    ctk.CTkLabel(
        windows,
        text="отдельные окна:",
        font=ctk.CTkFont(family=theme.FONT_FAMILY, size=12),
        text_color=theme.TEXT_DIM,
    ).pack(side="left", padx=(0, 8))

    for caption, command in (
        ("самооценка", open_self_report),
        ("уведомление", lambda: open_toast(False)),
        ("уведомление · демо", lambda: open_toast(True)),
    ):
        ctk.CTkButton(
            windows,
            text=caption,
            font=ctk.CTkFont(family=theme.FONT_FAMILY, size=11),
            height=26,
            width=110,
            corner_radius=theme.RADIUS_BUTTON,
            fg_color="transparent",
            hover_color=theme.CARD,
            text_color=theme.TEXT_SECONDARY,
            border_width=1,
            border_color=theme.BORDER,
            command=command,
        ).pack(side="left", padx=(0, 8))

    switcher = ctk.CTkFrame(root, fg_color="transparent")
    switcher.pack(fill="x", padx=10, pady=10)

    ctk.CTkLabel(
        switcher,
        text="выдуманная ситуация:",
        font=ctk.CTkFont(family=theme.FONT_FAMILY, size=12),
        text_color=theme.TEXT_DIM,
    ).pack(side="left", padx=(0, 8))

    selector = ctk.CTkSegmentedButton(
        switcher,
        values=list(SCENARIOS),
        font=ctk.CTkFont(family=theme.FONT_FAMILY, size=11),
        command=lambda name: screen.render(SCENARIOS[name]),
    )
    selector.pack(side="left", fill="x", expand=True)
    selector.set("4 · выражена")

    screen.render(SCENARIOS["4 · выражена"])
    root.mainloop()


if __name__ == "__main__":
    main()
