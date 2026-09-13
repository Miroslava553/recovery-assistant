"""Главный экран ассистента восстановления.

Экран строится один раз, дальше только обновляется методом `render()`.
Пересоздавать виджеты каждую секунду нельзя: окно будет мигать.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import tkinter as tk

import customtkinter as ctk

from ui import theme
from ui.formatting import wrap_width
from ui.view_model import MainScreenView


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ctk.CTkFont:
    return ctk.CTkFont(
        family=theme.FONT_MONO if mono else theme.FONT_FAMILY,
        size=size,
        weight="bold" if bold else "normal",
    )


class SegmentedBar(ctk.CTkFrame):
    """Полоска из отдельных сегментов — уровень или качество сигнала."""

    def __init__(self, master, *, segments: int = 5, height: int = 7, gap: int = 3):
        super().__init__(master, fg_color="transparent", height=height)
        self.pack_propagate(False)
        self._segments: list[ctk.CTkFrame] = []
        for index in range(segments):
            block = ctk.CTkFrame(
                self,
                fg_color=theme.LEVEL_EMPTY,
                corner_radius=height // 2 or 1,
                height=height,
            )
            padx = (0, gap) if index < segments - 1 else (0, 0)
            block.pack(side="left", expand=True, fill="both", padx=padx)
            self._segments.append(block)

    def render(self, *, filled: int, color: str, empty: str = theme.LEVEL_EMPTY) -> None:
        for index, block in enumerate(self._segments):
            block.configure(fg_color=color if index < filled else empty)


class Pill(ctk.CTkLabel):
    """Скруглённая метка: статус мониторинга, название канала."""

    def __init__(self, master, *, text: str, bg: str, fg: str, size: int = 11):
        super().__init__(
            master,
            text=text,
            font=_font(size),
            text_color=fg,
            fg_color=bg,
            corner_radius=theme.RADIUS_PILL,
            padx=10,
            pady=3,
        )


@dataclass(frozen=True, slots=True)
class _WrappedLabel:
    """Метка, ширину переноса которой считаем по фактической разметке.

    `container` — виджет, чья ширина и есть доступное место. `neighbours` —
    то, что стоит на той же строке слева или справа и это место отнимает
    (например плашка канала перед текстом основания). `gap` — отступ между
    меткой и соседями.
    """

    label: ctk.CTkLabel
    container: ctk.CTkBaseClass
    neighbours: tuple[ctk.CTkBaseClass, ...] = ()
    gap: int = 0


class Card(ctk.CTkFrame):
    def __init__(self, master, *, color: str = theme.CARD, **kwargs):
        super().__init__(
            master,
            fg_color=color,
            corner_radius=theme.RADIUS_CARD,
            **kwargs,
        )


class MainScreen(ctk.CTkFrame):
    """Три смысловых блока плюс компактный контекст сверху.

    Порядок продиктован продуктовым правилом: сотруднику нужны не пятьдесят
    параметров, а состояние, основание и рекомендация. Технические
    показатели живут за кнопкой «Подробнее».
    """

    CAMERA_WIDTH = 460
    CAMERA_HEIGHT = 460

    # Пикселей на одну «единицу» прокрутки. Больше — быстрее колесо.
    SCROLL_PIXELS_PER_UNIT = 5

    def __init__(
        self,
        master,
        *,
        on_start_break: Callable[[], None] | None = None,
        on_snooze: Callable[[], None] | None = None,
        on_dismiss: Callable[[], None] | None = None,
        on_self_report: Callable[[], None] | None = None,
        on_toggle_details: Callable[[bool], None] | None = None,
        on_technical: Callable[[], None] | None = None,
        on_camera: Callable[[], None] | None = None,
        on_monitoring_click: Callable[[], None] | None = None,
        max_content_width: int = 720,
    ):
        super().__init__(master, fg_color=theme.BG, corner_radius=theme.RADIUS_CARD)

        self._on_start_break = on_start_break
        self._on_snooze = on_snooze
        self._on_dismiss = on_dismiss
        self._on_toggle_details = on_toggle_details
        self._details_open = False
        self._camera_visible = False
        self._reason_rows: list[tuple[ctk.CTkFrame, Pill, ctk.CTkLabel]] = []
        self._detail_cards: list[tuple[Card, ctk.CTkLabel, ctk.CTkLabel]] = []

        # Метки, у которых ширину переноса надо пересчитывать при изменении
        # размера окна. Для каждой хранится контейнер, по которому меряется
        # доступная ширина, и соседи, занимающие часть той же строки.
        self._wrapping: list[_WrappedLabel] = []
        self._wrap_widths: dict[int, int] = {}
        self._wrap_pending = False

        self.max_content_width = max_content_width
        self._side_padding = 20
        self._last_content_width = 0
        self._last_inner_width = 0

        # Прокручиваемая область: при раскрытом «Подробнее» содержимое не
        # помещается в окно, и без прокрутки нижние карточки становятся
        # недоступны.
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True)
        self._tune_scroll_speed()

        self.outer = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.outer.pack(fill="both", expand=True, padx=self._side_padding, pady=18)

        self._build_header(self.outer, on_self_report, on_monitoring_click)
        self._build_top_cards(self.outer)
        self._build_camera_panel(self.outer)
        self._build_level(self.outer)
        self._build_reasons(self.outer)
        self._build_action(self.outer)
        self._build_details(self.outer, on_technical, on_camera)

        self.bind("<Configure>", self._on_resize)
        self.outer.bind("<Configure>", self._on_content_resize)

    # ------------------------------------------------------------------ шапка
    def _build_header(self, parent, on_self_report, on_monitoring_click=None) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 16))

        # Название программы здесь не повторяем: оно уже стоит в заголовке
        # окна. Раньше строка была занята им, и трём элементам не хватало
        # места — «Как я себя чувствую» сплющивалось в полоску.

        # Плашка кликабельна: по ней открывается диагностическое превью
        # камеры. Отдельной кнопки для этого нет — статус камеры и её
        # изображение логично держать в одном месте.
        self.monitor_pill = ctk.CTkButton(
            row,
            text="Мониторинг включён",
            font=_font(12),
            fg_color=theme.BADGE_ON_BG,
            hover_color=theme.ACTION_BORDER,
            text_color=theme.BADGE_ON_TEXT,
            corner_radius=theme.RADIUS_PILL,
            height=28,
            command=lambda: self._call(on_monitoring_click),
        )
        self.monitor_pill.pack(side="right")

        if on_self_report is not None:
            ctk.CTkButton(
                row,
                text="Как я себя чувствую",
                font=_font(12),
                fg_color="transparent",
                hover_color=theme.CARD,
                text_color=theme.TEXT_SECONDARY,
                border_width=1,
                border_color=theme.BORDER,
                corner_radius=theme.RADIUS_BUTTON,
                height=28,
                command=on_self_report,
            ).pack(side="left")

    # ------------------------------------------------- таймер и сигнал камеры
    def _build_top_cards(self, parent) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 12))
        self.top_cards_row = row
        row.grid_columnconfigure(0, weight=1, uniform="top")
        row.grid_columnconfigure(1, weight=1, uniform="top")

        # --- непрерывная работа
        timer = Card(row)
        timer.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        inner = ctk.CTkFrame(timer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        self.work_label = ctk.CTkLabel(
            inner,
            text="Непрерывная работа",
            font=_font(12),
            text_color=theme.TEXT_SECONDARY,
        )
        self.work_label.pack(anchor="w")

        self.work_time_label = ctk.CTkLabel(
            inner,
            text="00:00:00",
            font=_font(30, mono=True),
            text_color=theme.TEXT,
        )
        self.work_time_label.pack(anchor="w", pady=(4, 0))

        self.work_progress = ctk.CTkProgressBar(
            inner,
            height=5,
            corner_radius=3,
            fg_color=theme.BORDER,
            progress_color=theme.LEVEL_COLORS[1]["accent"],
        )
        self.work_progress.set(0.0)
        self.work_progress.pack(fill="x", pady=(10, 0))

        self.next_threshold_label = ctk.CTkLabel(
            inner,
            text="",
            font=_font(11),
            text_color=theme.TEXT_DIM,
            wraplength=120,
            anchor="w",
            justify="left",
        )
        self.next_threshold_label.pack(fill="x", pady=(5, 0))
        self._wrapping.append(_WrappedLabel(self.next_threshold_label, inner))

        # --- сигнал камеры
        signal = Card(row)
        signal.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        inner = ctk.CTkFrame(signal, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        ctk.CTkLabel(
            inner,
            text="Сигнал камеры",
            font=_font(12),
            text_color=theme.TEXT_SECONDARY,
        ).pack(anchor="w")

        self.signal_value_label = ctk.CTkLabel(
            inner,
            text="—",
            font=_font(30),
            text_color=theme.TEXT,
        )
        self.signal_value_label.pack(anchor="w", pady=(4, 0))

        self.signal_bar = SegmentedBar(inner, segments=5, height=5)
        self.signal_bar.pack(fill="x", pady=(10, 0))

        self.signal_hint_label = ctk.CTkLabel(
            inner,
            text="",
            font=_font(11),
            text_color=theme.TEXT_DIM,
            wraplength=120,
            anchor="w",
            justify="left",
        )
        self.signal_hint_label.pack(fill="x", pady=(5, 0))
        self._wrapping.append(_WrappedLabel(self.signal_hint_label, inner))

    # --------------------------------------------------- встроенная камера
    def _build_camera_panel(self, parent) -> None:
        """Превью прямо в экране, а не отдельным окном.

        Карточка не растягивается на всю ширину: она облегает изображение и
        прижата к правому краю, под кнопку мониторинга, которая её открывает.
        """

        self.camera_card = Card(parent)
        inner = ctk.CTkFrame(self.camera_card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        header = ctk.CTkFrame(inner, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(
            header,
            text="Диагностическое превью",
            font=_font(12),
            text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")
        ctk.CTkButton(
            header,
            text="Скрыть",
            font=_font(12),
            fg_color="transparent",
            hover_color=theme.CARD_MUTED,
            text_color=theme.TEXT_DIM,
            corner_radius=theme.RADIUS_BUTTON,
            height=26,
            width=70,
            command=self.hide_camera,
        ).pack(side="right")

        frame_box = ctk.CTkFrame(
            inner,
            fg_color=theme.CAMERA_BG,
            border_width=3,
            border_color=theme.CAMERA_BORDER,
            corner_radius=theme.RADIUS_CARD,
        )
        frame_box.pack(anchor="w", pady=(10, 0))

        self.camera_canvas = tk.Canvas(
            frame_box,
            width=self.CAMERA_WIDTH,
            height=self.CAMERA_HEIGHT,
            background=theme.CAMERA_BG,
            highlightthickness=0,
        )
        self.camera_canvas.pack(padx=7, pady=7)

        ctk.CTkLabel(
            inner,
            text="не сохраняется и никуда не отправляется",
            font=_font(11),
            text_color=theme.TEXT_DIM,
        ).pack(anchor="w", pady=(8, 0))

    @property
    def camera_visible(self) -> bool:
        return self._camera_visible

    def show_camera(self) -> None:
        if self._camera_visible:
            return
        self._camera_visible = True
        self.camera_card.pack(fill="x", pady=(0, 12), after=self.top_cards_row)

    def hide_camera(self) -> None:
        if not self._camera_visible:
            return
        self._camera_visible = False
        self.camera_card.pack_forget()

    def toggle_camera(self) -> None:
        if self._camera_visible:
            self.hide_camera()
        else:
            self.show_camera()

    # ------------------------------------------------------ уровень нагрузки
    def _build_level(self, parent) -> None:
        self.level_card = Card(parent, color=theme.LEVEL_COLORS[1]["bg"])
        self.level_card.pack(fill="x", pady=(0, 12))
        inner = ctk.CTkFrame(self.level_card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=18, pady=16)

        self.level_caption = ctk.CTkLabel(
            inner,
            text="Уровень 1 из 5",
            font=_font(12),
            text_color=theme.LEVEL_COLORS[1]["label"],
        )
        self.level_caption.pack(anchor="w")

        self.level_title = ctk.CTkLabel(
            inner,
            text="",
            font=_font(21),
            text_color=theme.LEVEL_COLORS[1]["title"],
            wraplength=120,
            anchor="w",
            justify="left",
        )
        self.level_title.pack(fill="x", pady=(6, 11))
        self._wrapping.append(_WrappedLabel(self.level_title, inner))

        self.level_bar = SegmentedBar(inner, segments=5, height=7)
        self.level_bar.pack(fill="x")

    # ------------------------------------------------- что на это повлияло
    def _build_reasons(self, parent) -> None:
        self.reasons_card = Card(parent)
        self.reasons_card.pack(fill="x", pady=(0, 12))
        inner = ctk.CTkFrame(self.reasons_card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        ctk.CTkLabel(
            inner,
            text="Основание",
            font=_font(12),
            text_color=theme.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(0, 8))

        # Строк заранее четыре — по числу каналов. Лишние прячутся, а не
        # пересоздаются: так экран не мигает при обновлении раз в секунду.
        for _ in range(4):
            row = ctk.CTkFrame(inner, fg_color="transparent")
            pill = Pill(row, text="", bg=theme.CARD_MUTED, fg=theme.TEXT_DIM)
            pill.pack(side="left")
            text = ctk.CTkLabel(
                row,
                text="",
                font=_font(13),
                text_color=theme.TEXT_SECONDARY,
                wraplength=120,
                anchor="w",
                justify="left",
            )
            text.pack(side="left", fill="x", expand=True, padx=(9, 0))
            # Плашка канала занимает часть строки, и занимает по-разному:
            # «сессия» и «самооценка» заметно разной ширины.
            self._wrapping.append(_WrappedLabel(text, row, (pill,), gap=9))
            self._reason_rows.append((row, pill, text))

    # ------------------------------------------------------------ действие
    def _build_action(self, parent) -> None:
        self.action_card = Card(parent, color=theme.ACTION_BG)
        self.action_card.pack(fill="x", pady=(0, 14))
        inner = ctk.CTkFrame(self.action_card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=18, pady=16)

        ctk.CTkLabel(
            inner,
            text="Рекомендация",
            font=_font(12),
            text_color=theme.ACTION_LABEL,
        ).pack(anchor="w")

        self.action_title = ctk.CTkLabel(
            inner,
            text="",
            font=_font(17),
            text_color=theme.ACTION_TITLE,
            wraplength=120,
            anchor="w",
            justify="left",
        )
        self.action_title.pack(fill="x", pady=(6, 0))
        self._wrapping.append(_WrappedLabel(self.action_title, inner))

        self.action_text = ctk.CTkLabel(
            inner,
            text="",
            font=_font(13),
            text_color=theme.ACTION_LABEL,
            wraplength=120,
            anchor="w",
            justify="left",
        )
        self.action_text.pack(fill="x", pady=(6, 0))
        self._wrapping.append(_WrappedLabel(self.action_text, inner))

        self.action_buttons = ctk.CTkFrame(inner, fg_color="transparent")
        self.action_buttons.pack(anchor="w", pady=(12, 0))

        self.break_button = ctk.CTkButton(
            self.action_buttons,
            text="Начать перерыв",
            font=_font(13),
            fg_color=theme.ACTION_PRIMARY,
            hover_color=theme.ACTION_PRIMARY_HOVER,
            text_color=theme.ACTION_PRIMARY_TEXT,
            corner_radius=theme.RADIUS_BUTTON,
            height=34,
            command=lambda: self._call(self._on_start_break),
        )
        self.break_button.pack(side="left")

        self.snooze_button = ctk.CTkButton(
            self.action_buttons,
            text="Отложить 10 мин",
            font=_font(13),
            fg_color="transparent",
            hover_color=theme.ACTION_BG,
            text_color=theme.ACTION_LABEL,
            border_width=1,
            border_color=theme.ACTION_BORDER,
            corner_radius=theme.RADIUS_BUTTON,
            height=34,
            command=lambda: self._call(self._on_snooze),
        )

        self.dismiss_button = ctk.CTkButton(
            self.action_buttons,
            text="Неуместно",
            font=_font(13),
            fg_color="transparent",
            hover_color=theme.CARD,
            text_color=theme.TEXT_DIM,
            border_width=1,
            border_color=theme.BORDER,
            corner_radius=theme.RADIUS_BUTTON,
            height=34,
            command=lambda: self._call(self._on_dismiss),
        )

    # ----------------------------------------------------------- подробнее
    def _build_details(self, parent, on_technical, on_camera) -> None:
        separator = ctk.CTkFrame(parent, fg_color=theme.CARD, height=1)
        separator.pack(fill="x")

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(10, 0))

        self.details_button = ctk.CTkButton(
            row,
            text="Подробнее  ▾",
            font=_font(13),
            fg_color="transparent",
            hover_color=theme.CARD,
            text_color=theme.TEXT_SECONDARY,
            corner_radius=theme.RADIUS_BUTTON,
            height=28,
            anchor="w",
            command=self.toggle_details,
        )
        self.details_button.pack(side="left")

        self.updated_label = ctk.CTkLabel(
            row,
            text="обновлено сейчас",
            font=_font(12),
            text_color=theme.TEXT_DIM,
        )
        self.updated_label.pack(side="right")

        self.details_panel = ctk.CTkFrame(parent, fg_color="transparent")

        ctk.CTkLabel(
            self.details_panel,
            text="Подробное обоснование",
            font=_font(15),
            text_color=theme.TEXT,
        ).pack(anchor="w", pady=(14, 12))

        self.details_cards_area = ctk.CTkFrame(
            self.details_panel, fg_color="transparent"
        )
        self.details_cards_area.pack(fill="x")

        for _ in range(5):
            card = Card(self.details_cards_area)
            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="both", expand=True, padx=16, pady=14)
            title = ctk.CTkLabel(
                inner,
                text="",
                font=_font(12),
                text_color=theme.TEXT_SECONDARY,
            )
            title.pack(anchor="w")
            body = ctk.CTkLabel(
                inner,
                text="",
                font=_font(14),
                text_color=theme.TEXT_SECONDARY,
                wraplength=120,
                anchor="w",
                justify="left",
            )
            body.pack(fill="x", pady=(5, 0))
            self._wrapping.append(_WrappedLabel(body, inner))
            self._detail_cards.append((card, title, body))

        buttons = ctk.CTkFrame(self.details_panel, fg_color="transparent")
        buttons.pack(fill="x", pady=(6, 0))

        for text, command in (("Технические показатели", on_technical),):
            ctk.CTkButton(
                buttons,
                text=text,
                font=_font(13),
                fg_color="transparent",
                hover_color=theme.CARD,
                text_color=theme.TEXT_SECONDARY,
                border_width=1,
                border_color=theme.BORDER,
                corner_radius=theme.RADIUS_BUTTON,
                height=32,
                command=(lambda c=command: self._call(c)),
            ).pack(side="left", padx=(0, 8))

    def _tune_scroll_speed(self) -> None:
        """Сделать шаг колеса заметнее.

        CustomTkinter прокручивает холст «единицами», а размер единицы
        задаётся самим холстом. По умолчанию он не задан, и шаг получается
        мелким. Явное значение в пикселях делает прокрутку предсказуемой:
        больше число — быстрее прокрутка.
        """

        try:
            self.scroll._parent_canvas.configure(
                yscrollincrement=self.SCROLL_PIXELS_PER_UNIT
            )
        except Exception:
            pass

    # ---------------------------------------------------------- размер окна
    def _on_resize(self, event) -> None:
        """Не давать содержимому растягиваться на всю ширину монитора.

        Строка текста в тысячу пикселей нечитаема: глаз теряет начало
        следующей строки. Поэтому содержимое ограничивается по ширине и
        центрируется, а лишнее место уходит в поля.
        """

        available = max(320, event.width - 2 * self._side_padding)
        content = min(available, self.max_content_width)
        extra = max(0, available - content)
        padding = self._side_padding + extra // 2

        if abs(content - self._last_content_width) < 4:
            return
        self._last_content_width = content

        self.outer.pack_configure(padx=padding)
        self._schedule_wrapping()

    def _on_content_resize(self, event) -> None:
        """Ширина содержимого меняется не только вместе с окном.

        Полоса прокрутки появляется и исчезает сама, и внутренняя ширина
        меняется без изменения размера окна. Поэтому слушаем и её.
        """

        if abs(event.width - self._last_inner_width) < 2:
            return
        self._last_inner_width = event.width
        self._schedule_wrapping()

    def _schedule_wrapping(self) -> None:
        """Отложить пересчёт до конца текущей раскладки.

        Ширину надо спрашивать у tkinter уже после того, как он разложил
        виджеты. Если спросить сразу, вернутся прошлые значения — заметно,
        например, когда плашка канала сменилась с «сессия» на «самооценка».
        """

        if self._wrap_pending:
            return
        self._wrap_pending = True
        try:
            self.after_idle(self._apply_wrapping)
        except Exception:
            self._wrap_pending = False

    def _apply_wrapping(self) -> None:
        """Ширина переноса берётся из разметки, а не из подобранных долей.

        Раньше здесь были доли от ширины окна: 0.40 для карточки таймера,
        0.78 для строки основания и так далее. Каждая доля — догадка о
        геометрии, и каждая ошибалась по-своему: в них не входили полоса
        прокрутки, зазор между верхними карточками и ширина плашки канала.
        Догадка получалась шире места, текст оставался одной строкой и
        обрезался с обеих сторон сразу — внутри метки он выровнен по
        центру, поэтому лишнее срезалось симметрично.

        Теперь ширину сообщает сам tkinter: он уже разложил карточки и
        знает точные числа. Доли не нужны.
        """

        self._wrap_pending = False
        scaling = self._widget_scaling()

        for item in self._wrapping:
            wrap = wrap_width(
                item.container.winfo_width(),
                occupied=tuple(n.winfo_width() for n in item.neighbours),
                gap=item.gap,
                limit=self._last_content_width,
                scaling=scaling,
            )
            if wrap is None:
                continue  # разметка ещё не построена
            if self._wrap_widths.get(id(item.label)) == wrap:
                continue
            self._wrap_widths[id(item.label)] = wrap
            item.label.configure(wraplength=wrap)

    @staticmethod
    def _widget_scaling() -> float:
        try:
            return float(ctk.ScalingTracker.get_widget_scaling(None))
        except Exception:
            return 1.0

    # ------------------------------------------------------------- поведение
    @staticmethod
    def _call(callback: Callable[[], None] | None) -> None:
        if callback is not None:
            callback()

    def toggle_details(self) -> None:
        self._details_open = not self._details_open
        if self._details_open:
            self.details_panel.pack(fill="x")
            self.details_button.configure(text="Подробнее  ▴")
        else:
            self.details_panel.pack_forget()
            self.details_button.configure(text="Подробнее  ▾")
        if self._on_toggle_details is not None:
            self._on_toggle_details(self._details_open)

    # --------------------------------------------------------------- отрисовка
    def render(self, view: MainScreenView) -> None:
        self.monitor_pill.configure(
            text=view.monitoring_text,
            fg_color=theme.BADGE_ON_BG if view.monitoring else theme.BADGE_OFF_BG,
            text_color=theme.BADGE_ON_TEXT if view.monitoring else theme.BADGE_OFF_TEXT,
        )

        palette = theme.LEVEL_COLORS[view.level]

        self.work_label.configure(text=view.work_label)
        self.work_time_label.configure(text=view.work_time_text)
        self.work_progress.configure(progress_color=palette["accent"])
        self.work_progress.set(max(0.0, min(1.0, view.work_progress)))
        self.next_threshold_label.configure(text=view.next_threshold_text)

        signal_color = theme.SIGNAL_COLORS[view.signal_tone]
        self.signal_value_label.configure(
            text=view.signal_value_text,
            text_color=theme.TEXT if view.signal_percent is not None else signal_color,
        )
        self.signal_bar.render(filled=view.signal_bars, color=signal_color)
        self.signal_hint_label.configure(
            text=view.signal_hint,
            text_color=theme.SIGNAL_HINT_COLORS[view.signal_tone],
        )

        self.level_card.configure(fg_color=palette["bg"])
        self.level_caption.configure(
            text=f"Уровень {view.level} из 5",
            text_color=palette["label"],
        )
        self.level_title.configure(text=view.level_title, text_color=palette["title"])
        self.level_bar.render(filled=view.level, color=palette["accent"])

        self._render_reasons(view)
        self._render_action(view)
        self._render_details(view)
        self.updated_label.configure(text=view.updated_text)

        # Новые тексты меняют ширину плашек каналов, а значит и место,
        # оставшееся строке основания.
        self._schedule_wrapping()

    def _render_reasons(self, view: MainScreenView) -> None:
        for index, (row, pill, text) in enumerate(self._reason_rows):
            if index >= len(view.reasons):
                row.pack_forget()
                continue
            reason = view.reasons[index]
            colors = theme.CHANNEL_COLORS.get(
                reason.channel, theme.CHANNEL_COLORS["ocular"]
            )
            pill.configure(
                text=reason.channel_label,
                fg_color=colors["bg"],
                text_color=colors["text"],
            )
            text.configure(
                text=reason.text,
                text_color=theme.TEXT_DIM if reason.dim else theme.TEXT_SECONDARY,
            )
            row.pack(fill="x", pady=(0, 8))

    def _render_action(self, view: MainScreenView) -> None:
        self.action_title.configure(text=view.action_title)
        self.action_text.configure(text=view.action_text)
        self.break_button.configure(
            text="Закончить перерыв" if view.break_active else "Начать перерыв"
        )
        if view.show_action_buttons and not view.break_active:
            self.snooze_button.pack(side="left", padx=8)
            self.dismiss_button.pack(side="left")
        else:
            self.snooze_button.pack_forget()
            self.dismiss_button.pack_forget()

    def _render_details(self, view: MainScreenView) -> None:
        for index, (card, title, body) in enumerate(self._detail_cards):
            if index >= len(view.details):
                card.pack_forget()
                continue
            block = view.details[index]
            colors = theme.CHANNEL_COLORS.get(
                block.channel, theme.CHANNEL_COLORS["ocular"]
            )
            card.configure(fg_color=theme.CARD_MUTED if block.dim else theme.CARD)
            title.configure(
                text=block.title,
                text_color=theme.TEXT_DIM if block.dim else colors["text"],
            )
            body.configure(
                text=block.text,
                text_color=theme.TEXT_DIM if block.dim else theme.TEXT_SECONDARY,
            )
            card.pack(fill="x", pady=(0, 10))
