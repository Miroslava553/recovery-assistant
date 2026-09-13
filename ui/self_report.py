"""Диалог самооценки: две независимые шкалы.

Программа не подсказывает пользователю значение и не подставляет его по
показаниям сенсоров. Самооценка должна оставаться независимым каналом:
если бы окно открывалось с уже выставленной «подходящей» оценкой, человек
соглашался бы с догадкой программы, и канал перестал бы быть независимым
источником. Поэтому сохранение недоступно, пока обе шкалы не тронуты
рукой.

Формулировки делений берутся из `ui.labels` и здесь не меняются: границы
соседних значений согласованы с пользователем отдельно.
"""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from ui import theme
from ui.formatting import neighbor_description, self_report_description, wrap_width
from ui.labels import FATIGUE_SP_LABELS, SLEEPINESS_KSS_LABELS


def _font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=theme.FONT_FAMILY, size=size, weight=weight)


class SelfReportDialog:
    """Две шкалы самооценки с ясными различиями между соседними делениями."""

    def __init__(
        self,
        root: tk.Misc,
        *,
        initial_fatigue: int | None,
        initial_sleepiness: int | None,
        on_save,
    ) -> None:
        self.on_save = on_save
        self.window = ctk.CTkToplevel(root)
        self.window.title("Как вы себя чувствуете")
        self.window.geometry("720x820")
        self.window.minsize(560, 560)
        self.window.configure(fg_color=theme.BG)
        self.window.transient(root)

        # Захват ввода ставим с задержкой: CTkToplevel досоздаёт себя уже
        # после возврата из конструктора, и немедленный grab_set на части
        # систем срывается.
        self.window.after(120, self._grab)

        self.fatigue_var = tk.IntVar(value=initial_fatigue or 3)
        self.sleepiness_var = tk.IntVar(value=initial_sleepiness or 5)
        self._fatigue_touched = initial_fatigue is not None
        self._sleepiness_touched = initial_sleepiness is not None

        self._wrapping: list[tuple[ctk.CTkLabel, ctk.CTkBaseClass]] = []
        self._wrap_widths: dict[int, int] = {}
        self._wrap_pending = False
        self._last_width = 0

        self._build()

        self.fatigue_var.trace_add("write", lambda *_: self._refresh_fatigue())
        self.sleepiness_var.trace_add("write", lambda *_: self._refresh_sleepiness())
        self._refresh_fatigue()
        self._refresh_sleepiness()
        self._refresh_save_state()

    # --- построение окна -------------------------------------------------

    def _build(self) -> None:
        body = ctk.CTkScrollableFrame(self.window, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=(18, 0))

        ctk.CTkLabel(
            body,
            text="Самооценка состояния",
            font=_font(20, "bold"),
            text_color=theme.TEXT,
            anchor="w",
        ).pack(fill="x")

        intro = ctk.CTkLabel(
            body,
            text=(
                "Выберите описание, которое точнее всего соответствует состоянию "
                "сейчас. Число вторично; ориентируйтесь на различия в формулировках."
            ),
            font=_font(13),
            text_color=theme.TEXT_SECONDARY,
            anchor="w",
            justify="left",
        )
        intro.pack(fill="x", pady=(6, 16))
        self._wrapping.append((intro, body))

        self._fatigue = self._build_scale_section(
            body,
            title="Общая усталость — Samn–Perelli, 1–7",
            variable=self.fatigue_var,
            minimum=1,
            maximum=7,
            touch_callback=self._mark_fatigue_touched,
        )
        self._sleepiness = self._build_scale_section(
            body,
            title="Сонливость — KSS, 1–9",
            variable=self.sleepiness_var,
            minimum=1,
            maximum=9,
            touch_callback=self._mark_sleepiness_touched,
        )

        footer = ctk.CTkFrame(self.window, fg_color="transparent")
        footer.pack(fill="x", padx=18, pady=(12, 18))

        self.notice_label = ctk.CTkLabel(
            footer,
            text="Перед сохранением выберите значение на обеих шкалах.",
            font=_font(12),
            text_color=theme.TEXT_DIM,
            anchor="w",
            justify="left",
        )
        self.notice_label.pack(fill="x", pady=(0, 12))
        self._wrapping.append((self.notice_label, footer))

        buttons = ctk.CTkFrame(footer, fg_color="transparent")
        buttons.pack(fill="x")

        self.save_button = ctk.CTkButton(
            buttons,
            text="Сохранить самооценку",
            font=_font(13, "bold"),
            corner_radius=theme.RADIUS_BUTTON,
            height=36,
            fg_color=theme.ACTION_PRIMARY,
            hover_color=theme.ACTION_PRIMARY_HOVER,
            text_color=theme.ACTION_PRIMARY_TEXT,
            text_color_disabled=theme.TEXT_FAINT,
            command=self._save,
        )
        self.save_button.pack(side="right")

        ctk.CTkButton(
            buttons,
            text="Отмена",
            font=_font(13),
            corner_radius=theme.RADIUS_BUTTON,
            height=36,
            fg_color="transparent",
            hover_color=theme.CARD,
            text_color=theme.TEXT_SECONDARY,
            border_width=1,
            border_color=theme.BORDER,
            command=self.close,
        ).pack(side="right", padx=(0, 10))

        self.window.bind("<Configure>", self._on_resize)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def _build_scale_section(
        self,
        parent: ctk.CTkBaseClass,
        *,
        title: str,
        variable: tk.IntVar,
        minimum: int,
        maximum: int,
        touch_callback,
    ) -> dict:
        card = ctk.CTkFrame(
            parent,
            fg_color=theme.CARD,
            corner_radius=theme.RADIUS_CARD,
        )
        card.pack(fill="x", pady=(0, 14))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        ctk.CTkLabel(
            inner,
            text=title,
            font=_font(12),
            text_color=theme.TEXT_DIM,
            anchor="w",
        ).pack(fill="x")

        slider = ctk.CTkSlider(
            inner,
            from_=minimum,
            to=maximum,
            number_of_steps=maximum - minimum,
            variable=variable,
            height=18,
            progress_color=theme.ACTION_PRIMARY,
            button_color=theme.ACTION_PRIMARY,
            button_hover_color=theme.ACTION_PRIMARY_HOVER,
            fg_color=theme.BORDER,
        )
        slider.pack(fill="x", pady=(12, 6))
        # Ползунок двигают и мышью, и стрелками. Отмечаем оба способа: иначе
        # можно было бы сохранить значение, которого человек не выбирал.
        for sequence in ("<Button-1>", "<B1-Motion>", "<Key>"):
            slider.bind(sequence, touch_callback)

        # Деления подписываем сами: CTkSlider своих подписей не рисует, а без
        # них шкала перестаёт быть счётной и превращается в «примерно левее».
        ticks = ctk.CTkFrame(inner, fg_color="transparent")
        ticks.pack(fill="x")
        for column in range(maximum - minimum + 1):
            ticks.grid_columnconfigure(column, weight=1, uniform="ticks")
            ctk.CTkLabel(
                ticks,
                text=str(minimum + column),
                font=_font(11),
                text_color=theme.TEXT_FAINT,
            ).grid(row=0, column=column, sticky="ew")

        description = ctk.CTkLabel(
            inner,
            text="",
            font=_font(14, "bold"),
            text_color=theme.TEXT,
            anchor="w",
            justify="left",
        )
        description.pack(fill="x", pady=(12, 5))
        self._wrapping.append((description, inner))

        neighbours = ctk.CTkLabel(
            inner,
            text="",
            font=_font(12),
            text_color=theme.TEXT_DIM,
            anchor="w",
            justify="left",
        )
        neighbours.pack(fill="x")
        self._wrapping.append((neighbours, inner))

        return {"description": description, "neighbours": neighbours}

    # --- перенос текста ---------------------------------------------------

    def _on_resize(self, event) -> None:
        if abs(event.width - self._last_width) < 4:
            return
        self._last_width = event.width
        self._schedule_wrapping()

    def _schedule_wrapping(self) -> None:
        if self._wrap_pending or self.window is None:
            return
        self._wrap_pending = True
        try:
            self.window.after_idle(self._apply_wrapping)
        except Exception:
            self._wrap_pending = False

    def _apply_wrapping(self) -> None:
        self._wrap_pending = False
        try:
            scaling = float(ctk.ScalingTracker.get_widget_scaling(None))
        except Exception:
            scaling = 1.0

        for label, container in self._wrapping:
            try:
                wrap = wrap_width(container.winfo_width(), scaling=scaling)
            except tk.TclError:
                return  # окно уже закрыто
            if wrap is None or self._wrap_widths.get(id(label)) == wrap:
                continue
            self._wrap_widths[id(label)] = wrap
            label.configure(wraplength=wrap)

    # --- состояние --------------------------------------------------------

    def _grab(self) -> None:
        try:
            if self.window is not None and self.window.winfo_exists():
                self.window.grab_set()
                self.window.focus_force()
        except tk.TclError:
            pass

    def _mark_fatigue_touched(self, _event=None) -> None:
        self._fatigue_touched = True
        self.window.after_idle(self._refresh_save_state)

    def _mark_sleepiness_touched(self, _event=None) -> None:
        self._sleepiness_touched = True
        self.window.after_idle(self._refresh_save_state)

    def _refresh_fatigue(self) -> None:
        self._refresh_scale(self._fatigue, self.fatigue_var, FATIGUE_SP_LABELS)

    def _refresh_sleepiness(self) -> None:
        self._refresh_scale(self._sleepiness, self.sleepiness_var, SLEEPINESS_KSS_LABELS)

    def _refresh_scale(self, section: dict, variable: tk.IntVar, labels) -> None:
        value = int(round(float(variable.get())))
        short, description = self_report_description(labels, value)
        section["description"].configure(text=f"{value} — {short}. {description}")
        section["neighbours"].configure(text=neighbor_description(labels, value))
        self._schedule_wrapping()

    def _refresh_save_state(self) -> None:
        ready = self._fatigue_touched and self._sleepiness_touched
        self.save_button.configure(state="normal" if ready else "disabled")
        if ready:
            self.notice_label.configure(
                text=(
                    "Самооценка будет отдельным каналом и не заменит данные "
                    "рабочей сессии, печати или камеры."
                )
            )

    # --- завершение -------------------------------------------------------

    def _save(self) -> None:
        if not (self._fatigue_touched and self._sleepiness_touched):
            return
        fatigue = int(round(float(self.fatigue_var.get())))
        sleepiness = int(round(float(self.sleepiness_var.get())))
        self.on_save(fatigue, sleepiness)
        self.close()

    def close(self) -> None:
        window = self.window
        if window is None:
            return
        self.window = None
        try:
            if window.winfo_exists():
                window.grab_release()
                window.destroy()
        except tk.TclError:
            pass
