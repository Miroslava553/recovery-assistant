"""Диалог самооценки: две независимые шкалы.

Программа не подсказывает пользователю значение и не подставляет его по
показаниям сенсоров. Самооценка должна оставаться независимым каналом.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui.formatting import neighbor_description, self_report_description
from ui.labels import FATIGUE_SP_LABELS, SLEEPINESS_KSS_LABELS


class SelfReportDialog:
    """Две шкалы самооценки с ясными различиями между соседними уровнями."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        initial_fatigue: int | None,
        initial_sleepiness: int | None,
        on_save,
    ) -> None:
        self.on_save = on_save
        self.window = tk.Toplevel(root)
        self.window.title("Как вы себя чувствуете")
        self.window.geometry("760x680")
        self.window.minsize(700, 620)
        self.window.transient(root)
        self.window.grab_set()

        self.fatigue_var = tk.IntVar(value=initial_fatigue or 3)
        self.sleepiness_var = tk.IntVar(value=initial_sleepiness or 5)
        self._fatigue_touched = initial_fatigue is not None
        self._sleepiness_touched = initial_sleepiness is not None

        outer = ttk.Frame(self.window, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="Самооценка состояния",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Выберите описание, которое точнее всего соответствует состоянию сейчас. "
                "Число вторично; ориентируйтесь на различия в формулировках."
            ),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(4, 14))

        self._fatigue_description_var = tk.StringVar()
        self._fatigue_neighbors_var = tk.StringVar()
        self._sleepiness_description_var = tk.StringVar()
        self._sleepiness_neighbors_var = tk.StringVar()

        self._build_scale_section(
            outer,
            title="Общая усталость — Samn–Perelli, 1–7",
            variable=self.fatigue_var,
            minimum=1,
            maximum=7,
            description_var=self._fatigue_description_var,
            neighbors_var=self._fatigue_neighbors_var,
            labels=FATIGUE_SP_LABELS,
            touch_callback=self._mark_fatigue_touched,
        )
        self._build_scale_section(
            outer,
            title="Сонливость — KSS, 1–9",
            variable=self.sleepiness_var,
            minimum=1,
            maximum=9,
            description_var=self._sleepiness_description_var,
            neighbors_var=self._sleepiness_neighbors_var,
            labels=SLEEPINESS_KSS_LABELS,
            touch_callback=self._mark_sleepiness_touched,
        )

        self.notice_var = tk.StringVar(
            value="Перед сохранением выберите значение на обеих шкалах."
        )
        ttk.Label(
            outer,
            textvariable=self.notice_var,
            wraplength=700,
            foreground="#4a4a4a",
        ).pack(anchor="w", pady=(8, 10))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x")
        self.save_button = ttk.Button(
            buttons,
            text="Сохранить самооценку",
            command=self._save,
        )
        self.save_button.pack(side="right")
        ttk.Button(buttons, text="Отмена", command=self.window.destroy).pack(
            side="right", padx=(0, 8)
        )

        self.fatigue_var.trace_add("write", lambda *_: self._refresh_fatigue())
        self.sleepiness_var.trace_add("write", lambda *_: self._refresh_sleepiness())
        self._refresh_fatigue()
        self._refresh_sleepiness()
        self._refresh_save_state()

    def _build_scale_section(
        self,
        parent: ttk.Frame,
        *,
        title: str,
        variable: tk.IntVar,
        minimum: int,
        maximum: int,
        description_var: tk.StringVar,
        neighbors_var: tk.StringVar,
        labels: dict[int, tuple[str, str]],
        touch_callback,
    ) -> None:
        card = ttk.LabelFrame(parent, text=title, padding=12)
        card.pack(fill="x", pady=(0, 12))

        scale = tk.Scale(
            card,
            from_=minimum,
            to=maximum,
            orient="horizontal",
            resolution=1,
            showvalue=True,
            variable=variable,
            length=620,
            tickinterval=1,
            highlightthickness=0,
        )
        scale.pack(fill="x")
        scale.bind("<Button-1>", touch_callback)
        scale.bind("<B1-Motion>", touch_callback)
        scale.bind("<Key>", touch_callback)

        ttk.Label(
            card,
            textvariable=description_var,
            font=("Segoe UI", 10, "bold"),
            wraplength=660,
            justify="left",
        ).pack(anchor="w", pady=(6, 4))
        ttk.Label(
            card,
            textvariable=neighbors_var,
            foreground="#555555",
            wraplength=660,
            justify="left",
        ).pack(anchor="w")

    def _mark_fatigue_touched(self, _event=None) -> None:
        self._fatigue_touched = True
        self.window.after_idle(self._refresh_save_state)

    def _mark_sleepiness_touched(self, _event=None) -> None:
        self._sleepiness_touched = True
        self.window.after_idle(self._refresh_save_state)

    def _refresh_fatigue(self) -> None:
        value = int(self.fatigue_var.get())
        short, description = self_report_description(FATIGUE_SP_LABELS, value)
        self._fatigue_description_var.set(f"{value} — {short}. {description}")
        self._fatigue_neighbors_var.set(neighbor_description(FATIGUE_SP_LABELS, value))

    def _refresh_sleepiness(self) -> None:
        value = int(self.sleepiness_var.get())
        short, description = self_report_description(SLEEPINESS_KSS_LABELS, value)
        self._sleepiness_description_var.set(f"{value} — {short}. {description}")
        self._sleepiness_neighbors_var.set(neighbor_description(SLEEPINESS_KSS_LABELS, value))

    def _refresh_save_state(self) -> None:
        ready = self._fatigue_touched and self._sleepiness_touched
        self.save_button.configure(state="normal" if ready else "disabled")
        if ready:
            self.notice_var.set(
                "Самооценка будет отдельным каналом и не заменит данные рабочей сессии, печати или камеры."
            )

    def _save(self) -> None:
        if not (self._fatigue_touched and self._sleepiness_touched):
            return
        self.on_save(int(self.fatigue_var.get()), int(self.sleepiness_var.get()))
        self.window.destroy()
