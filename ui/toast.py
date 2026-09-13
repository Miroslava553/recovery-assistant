"""Неблокирующее уведомление о рекомендации."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class RecommendationToast:
    """Неблокирующее всплывающее уведомление поверх других окон."""

    WIDTH = 470

    def __init__(
        self,
        root: tk.Tk,
        *,
        title: str,
        message: str,
        duration_text: str,
        demo_mode: bool,
        on_accept,
        on_snooze,
        on_irrelevant,
        on_close,
    ) -> None:
        self._on_close = on_close
        self.window = tk.Toplevel(root)
        self.window.title("Ассистент восстановления")
        self.window.resizable(False, False)
        self.window.attributes("-topmost", True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        background = "#fff6dd" if demo_mode else "#eef4fb"
        accent = "#6b4f00" if demo_mode else "#173b5e"
        container = tk.Frame(
            self.window,
            background=background,
            padx=18,
            pady=16,
            width=self.WIDTH,
        )
        container.pack(fill="both", expand=True)

        eyebrow = "ДЕМОНСТРАЦИЯ" if demo_mode else "РЕКОМЕНДАЦИЯ"
        tk.Label(
            container,
            text=eyebrow,
            background=background,
            foreground=accent,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            container,
            text=title,
            background=background,
            foreground="#111111",
            font=("Segoe UI", 14, "bold"),
            justify="left",
            anchor="w",
            wraplength=self.WIDTH - 36,
        ).pack(fill="x", pady=(5, 6))
        tk.Label(
            container,
            text=message,
            background=background,
            foreground="#222222",
            font=("Segoe UI", 10),
            justify="left",
            anchor="w",
            wraplength=self.WIDTH - 36,
        ).pack(fill="x")
        tk.Label(
            container,
            text=f"Рекомендуемая длительность: {duration_text}",
            background=background,
            foreground="#333333",
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", pady=(10, 12))

        buttons = tk.Frame(container, background=background)
        buttons.pack(fill="x")
        ttk.Button(
            buttons,
            text="Начать перерыв",
            command=lambda: self._run_and_close(on_accept),
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="Отложить 10 минут",
            command=lambda: self._run_and_close(on_snooze),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons,
            text="Неуместно",
            command=lambda: self._run_and_close(on_irrelevant),
        ).pack(side="left", padx=(8, 0))

        self.window.update_idletasks()
        width = max(self.WIDTH, self.window.winfo_reqwidth())
        height = self.window.winfo_reqheight()
        x = max(0, self.window.winfo_screenwidth() - width - 24)
        y = max(0, self.window.winfo_screenheight() - height - 72)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.lift()

    def _run_and_close(self, callback) -> None:
        try:
            callback()
        finally:
            self.close()

    def close(self) -> None:
        window = self.window
        if window is None:
            return
        self.window = None
        try:
            if window.winfo_exists():
                window.destroy()
        except tk.TclError:
            pass
        callback = self._on_close
        self._on_close = None
        if callback is not None:
            callback()
