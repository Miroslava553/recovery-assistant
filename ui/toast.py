"""Неблокирующее уведомление о рекомендации."""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from ui import theme


def _font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=theme.FONT_FAMILY, size=size, weight=weight)


class RecommendationToast:
    """Всплывающее уведомление поверх других окон.

    Окно намеренно не перехватывает ввод: рекомендация — это предложение, а
    не требование, и работу она прерывать не должна. Человек может её просто
    не заметить, и это допустимо.
    """

    WIDTH = 470

    def __init__(
        self,
        root: tk.Misc,
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
        self.window = ctk.CTkToplevel(root)
        self.window.title("Ассистент восстановления")
        self.window.resizable(False, False)
        self.window.configure(fg_color=theme.BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(80, self._raise_above_others)

        # В демо-режиме уведомление выглядит иначе намеренно: показанный на
        # ускоренных порогах совет не является выводом об усталости, и по виду
        # окна это должно быть понятно без объяснений.
        if demo_mode:
            accent = theme.LEVEL_COLORS[4]["accent"]
            border = theme.CAMERA_BORDER
            eyebrow = "ДЕМОНСТРАЦИЯ"
        else:
            accent = theme.ACTION_LABEL
            border = theme.ACTION_BORDER
            eyebrow = "РЕКОМЕНДАЦИЯ"

        card = ctk.CTkFrame(
            self.window,
            fg_color=theme.CARD,
            corner_radius=theme.RADIUS_CARD,
            border_width=1,
            border_color=border,
        )
        card.pack(fill="both", expand=True, padx=8, pady=8)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        wrap = self.WIDTH - 60

        ctk.CTkLabel(
            inner,
            text=eyebrow,
            font=_font(11, "bold"),
            text_color=accent,
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            inner,
            text=title,
            font=_font(16, "bold"),
            text_color=theme.TEXT,
            anchor="w",
            justify="left",
            wraplength=wrap,
        ).pack(fill="x", pady=(6, 6))

        ctk.CTkLabel(
            inner,
            text=message,
            font=_font(13),
            text_color=theme.TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=wrap,
        ).pack(fill="x")

        ctk.CTkLabel(
            inner,
            text=f"Рекомендуемая длительность: {duration_text}",
            font=_font(12),
            text_color=theme.TEXT_DIM,
            anchor="w",
            justify="left",
            wraplength=wrap,
        ).pack(fill="x", pady=(10, 14))

        buttons = ctk.CTkFrame(inner, fg_color="transparent")
        buttons.pack(fill="x")

        ctk.CTkButton(
            buttons,
            text="Начать перерыв",
            font=_font(13, "bold"),
            corner_radius=theme.RADIUS_BUTTON,
            height=34,
            fg_color=theme.ACTION_PRIMARY,
            hover_color=theme.ACTION_PRIMARY_HOVER,
            text_color=theme.ACTION_PRIMARY_TEXT,
            command=lambda: self._run_and_close(on_accept),
        ).pack(side="left")

        for text, callback in (
            ("Отложить 10 минут", on_snooze),
            ("Неуместно", on_irrelevant),
        ):
            ctk.CTkButton(
                buttons,
                text=text,
                font=_font(13),
                corner_radius=theme.RADIUS_BUTTON,
                height=34,
                fg_color="transparent",
                hover_color=theme.CARD_MUTED,
                text_color=theme.TEXT_SECONDARY,
                border_width=1,
                border_color=theme.BORDER,
                command=lambda cb=callback: self._run_and_close(cb),
            ).pack(side="left", padx=(8, 0))

        self._place_bottom_right()

    def _place_bottom_right(self) -> None:
        self.window.update_idletasks()
        width = max(self.WIDTH, self.window.winfo_reqwidth())
        height = self.window.winfo_reqheight()
        x = max(0, self.window.winfo_screenwidth() - width - 24)
        y = max(0, self.window.winfo_screenheight() - height - 72)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def _raise_above_others(self) -> None:
        """Поднять окно, не удерживая его поверх всего навсегда.

        Постоянный `-topmost` мешал бы работе: уведомление закрывало бы то,
        над чем человек как раз работает. Поднимаем один раз при появлении.
        """

        try:
            if self.window is None or not self.window.winfo_exists():
                return
            self.window.attributes("-topmost", True)
            self.window.lift()
            self.window.after(400, self._release_topmost)
        except tk.TclError:
            pass

    def _release_topmost(self) -> None:
        try:
            if self.window is not None and self.window.winfo_exists():
                self.window.attributes("-topmost", False)
        except tk.TclError:
            pass

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
