"""Приложение ассистента восстановления на новом экране.

Окно ничего не решает само: оно опрашивает `SessionMonitor`, отдаёт сигналы
`AdaptiveRecoveryEngine`, переводит его ответ в `MainScreenView` через
`ui.presenter` и рисует. Вся логика остаётся в `app_core` и проверяется
тестами.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import customtkinter as ctk
import cv2
import numpy as np

try:
    import winsound
except ImportError:  # не Windows
    winsound = None

from app_core.recovery_engine import (
    AdaptiveRecoveryEngine,
    RecoveryAssessment,
    RecoveryRecommendation,
    RecoverySignals,
    build_recovery_signals,
)
from app_core.recovery_storage import RecoveryEventStore
from app_core.session_monitor import SessionMonitor
from app_core.state_machine import UserStateManager
from app_core.storage import AppDatabase
from app_core.typing_baseline import TypingBaselineService
from ui import theme
from ui.formatting import format_duration
from ui.main_screen import MainScreen
from ui.presenter import build_main_screen_view
from ui.self_report import SelfReportDialog
from ui.toast import RecommendationToast

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "fatigue_assistant.sqlite3"


class RecoveryAssistantApp:
    ASSESSMENT_POLL_MS = 1000
    PREVIEW_POLL_MS = 100

    def __init__(self, root: ctk.CTk, *, demo_mode: bool) -> None:
        self.root = root
        self.demo_mode = bool(demo_mode)
        self.root.title("Ассистент восстановления")
        self.root.geometry("820x900")
        self.root.minsize(620, 700)
        self.root.configure(fg_color=theme.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        database = AppDatabase(DATABASE_PATH)
        self.profile_id = database.get_or_create_profile("Основной пользователь")
        self.typing_baseline = TypingBaselineService(
            DATABASE_PATH,
            profile_id=self.profile_id,
            sample_interval_sec=60.0,
            min_observation_sec=50.0,
            target_relevant_keys=25,
        )
        # В демо-режиме защитный период после перерыва тоже ускорен, иначе
        # проверка интерфейса упирается в минуту неподвижных счётчиков.
        self.returning_duration_sec = 6.0 if self.demo_mode else 60.0
        self.monitor = SessionMonitor(
            typing_baseline_service=self.typing_baseline,
            target_visual_fps=20.0,
            include_diagnostic_frame=False,
            vision_required=False,
            state_manager=UserStateManager(
                returning_duration_sec=self.returning_duration_sec
            ),
        )
        self.store = RecoveryEventStore(DATABASE_PATH, profile_id=self.profile_id)

        if self.demo_mode:
            self.engine = AdaptiveRecoveryEngine(
                eye_rest_after_sec=45.0,
                microbreak_after_sec=70.0,
                recovery_break_after_sec=105.0,
                meaningful_break_sec=15.0,
                alert_cooldown_sec=30.0,
                typing_evaluation_interval_sec=5.0,
                ocular_decisions_enabled=False,
                level_rise_confirmations=3,
                level_drop_confirmations=6,
            )
        else:
            self.engine = AdaptiveRecoveryEngine(ocular_decisions_enabled=False)

        self._monitor_running = False
        self._manual_break = False
        self._closing = False
        self._last_assessment: RecoveryAssessment | None = None
        self._notification: RecommendationToast | None = None
        self._notification_recommendation_id: int | None = None

        self._camera_photo: tk.PhotoImage | None = None
        self._last_preview_captured_at: float | None = None
        self._technical_window: tk.Toplevel | None = None
        self._technical_text: tk.Text | None = None

        self.screen = MainScreen(
            self.root,
            on_start_break=self._toggle_break,
            on_snooze=self._snooze,
            on_dismiss=self._mark_irrelevant,
            on_self_report=self._self_report,
            on_technical=self._toggle_technical,
            on_monitoring_click=self._toggle_camera_preview,
        )
        self.screen.pack(fill="both", expand=True, padx=10, pady=10)

        self._start_monitor()
        self.root.after(self.ASSESSMENT_POLL_MS, self._poll)
        self.root.after(self.PREVIEW_POLL_MS, self._poll_camera_preview)

    # ------------------------------------------------------------------ запуск
    def _start_monitor(self) -> None:
        if self._monitor_running:
            return
        try:
            self.monitor.start()
        except Exception as error:
            messagebox.showerror(
                "Не удалось запустить мониторинг",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return

        self._monitor_running = True
        if not self.monitor.vision_available:
            messagebox.showwarning(
                "Камера недоступна",
                "Визуальный канал отключён, остальные продолжают работать.\n\n"
                f"Причина: {self.monitor.vision_unavailable_reason}",
                parent=self.root,
            )
        self.store.log_event(
            "monitor_started",
            payload={
                "demo_mode": self.demo_mode,
                "vision_available": self.monitor.vision_available,
            },
        )

    # -------------------------------------------------------------------- цикл
    def _poll(self) -> None:
        if self._closing:
            return
        if self._monitor_running:
            try:
                snapshot = self.monitor.snapshot()
                signals = build_recovery_signals(snapshot, self.typing_baseline)
                assessment = self.engine.update(signals)
                self._last_assessment = assessment
                self._render(snapshot, signals, assessment)
                if assessment.recommendation_is_new and assessment.recommendation is not None:
                    self._log_recommendation(assessment.recommendation)
                    self._show_recommendation_notification(assessment.recommendation)
            except Exception as error:
                self._render_sensor_error(error)
        self.root.after(self.ASSESSMENT_POLL_MS, self._poll)

    def _render(
        self,
        snapshot,
        signals: RecoverySignals,
        assessment: RecoveryAssessment,
    ) -> None:
        view = build_main_screen_view(
            snapshot,
            signals,
            assessment,
            now=time.monotonic(),
            monitoring=self._monitor_running,
            vision_available=self.monitor.vision_available,
            break_active=self._manual_break,
            meaningful_break_sec=self.engine.meaningful_break_sec,
            returning_duration_sec=self.returning_duration_sec,
            eye_rest_after_sec=self.engine.eye_rest_after_sec,
            microbreak_after_sec=self.engine.microbreak_after_sec,
            recovery_break_after_sec=self.engine.recovery_break_after_sec,
        )
        self.screen.render(view)
        if self._technical_text is not None:
            self._refresh_technical(snapshot, signals, assessment)

    def _render_sensor_error(self, error: Exception) -> None:
        """Ошибка датчика не должна выглядеть как вывод об усталости."""

        from ui.view_model import MainScreenView

        self.screen.render(
            MainScreenView(
                monitoring=False,
                monitoring_text="Ошибка датчика",
                work_time_text="--:--:--",
                next_threshold_text="оценка приостановлена",
                signal_value_text="нет",
                signal_tone="off",
                signal_hint="данные недоступны",
                level=1,
                level_title="Оценка временно недоступна",
                action_title="Рекомендация недоступна",
                action_text=f"{type(error).__name__}: {error}",
                updated_text="обновлено сейчас",
            )
        )

    # ------------------------------------------------------------ уведомление
    def _show_recommendation_notification(
        self, recommendation: RecoveryRecommendation
    ) -> None:
        if self._notification_recommendation_id == recommendation.recommendation_id:
            return

        self._close_recommendation_notification()
        if self.demo_mode:
            title = "ДЕМО: проверка уведомления"
            message = (
                "Сработал ускоренный временной порог. Это не вывод об усталости "
                "и не результат анализа глаз или печати."
            )
        else:
            title = recommendation.title
            message = recommendation.message

        self._notification = RecommendationToast(
            self.root,
            title=title,
            message=message,
            duration_text=format_duration(recommendation.suggested_break_sec),
            demo_mode=self.demo_mode,
            on_accept=self._start_break,
            on_snooze=self._snooze,
            on_irrelevant=self._mark_irrelevant,
            on_close=self._notification_closed,
        )
        self._notification_recommendation_id = recommendation.recommendation_id
        self._play_notification_sound()

    def _play_notification_sound(self) -> None:
        if winsound is None:
            return
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    def _notification_closed(self) -> None:
        self._notification = None
        self._notification_recommendation_id = None

    def _close_recommendation_notification(self) -> None:
        toast = self._notification
        self._notification = None
        self._notification_recommendation_id = None
        if toast is not None:
            try:
                toast.close()
            except Exception:
                pass

    # ------------------------------------------------------------- действия
    def _toggle_break(self) -> None:
        if self._manual_break:
            self._finish_break()
        else:
            self._start_break()

    def _start_break(self) -> None:
        if not self._monitor_running or self._manual_break:
            return
        recommendation = self.engine.current_recommendation
        self._manual_break = True
        self.monitor.set_manual_break(True)
        resolved = self.engine.resolve_recommendation(captured_at=time.monotonic())
        self.engine.begin_manual_break()
        self.store.log_event(
            "break_started",
            recommendation_id=(resolved.recommendation_id if resolved else None),
            payload={
                "suggested_break_sec": (
                    recommendation.suggested_break_sec if recommendation else None
                )
            },
        )
        self._close_recommendation_notification()

    def _finish_break(self) -> None:
        if not self._monitor_running or not self._manual_break:
            return
        self._manual_break = False
        self.monitor.set_manual_break(False)
        self.engine.end_manual_break()
        self.store.log_event("break_finished")

    def _snooze(self) -> None:
        recommendation = self.engine.current_recommendation
        if recommendation is None:
            self._close_recommendation_notification()
            return
        self.engine.snooze(minutes=10.0, captured_at=time.monotonic())
        self.store.log_event(
            "recommendation_snoozed",
            recommendation_id=recommendation.recommendation_id,
            payload={"minutes": 10},
        )
        self._close_recommendation_notification()

    def _mark_irrelevant(self) -> None:
        recommendation = self.engine.resolve_recommendation(
            captured_at=time.monotonic(),
            longer_cooldown=True,
        )
        if recommendation is not None:
            self.store.log_event(
                "recommendation_irrelevant",
                recommendation_id=recommendation.recommendation_id,
            )
        self._close_recommendation_notification()

    def _self_report(self) -> None:
        assessment = self._last_assessment
        SelfReportDialog(
            self.root,
            initial_fatigue=assessment.self_report_fatigue_sp if assessment else None,
            initial_sleepiness=assessment.self_report_sleepiness_kss if assessment else None,
            on_save=self._save_self_report,
        )

    def _save_self_report(self, fatigue_sp: int, sleepiness_kss: int) -> None:
        self.engine.record_self_report(
            fatigue_sp_1_7=float(fatigue_sp),
            sleepiness_kss_1_9=float(sleepiness_kss),
            captured_at=time.monotonic(),
        )
        self.store.add_self_report(
            fatigue_sp_1_7=float(fatigue_sp),
            sleepiness_kss_1_9=float(sleepiness_kss),
        )
        self.store.log_event(
            "self_report_added",
            payload={
                "fatigue_sp_1_7": fatigue_sp,
                "sleepiness_kss_1_9": sleepiness_kss,
            },
        )

    def _log_recommendation(self, recommendation: RecoveryRecommendation) -> None:
        self.store.log_event(
            "recommendation_created",
            recommendation_id=recommendation.recommendation_id,
            payload={
                "kind": recommendation.kind.value,
                "title": recommendation.title,
                "suggested_break_sec": recommendation.suggested_break_sec,
                "reason_codes": list(recommendation.reason_codes),
                "demo_mode": self.demo_mode,
            },
        )

    # -------------------------------------------------- технические показатели
    def _toggle_technical(self) -> None:
        if self._technical_window is not None:
            try:
                if self._technical_window.winfo_exists():
                    self._close_technical()
                    return
            except tk.TclError:
                pass

        window = tk.Toplevel(self.root)
        window.title("Технические показатели")
        window.geometry("640x560")
        window.protocol("WM_DELETE_WINDOW", self._close_technical)

        text = tk.Text(
            window,
            wrap="word",
            font=("Consolas", 10),
            background="#1C1C1A",
            foreground="#D3D1C7",
            insertbackground="#D3D1C7",
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self._technical_window = window
        self._technical_text = text

    def _close_technical(self) -> None:
        window = self._technical_window
        self._technical_window = None
        self._technical_text = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass

    def _refresh_technical(self, snapshot, signals, assessment) -> None:
        text = self._technical_text
        if text is None:
            return
        lines: list[str] = []

        lines.append("=== РАБОЧИЙ КОНТЕКСТ ===")
        lines.append(f"состояние: {snapshot.state_label}")
        lines.append(f"причина: {snapshot.state_reason}")
        lines.append(f"в состоянии, с: {snapshot.seconds_in_state:.1f}")
        lines.append(f"без ввода, с: {snapshot.input_idle_sec}")
        lines.append(f"анализ разрешён: {snapshot.fatigue_analysis_allowed}")

        lines.append("")
        lines.append("=== КАМЕРА И ГЛАЗА ===")
        lines.append(f"камера: {snapshot.camera_status}")
        lines.append(f"внимание: {snapshot.attention_status}")
        lines.append(f"глаза: {snapshot.ocular_status}")
        lines.append(f"лицо в кадре: {snapshot.face_detected}")
        lines.append(f"взгляд к экрану: {snapshot.gaze_on_screen}")
        lines.append(f"возраст кадра, с: {snapshot.visual_age_sec:.2f}")
        for key, value in sorted(vars(snapshot.ocular_metrics).items()):
            lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append("=== ПЕЧАТЬ ===")
        for key, value in sorted(vars(snapshot.typing_metrics).items()):
            lines.append(f"  {key}: {value}")
        lines.append(f"личная норма готова: {signals.typing_baseline_ready}")
        lines.append(f"калибровка: {signals.typing_calibration_progress:.0%}")
        for name, deviation in sorted(signals.typing_deviations.items()):
            lines.append(
                f"  {name}: z={deviation.robust_z}, готов={deviation.ready}, "
                f"медиана={deviation.baseline_median}"
            )

        lines.append("")
        lines.append("=== РЕШЕНИЕ ===")
        lines.append(f"уровень: {int(assessment.workload_level)} — {assessment.workload_title}")
        lines.append(f"надёжность: {assessment.reliability_label}")
        lines.append(f"основание: {assessment.decision_basis}")
        lines.append(f"глаза влияют на решение: {assessment.ocular_used_for_decision}")
        for item in assessment.evidence:
            mark = "+" if item.decision_ready else "·"
            lines.append(f"  {mark} [{item.source}] {item.code} (сила {item.severity})")
        for key, value in sorted(
            self.engine.diagnostics(captured_at=time.monotonic()).items()
        ):
            lines.append(f"  {key}: {value}")

        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    # ------------------------------------------------------------ превью камеры
    def _toggle_camera_preview(self) -> None:
        if self.screen.camera_visible:
            self.screen.hide_camera()
            self._last_preview_captured_at = None
            return

        if not self.monitor.vision_available:
            messagebox.showinfo(
                "Камера недоступна",
                "Визуальный канал сейчас отключён, показывать нечего.",
                parent=self.root,
            )
            return

        self._last_preview_captured_at = None
        self.screen.show_camera()

    def _poll_camera_preview(self) -> None:
        if self._closing:
            return
        if self._monitor_running and self.screen.camera_visible:
            try:
                captured_at, frame = self.monitor.latest_visual_frame()
                if (
                    frame is not None
                    and captured_at is not None
                    and captured_at != self._last_preview_captured_at
                ):
                    canvas = self.screen.camera_canvas
                    self._camera_photo = self._frame_to_photo(frame)
                    canvas.delete("all")
                    canvas.create_image(
                        self.screen.CAMERA_WIDTH // 2,
                        self.screen.CAMERA_HEIGHT // 2,
                        image=self._camera_photo,
                        anchor="center",
                    )
                    self._last_preview_captured_at = captured_at
            except tk.TclError:
                pass
            except Exception:
                pass
        self.root.after(self.PREVIEW_POLL_MS, self._poll_camera_preview)

    def _frame_to_photo(self, frame) -> tk.PhotoImage:
        # Превью квадратное: берём центральный квадрат кадра, иначе широкая
        # картинка оставляла бы пустые поля по бокам.
        height, width = frame.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        frame = frame[top : top + side, left : left + side]
        height, width = frame.shape[:2]
        width_px = self.screen.CAMERA_WIDTH
        height_px = self.screen.CAMERA_HEIGHT
        scale = min(width_px / width, height_px / height)
        target_w = max(1, int(width * scale))
        target_h = max(1, int(height * scale))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((height_px, width_px, 3), 36, dtype=np.uint8)
        x0 = (width_px - target_w) // 2
        y0 = (height_px - target_h) // 2
        canvas[y0 : y0 + target_h, x0 : x0 + target_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        header = f"P6\n{width_px} {height_px}\n255\n".encode("ascii")
        return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")

    # ------------------------------------------------------------- завершение
    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._close_recommendation_notification()
        self._close_technical()
        try:
            self.monitor.close()
        except Exception:
            pass
        try:
            self.store.log_event("application_closed")
        except Exception:
            pass
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ассистент восстановления: локальная оценка рабочей нагрузки."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Ускоренные временные пороги для проверки интерфейса. "
            "Уведомление в этом режиме не является выводом об усталости."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    RecoveryAssistantApp(root, demo_mode=args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()
