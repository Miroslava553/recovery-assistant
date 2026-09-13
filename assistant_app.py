from __future__ import annotations

import argparse
import time
import tkinter as tk
from dataclasses import fields, is_dataclass
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

try:
    import winsound
except ImportError:  # pragma: no cover - Windows-specific helper.
    winsound = None

import cv2
import numpy as np

from app_core.recovery_engine import (
    AdaptiveRecoveryEngine,
    AssessmentReliability,
    EvidenceSource,
    RecoveryAssessment,
    RecoveryRecommendation,
    RecoverySignals,
    WorkloadLevel,
    build_recovery_signals,
)
from app_core.recovery_storage import RecoveryEventStore
from app_core.session_monitor import SessionMonitor
from app_core.state_machine import UserState
from app_core.storage import AppDatabase
from app_core.typing_baseline import TypingBaselineService
from ui.formatting import (
    format_assessment_age,
    format_duration,
    format_optional,
    neighbor_description,
    recommended_action,
    self_report_description,
    user_facing_summary,
)
from ui.labels import (
    EVIDENCE_SOURCE_LABELS,
    FATIGUE_SP_LABELS,
    SLEEPINESS_KSS_LABELS,
    STATE_USER_LABELS,
)
from ui.self_report import SelfReportDialog
from ui.toast import RecommendationToast


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "fatigue_assistant.sqlite3"


class RecoveryAssistantApp:
    ASSESSMENT_POLL_MS = 1000
    PREVIEW_POLL_MS = 100
    PREVIEW_WIDTH = 520
    PREVIEW_HEIGHT = 390

    def __init__(self, root: tk.Tk, *, demo_mode: bool) -> None:
        self.root = root
        self.demo_mode = bool(demo_mode)
        self.root.title("Ассистент восстановления")
        self.root.geometry("960x690")
        self.root.minsize(880, 650)
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
        self.monitor = SessionMonitor(
            typing_baseline_service=self.typing_baseline,
            target_visual_fps=20.0,
            include_diagnostic_frame=False,
            # Basic mode: занятая или отсутствующая камера не должна мешать
            # работе остальных каналов.
            vision_required=False,
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
        self._last_assessment: RecoveryAssessment | None = None
        self._last_signals: RecoverySignals | None = None
        self._closing = False
        self._last_recommendation_id: int | None = None
        self._notification_window: tk.Toplevel | None = None
        self._notification_recommendation_id: int | None = None
        self._details_visible = False

        self._camera_window: tk.Toplevel | None = None
        self._camera_canvas: tk.Canvas | None = None
        self._camera_photo: tk.PhotoImage | None = None
        self._last_preview_captured_at: float | None = None

        self._create_style()
        self._create_widgets()
        self._start_monitor()
        self.root.after(self.ASSESSMENT_POLL_MS, self._poll)
        self.root.after(self.PREVIEW_POLL_MS, self._poll_camera_preview)

    def _create_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("AppTitle.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("SectionTag.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("Hero.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("State.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Body.TLabel", font=("Segoe UI", 10))
        style.configure("Small.TLabel", font=("Segoe UI", 9))
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.configure("Secondary.TButton", padding=(12, 8))

    def _create_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        self.outer = outer

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Ассистент восстановления", style="AppTitle.TLabel").pack(
            side="left"
        )
        self.monitor_badge_var = tk.StringVar(value="● Запуск")
        ttk.Label(header, textvariable=self.monitor_badge_var, style="State.TLabel").pack(
            side="right"
        )

        if self.demo_mode:
            demo_banner = tk.Frame(outer, background="#fff3cd", padx=12, pady=8)
            demo_banner.pack(fill="x", pady=(0, 10))
            tk.Label(
                demo_banner,
                text=(
                    "ДЕМО-РЕЖИМ: временные пороги ускорены. Уведомления в этом режиме "
                    "проверяют интерфейс и не являются выводом об усталости."
                ),
                background="#fff3cd",
                foreground="#5f4700",
                font=("Segoe UI", 9, "bold"),
                anchor="w",
            ).pack(fill="x")

        # На основном экране только три смысловых блока.
        self._build_status_card(outer)
        self._build_reason_card(outer)
        self._build_action_card(outer)
        self._build_footer(outer)
        self._build_details_panel(outer)

    def _build_status_card(self, parent: ttk.Frame) -> None:
        card = ttk.LabelFrame(parent, text="1. СОСТОЯНИЕ", padding=14)
        card.pack(fill="x", pady=(0, 10))

        self.context_var = tk.StringVar(value="Контекст уточняется")
        ttk.Label(card, textvariable=self.context_var, style="Small.TLabel").pack(anchor="w")

        self.hero_title_var = tk.StringVar(value="Оценка запускается")
        ttk.Label(
            card,
            textvariable=self.hero_title_var,
            style="Hero.TLabel",
            wraplength=840,
        ).pack(anchor="w", pady=(4, 3))

        self.hero_text_var = tk.StringVar(value="Подготавливаю датчики и личные настройки.")
        ttk.Label(
            card,
            textvariable=self.hero_text_var,
            style="Body.TLabel",
            wraplength=840,
        ).pack(anchor="w")

        self.session_context_var = tk.StringVar(
            value="Непрерывная работа: 00:00   ·   Оценка обновлена: сейчас"
        )
        ttk.Label(
            card,
            textvariable=self.session_context_var,
            style="Small.TLabel",
            foreground="#333333",
        ).pack(anchor="w", pady=(7, 0))

        self.reliability_var = tk.StringVar(value="Оценка ограничена")
        ttk.Label(
            card,
            textvariable=self.reliability_var,
            style="Small.TLabel",
            foreground="#555555",
        ).pack(anchor="w", pady=(2, 0))

    def _build_reason_card(self, parent: ttk.Frame) -> None:
        card = ttk.LabelFrame(parent, text="2. ОСНОВАНИЕ", padding=14)
        card.pack(fill="x", pady=(0, 10))
        self.reason1_var = tk.StringVar(value="Собираются первые измерения")
        self.reason2_var = tk.StringVar(value="Личная норма ритма ещё формируется")
        ttk.Label(card, textvariable=self.reason1_var, style="State.TLabel", wraplength=840).pack(
            anchor="w"
        )
        ttk.Label(card, textvariable=self.reason2_var, style="Body.TLabel", wraplength=840).pack(
            anchor="w", pady=(5, 0)
        )

    def _build_action_card(self, parent: ttk.Frame) -> None:
        card = ttk.LabelFrame(parent, text="3. РЕКОМЕНДАЦИЯ", padding=14)
        card.pack(fill="x", pady=(0, 10))

        self.action_title_var = tk.StringVar(value="Продолжайте работу в обычном режиме")
        ttk.Label(
            card,
            textvariable=self.action_title_var,
            style="State.TLabel",
            wraplength=840,
        ).pack(anchor="w")
        self.action_text_var = tk.StringVar(
            value="Система продолжает наблюдение и уточняет личный рабочий диапазон."
        )
        ttk.Label(
            card,
            textvariable=self.action_text_var,
            style="Body.TLabel",
            wraplength=840,
        ).pack(anchor="w", pady=(4, 10))

        actions = ttk.Frame(card)
        actions.pack(fill="x")
        self.break_button = ttk.Button(
            actions,
            text="Начать перерыв",
            command=self._toggle_break,
            style="Primary.TButton",
        )
        self.break_button.pack(side="left")
        self.snooze_button = ttk.Button(
            actions,
            text="Отложить 10 минут",
            command=self._snooze,
            style="Secondary.TButton",
            state="disabled",
        )
        self.snooze_button.pack(side="left", padx=(8, 0))
        self.irrelevant_button = ttk.Button(
            actions,
            text="Неуместно",
            command=self._mark_irrelevant,
            style="Secondary.TButton",
            state="disabled",
        )
        self.irrelevant_button.pack(side="left", padx=(8, 0))

    def _build_footer(self, parent: ttk.Frame) -> None:
        footer = ttk.Frame(parent)
        footer.pack(fill="x")

        ttk.Button(
            footer,
            text="Как я себя чувствую",
            command=self._self_report,
            style="Secondary.TButton",
        ).pack(side="left")

        self.camera_status_var = tk.StringVar(value="● Камера: проверяю сигнал")
        ttk.Button(
            footer,
            textvariable=self.camera_status_var,
            command=self._toggle_camera_preview,
            style="Secondary.TButton",
        ).pack(side="left", padx=(8, 0))

        self.pause_button = ttk.Button(
            footer,
            text="Приостановить",
            command=self._toggle_monitor,
            style="Secondary.TButton",
        )
        self.pause_button.pack(side="left", padx=(8, 0))

        self.details_button = ttk.Button(
            footer,
            text="Подробнее ▾",
            command=self._toggle_details,
            style="Secondary.TButton",
        )
        self.details_button.pack(side="right")

    def _build_details_panel(self, parent: ttk.Frame) -> None:
        self.details_frame = ttk.LabelFrame(
            parent,
            text="Технические подробности",
            padding=8,
        )
        self.details_notebook = ttk.Notebook(self.details_frame)
        self.details_notebook.pack(fill="both", expand=True)

        self.detail_texts: dict[str, scrolledtext.ScrolledText] = {}
        for title in ("Сессия", "Печать", "Глаза", "Решение"):
            frame = ttk.Frame(self.details_notebook, padding=6)
            text = scrolledtext.ScrolledText(
                frame,
                height=15,
                wrap="word",
                state="disabled",
                font=("Consolas", 9),
            )
            text.pack(fill="both", expand=True)
            self.details_notebook.add(frame, text=title)
            self.detail_texts[title] = text

    def _start_monitor(self) -> None:
        try:
            self.monitor.start()
        except Exception as error:
            messagebox.showerror(
                "Не удалось запустить монитор",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            self.monitor_badge_var.set("● Ошибка")
            return
        self._monitor_running = True
        self.pause_button.configure(text="Приостановить")
        if self.monitor.vision_available:
            self.monitor_badge_var.set("● Мониторинг включён")
        else:
            self.monitor_badge_var.set("● Мониторинг включён (без камеры)")
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

    def _poll(self) -> None:
        if self._closing:
            return
        if self._monitor_running:
            try:
                snapshot = self.monitor.snapshot()
                signals = build_recovery_signals(snapshot, self.typing_baseline)
                assessment = self.engine.update(signals)
                self._last_assessment = assessment
                self._last_signals = signals
                self._render(snapshot, signals, assessment)
                if assessment.recommendation_is_new and assessment.recommendation is not None:
                    self._log_recommendation(assessment.recommendation)
                    self._show_recommendation_notification(assessment.recommendation)
            except Exception as error:
                self.monitor_badge_var.set("● Ошибка датчика")
                self.hero_title_var.set("Оценка временно недоступна")
                self.hero_text_var.set(f"{type(error).__name__}: {error}")
        self.root.after(self.ASSESSMENT_POLL_MS, self._poll)

    def _poll_camera_preview(self) -> None:
        if self._closing:
            return
        window = self._camera_window
        canvas = self._camera_canvas
        if (
            self._monitor_running
            and window is not None
            and canvas is not None
        ):
            try:
                if window.winfo_exists():
                    captured_at, frame = self.monitor.latest_visual_frame()
                    if (
                        frame is not None
                        and captured_at is not None
                        and captured_at != self._last_preview_captured_at
                    ):
                        self._camera_photo = self._frame_to_photo(frame)
                        canvas.delete("all")
                        canvas.create_image(
                            self.PREVIEW_WIDTH // 2,
                            self.PREVIEW_HEIGHT // 2,
                            image=self._camera_photo,
                            anchor="center",
                        )
                        self._last_preview_captured_at = captured_at
            except tk.TclError:
                self._close_camera_preview()
            except Exception:
                pass
        self.root.after(self.PREVIEW_POLL_MS, self._poll_camera_preview)

    def _render(
        self,
        snapshot,
        signals: RecoverySignals,
        assessment: RecoveryAssessment,
    ) -> None:
        self.context_var.set(STATE_USER_LABELS.get(snapshot.state, snapshot.state_label))
        title, detail = user_facing_summary(assessment, demo_mode=self.demo_mode)
        self.hero_title_var.set(title)
        self.hero_text_var.set(detail)
        self.session_context_var.set(
            "Непрерывная работа: "
            + format_duration(assessment.continuous_work_sec)
            + "   ·   Оценка обновлена: "
            + format_assessment_age(time.monotonic() - assessment.captured_at)
        )
        self.reliability_var.set(assessment.reliability_label)

        reasons = list(assessment.primary_reasons)
        self.reason1_var.set(reasons[0] if reasons else "Достаточных оснований для изменения уровня пока нет")
        self.reason2_var.set(reasons[1] if len(reasons) > 1 else assessment.decision_basis)

        self._render_camera_status(snapshot)
        self._render_action(assessment)
        if self._details_visible:
            self._render_details(snapshot, signals, assessment)

    def _render_camera_status(self, snapshot) -> None:
        ocular = snapshot.ocular_metrics
        if snapshot.face_detected is True:
            if ocular.signal_coverage >= 0.85:
                self.camera_status_var.set("● Камера: сигнал устойчив")
            else:
                self.camera_status_var.set("● Камера: сигнал ограничен")
        elif snapshot.face_detected is False:
            self.camera_status_var.set("● Камера: лицо не в кадре")
        else:
            self.camera_status_var.set("● Камера: качество ограничено")

    def _render_action(self, assessment: RecoveryAssessment) -> None:
        recommendation = assessment.recommendation
        if recommendation is None:
            if self._notification_recommendation_id is not None:
                self._close_recommendation_notification()
            self.snooze_button.configure(state="disabled")
            self.irrelevant_button.configure(state="disabled")

            title, text = recommended_action(assessment)
            self.action_title_var.set(title)
            self.action_text_var.set(text)
            return

        self.snooze_button.configure(state="normal")
        self.irrelevant_button.configure(state="normal")
        if self.demo_mode:
            self.action_title_var.set("ДЕМО: рекомендация сформирована")
            self.action_text_var.set(
                "Сработал ускоренный временной порог. Это проверка механизма уведомлений, а не вывод об усталости."
            )
        else:
            self.action_title_var.set(recommendation.title)
            self.action_text_var.set(recommendation.message)

    def _show_recommendation_notification(
        self,
        recommendation: RecoveryRecommendation,
    ) -> None:
        if self._notification_recommendation_id == recommendation.recommendation_id:
            window = self._notification_window
            if window is not None and window.winfo_exists():
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

        toast = RecommendationToast(
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
        self._notification_window = toast.window
        self._notification_recommendation_id = recommendation.recommendation_id
        self._play_notification_sound()

    def _play_notification_sound(self) -> None:
        if winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
                return
            except RuntimeError:
                pass
        try:
            self.root.bell()
        except tk.TclError:
            pass

    def _notification_closed(self) -> None:
        self._notification_window = None

    def _close_recommendation_notification(self) -> None:
        window = self._notification_window
        self._notification_window = None
        self._notification_recommendation_id = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass

    def _toggle_camera_preview(self) -> None:
        if self._camera_window is not None:
            try:
                if self._camera_window.winfo_exists():
                    self._close_camera_preview()
                    return
            except tk.TclError:
                pass

        window = tk.Toplevel(self.root)
        window.title("Камера — диагностическое превью")
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", self._close_camera_preview)
        container = ttk.Frame(window, padding=10)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(
            container,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            background="#202124",
            highlightthickness=0,
        )
        canvas.pack()
        canvas.create_text(
            self.PREVIEW_WIDTH // 2,
            self.PREVIEW_HEIGHT // 2,
            text="Получаю изображение…",
            fill="white",
            font=("Segoe UI", 11),
        )
        ttk.Label(
            container,
            text="Превью не сохраняется. Закрытие окна прекращает только его отрисовку.",
            wraplength=self.PREVIEW_WIDTH,
        ).pack(anchor="w", pady=(8, 0))
        self._camera_window = window
        self._camera_canvas = canvas
        self._last_preview_captured_at = None

    def _close_camera_preview(self) -> None:
        window = self._camera_window
        self._camera_window = None
        self._camera_canvas = None
        self._camera_photo = None
        self._last_preview_captured_at = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass

    def _frame_to_photo(self, frame) -> tk.PhotoImage:
        height, width = frame.shape[:2]
        scale = min(self.PREVIEW_WIDTH / width, self.PREVIEW_HEIGHT / height)
        target_w = max(1, int(width * scale))
        target_h = max(1, int(height * scale))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((self.PREVIEW_HEIGHT, self.PREVIEW_WIDTH, 3), 32, dtype=np.uint8)
        x0 = (self.PREVIEW_WIDTH - target_w) // 2
        y0 = (self.PREVIEW_HEIGHT - target_h) // 2
        canvas[y0 : y0 + target_h, x0 : x0 + target_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        header = f"P6\n{self.PREVIEW_WIDTH} {self.PREVIEW_HEIGHT}\n255\n".encode("ascii")
        ppm = header + rgb.tobytes()
        return tk.PhotoImage(data=ppm, format="PPM")

    def _toggle_details(self) -> None:
        self._details_visible = not self._details_visible
        if self._details_visible:
            self.details_frame.pack(fill="both", expand=True, pady=(10, 0))
            self.details_button.configure(text="Подробнее ▴")
            self.root.geometry("1000x900")
            if self._last_assessment is not None and self._last_signals is not None:
                try:
                    snapshot = self.monitor.snapshot()
                    self._render_details(snapshot, self._last_signals, self._last_assessment)
                except Exception:
                    pass
        else:
            self.details_frame.pack_forget()
            self.details_button.configure(text="Подробнее ▾")
            self.root.geometry("960x690")

    def _render_details(
        self,
        snapshot,
        signals: RecoverySignals,
        assessment: RecoveryAssessment,
    ) -> None:
        session_lines = [
            "РАБОЧИЙ КОНТЕКСТ",
            f"state = {snapshot.state.value}",
            f"state_label = {snapshot.state_label}",
            f"state_reason = {snapshot.state_reason}",
            f"seconds_in_state = {snapshot.seconds_in_state:.2f}",
            f"input_idle_sec = {format_optional(snapshot.input_idle_sec)}",
            f"face_detected = {format_optional(snapshot.face_detected)}",
            f"gaze_on_screen = {format_optional(snapshot.gaze_on_screen)}",
            f"fatigue_analysis_allowed = {snapshot.fatigue_analysis_allowed}",
            f"baseline_update_allowed = {snapshot.baseline_update_allowed}",
            f"camera_status = {snapshot.camera_status}",
            f"attention_status = {snapshot.attention_status}",
            f"ocular_status = {snapshot.ocular_status}",
            f"visual_age_sec = {snapshot.visual_age_sec:.3f}",
            "",
            "НАКОПЛЕНИЕ",
            f"continuous_work_sec = {assessment.continuous_work_sec:.2f}",
            f"current_break_sec = {assessment.current_break_sec:.2f}",
            f"typing_calibration_progress = {assessment.typing_calibration_progress:.3f}",
            f"self_report_fatigue_sp = {format_optional(assessment.self_report_fatigue_sp)}",
            f"self_report_sleepiness_kss = {format_optional(assessment.self_report_sleepiness_kss)}",
        ]

        typing_lines = ["ТЕКУЩЕЕ ОКНО ПЕЧАТИ"]
        typing_lines.extend(self._dataclass_lines(snapshot.typing_metrics))
        typing_lines.extend(["", "СРАВНЕНИЕ С ЛИЧНОЙ НОРМОЙ"])
        for feature_name, deviation in signals.typing_deviations.items():
            typing_lines.extend(
                [
                    f"[{feature_name}]",
                    f"  actual = {format_optional(deviation.actual_value)}",
                    f"  baseline_median = {format_optional(deviation.baseline_median)}",
                    f"  robust_z = {format_optional(deviation.robust_z)}",
                    f"  ready = {deviation.ready}",
                    f"  confidence = {deviation.confidence:.3f}",
                ]
            )

        ocular_lines = ["ГЛАЗНОЙ КАНАЛ — ДИАГНОСТИКА"]
        ocular_lines.append(
            "До завершения размеченной валидации эти показатели не меняют итоговый уровень."
        )
        ocular_lines.append("")
        ocular_lines.extend(self._dataclass_lines(snapshot.ocular_metrics))

        diagnostics = self.engine.diagnostics(captured_at=assessment.captured_at)
        decision_lines = [
            "ИТОГОВАЯ ЛОГИКА",
            f"workload_level = {int(assessment.workload_level)}",
            f"workload_title = {assessment.workload_title}",
            f"reliability = {assessment.reliability.value}",
            f"reliability_detail = {assessment.reliability_detail}",
            f"decision_basis = {assessment.decision_basis}",
            f"ocular_used_for_decision = {assessment.ocular_used_for_decision}",
            "contributing_sources = "
            + ", ".join(EVIDENCE_SOURCE_LABELS[source] for source in assessment.contributing_sources),
            "",
            "ГИСТЕРЕЗИС И ОГРАНИЧЕНИЯ",
        ]
        decision_lines.extend(f"{key} = {format_optional(value)}" for key, value in diagnostics.items())
        decision_lines.extend(["", "ДОКАЗАТЕЛЬСТВА"])
        if assessment.evidence:
            for item in assessment.evidence:
                decision_lines.extend(
                    [
                        f"[{item.code}] source={item.source.value}; severity={item.severity}; decision_ready={item.decision_ready}",
                        f"  {item.title}",
                        f"  {item.detail}",
                    ]
                )
        else:
            decision_lines.append("нет активных положительных сигналов")

        if assessment.recommendation is not None:
            recommendation = assessment.recommendation
            decision_lines.extend(
                [
                    "",
                    "РЕКОМЕНДАЦИЯ",
                    f"id = {recommendation.recommendation_id}",
                    f"kind = {recommendation.kind.value}",
                    f"title = {recommendation.title}",
                    f"message = {recommendation.message}",
                    f"suggested_break_sec = {recommendation.suggested_break_sec:.1f}",
                    f"reason_codes = {', '.join(recommendation.reason_codes)}",
                ]
            )

        self._set_detail_text("Сессия", "\n".join(session_lines))
        self._set_detail_text("Печать", "\n".join(typing_lines))
        self._set_detail_text("Глаза", "\n".join(ocular_lines))
        self._set_detail_text("Решение", "\n".join(decision_lines))

    @staticmethod
    def _dataclass_lines(value) -> list[str]:
        if not is_dataclass(value):
            return [str(value)]
        result: list[str] = []
        for field in fields(value):
            result.append(f"{field.name} = {format_optional(getattr(value, field.name))}")
        return result

    def _set_detail_text(self, title: str, content: str) -> None:
        widget = self.detail_texts[title]
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.configure(state="disabled")

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
        self.break_button.configure(text="Закончить перерыв")
        resolved = self.engine.resolve_recommendation(captured_at=time.monotonic())
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
        self.break_button.configure(text="Начать перерыв")
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
        initial_fatigue = assessment.self_report_fatigue_sp if assessment else None
        initial_sleepiness = assessment.self_report_sleepiness_kss if assessment else None
        SelfReportDialog(
            self.root,
            initial_fatigue=initial_fatigue,
            initial_sleepiness=initial_sleepiness,
            on_save=self._save_self_report,
        )

    def _save_self_report(self, fatigue_sp: int, sleepiness_kss: int) -> None:
        now = time.monotonic()
        self.engine.record_self_report(
            fatigue_sp_1_7=float(fatigue_sp),
            sleepiness_kss_1_9=float(sleepiness_kss),
            captured_at=now,
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

    def _toggle_monitor(self) -> None:
        if self._monitor_running:
            try:
                self.monitor.stop()
            finally:
                self._monitor_running = False
                self._close_recommendation_notification()
                self._close_camera_preview()
                self.pause_button.configure(text="Возобновить")
                self.monitor_badge_var.set("● Мониторинг остановлен")
                self.store.log_event("monitor_paused")
        else:
            self._start_monitor()

    def _log_recommendation(self, recommendation: RecoveryRecommendation) -> None:
        self._last_recommendation_id = recommendation.recommendation_id
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

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._close_recommendation_notification()
        self._close_camera_preview()
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
    parser = argparse.ArgumentParser(description="Пользовательский MVP ассистента восстановления.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Показать тестовую рекомендацию по ускоренным временным порогам. "
            "В этом режиме уведомление не является выводом об усталости."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    RecoveryAssistantApp(root, demo_mode=args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()
