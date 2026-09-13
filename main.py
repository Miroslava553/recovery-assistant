from __future__ import annotations

import time
from pathlib import Path

from app_core.session_monitor import SessionMonitor
from app_core.storage import AppDatabase
from app_core.typing_baseline import TypingBaselineService


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "fatigue_assistant.sqlite3"


def main() -> None:
    database = AppDatabase(DATABASE_PATH)
    profile_id = database.get_or_create_profile("Основной пользователь")

    typing_baseline = TypingBaselineService(
        DATABASE_PATH,
        profile_id=profile_id,
        sample_interval_sec=60.0,
        min_observation_sec=50.0,
        target_relevant_keys=25,
    )

    monitor = SessionMonitor(
        typing_baseline_service=typing_baseline,
        include_diagnostic_frame=False,
    )

    print("Фоновый монитор рабочей сессии запущен.")
    print("Окно камеры не показывается; кадры не сохраняются.")
    print("Текст и названия клавиш не записываются.")
    print("Для временной остановки нажми Ctrl+C.")
    print()

    previous_state = None

    try:
        monitor.start()

        while True:
            snapshot = monitor.snapshot()

            if snapshot.state is not previous_state:
                print(f"Состояние: {snapshot.state_label}")
                print(f"Причина: {snapshot.state_reason}")
                print()
                previous_state = snapshot.state

            baseline_result = snapshot.typing_baseline
            if baseline_result is not None and baseline_result.accepted:
                print(
                    "Личная норма печати обновлена: "
                    f"{snapshot.typing_metrics.keys_per_minute:.1f} "
                    "условных нажатий/мин."
                )
                print(
                    "Калибровка: "
                    f"{baseline_result.calibration_day_count}/5 дней; "
                    f"прогресс {baseline_result.calibration_progress:.0%}."
                )
                print()

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\nФоновый монитор остановлен.")

    finally:
        monitor.close()


if __name__ == "__main__":
    main()