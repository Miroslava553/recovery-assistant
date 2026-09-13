from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from app_core.baseline_engine import PersonalBaselineEngine, WorkContext
from app_core.state_machine import UserState
from app_core.storage import AppDatabase
from app_core.typing_activity import GlobalKeyboardTimingSensor
from app_core.typing_baseline import TypingBaselineService


def main() -> None:
    database = AppDatabase(
        Path("../data") / "fatigue_assistant.sqlite3"
    )
    profile_id = database.get_or_create_profile(
        "Основной пользователь"
    )

    baseline = PersonalBaselineEngine()
    service = TypingBaselineService(
        database.db_path,
        profile_id=profile_id,
        baseline=baseline,
    )

    sensor = GlobalKeyboardTimingSensor()
    started_at = time.monotonic()
    workday = datetime.now().date()

    print("Персональная норма печати запущена.")
    print("Можно переключиться в любое другое окно и печатать там.")
    print("Текст и названия клавиш не сохраняются.")
    print("Первое полноценное окно формируется примерно 60 секунд.")
    print("Ctrl+C — завершить проверку.\n")

    sensor.start()

    try:
        while True:
            time.sleep(5.0)
            metrics = sensor.snapshot()
            minutes_since_start = (
                time.monotonic() - started_at
            ) / 60.0

            result = service.observe(
                metrics,
                state=UserState.ACTIVE_WORK,
                workday=workday,
                minutes_since_workday_start=minutes_since_start,
                captured_at=datetime.now().astimezone(),
            )

            print(
                f"Клавиш/мин: {metrics.keys_per_minute:6.1f} | "
                f"событий: {metrics.relevant_key_count:3d} | "
                f"готово: {'да' if metrics.data_ready else 'нет'}"
            )

            if result.accepted:
                stats = baseline.stats(
                    WorkContext.TYPING,
                    "typing_keys_per_minute",
                )
                median_text = (
                    f"{stats.median:.1f}"
                    if stats is not None
                    else "формируется"
                )
                print(
                    "  ✓ Окно принято. "
                    f"Дней калибровки: {result.calibration_day_count}/5; "
                    f"медиана: {median_text}."
                )
            elif result.attempted:
                print(f"  — Окно проверено: {result.reason}")

    except KeyboardInterrupt:
        print("\nПроверка завершена.")

    finally:
        sensor.stop()


if __name__ == "__main__":
    main()
