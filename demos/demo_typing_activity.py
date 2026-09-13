import time

from app_core.typing_activity import (
    GlobalKeyboardTimingSensor,
    TypingActivityCollector,
)


def format_number(value: float | None) -> str:
    if value is None:
        return "нет данных"

    return f"{value:.2f}"


def main() -> None:
    collector = TypingActivityCollector(
        window_sec=60.0,
        long_pause_sec=2.0,
        burst_gap_sec=1.5,
        min_observation_sec=15.0,
        min_relevant_keys=20,
    )

    sensor = GlobalKeyboardTimingSensor(collector)

    print("Обезличенный датчик печати запущен.")
    print("Текст и конкретные клавиши не сохраняются.")
    print("Открой Блокнот и начни печатать.")
    print("Для остановки нажми Ctrl+C.")
    print()

    sensor.start()

    try:
        while True:
            metrics = sensor.snapshot()

            print("\033[2J\033[H", end="")

            print("АНАЛИЗ ТЕМПА ПЕЧАТИ")
            print("====================")
            print(
                f"Время наблюдения: "
                f"{metrics.observation_sec:.1f} с"
            )
            print(
                f"Данных достаточно: "
                f"{'да' if metrics.data_ready else 'нет'}"
            )
            print()

            print(
                f"Обычных нажатий: "
                f"{metrics.text_key_count}"
            )
            print(
                f"Исправлений: "
                f"{metrics.correction_key_count}"
            )
            print(
                f"Всего учтено: "
                f"{metrics.relevant_key_count}"
            )
            print(
                f"Темп: "
                f"{metrics.keys_per_minute:.1f} "
                f"нажатий/мин"
            )
            print()

            print(
                "Средний интервал: "
                f"{format_number(metrics.mean_interval_sec)} с"
            )
            print(
                "Медианный интервал: "
                f"{format_number(metrics.median_interval_sec)} с"
            )
            print(
                "Неровность ритма: "
                f"{format_number(metrics.rhythm_cv)}"
            )
            print(
                "Самая длинная пауза: "
                f"{format_number(metrics.longest_pause_sec)} с"
            )
            print(
                f"Длинных пауз: "
                f"{metrics.long_pause_count}"
            )
            print(
                f"Серий печати: "
                f"{metrics.burst_count}"
            )
            print(
                f"Доля исправлений: "
                f"{metrics.correction_ratio:.1%}"
            )
            print(
                "Без учитываемого нажатия: "
                f"{format_number(metrics.current_idle_sec)} с"
            )

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nДатчик остановлен.")

    finally:
        sensor.stop()


if __name__ == "__main__":
    main()