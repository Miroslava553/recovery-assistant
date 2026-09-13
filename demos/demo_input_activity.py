import time

from app_core.input_activity import (
    InputActivityError,
    WindowsInputActivitySensor,
)


def main() -> None:
    sensor = WindowsInputActivitySensor(
        recent_threshold_sec=8.0
    )

    print("Датчик активности Windows запущен.")
    print("Он не сохраняет клавиши, текст или координаты мыши.")
    print("Для остановки нажмите Ctrl+C.")
    print()

    try:
        while True:
            snapshot = sensor.snapshot()

            if snapshot.recent_input:
                activity_text = "недавний ввод есть"
            else:
                activity_text = "ввода давно не было"

            print(
                f"\rБез ввода: {snapshot.idle_seconds:6.1f} с | "
                f"{activity_text}",
                end="",
                flush=True,
            )

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\nДатчик остановлен.")

    except InputActivityError as error:
        print(f"\nОшибка датчика: {error}")


if __name__ == "__main__":
    main()