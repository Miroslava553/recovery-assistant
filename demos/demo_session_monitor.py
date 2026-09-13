import cv2
import numpy as np

from app_core.session_monitor import SessionMonitor
from app_core.state_machine import UserState


STATE_TEXT: dict[UserState, str] = {
    UserState.UNKNOWN: "DETERMINING STATE...",
    UserState.ACTIVE_WORK: "ACTIVE WORK",
    UserState.PASSIVE_WORK: "PASSIVE WORK",
    UserState.AWAY: "AWAY",
    UserState.BREAK: "BREAK",
}

STATE_COLORS: dict[UserState, tuple[int, int, int]] = {
    UserState.UNKNOWN: (0, 165, 255),
    UserState.ACTIVE_WORK: (0, 210, 0),
    UserState.PASSIVE_WORK: (0, 210, 210),
    UserState.AWAY: (150, 150, 150),
    UserState.BREAK: (255, 160, 0),
}


def optional_number(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def main() -> None:
    monitor = SessionMonitor(
        include_diagnostic_frame=True,
        target_visual_fps=30.0,
        background_visual_processing=True,
    )

    print("Диагностический монитор запущен.")
    print("Q или Escape — завершить.")
    print("Кадры показываются только здесь и не сохраняются.")
    print("Категории full/incomplete пока диагностические, а не доказанные.")
    print("Для измерения точности используй ocular_validation.py.")
    print("Winks — односторонние закрытия, они не входят в моргания.")

    previous_state = None

    try:
        monitor.start()
        while True:
            snapshot = monitor.snapshot()
            if snapshot.state is not previous_state:
                print()
                print(f"Состояние: {snapshot.state_label}")
                print(f"Причина: {snapshot.state_reason}")
                previous_state = snapshot.state

            if snapshot.frame is None:
                continue

            frame = snapshot.frame.copy()
            state_color = STATE_COLORS[snapshot.state]
            ocular = snapshot.ocular_metrics

            if snapshot.face_box is not None:
                x, y, width, height = snapshot.face_box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    state_color,
                    2,
                )

            # Диагностическая панель располагается рядом с изображением,
            # а не поверх лица. Так она не закрывает глаза и не мешает
            # оценивать работу детектора.
            panel_width = 360
            panel = np.full(
                (frame.shape[0], panel_width, 3),
                (25, 25, 25),
                dtype=np.uint8,
            )

            lines = [
                (STATE_TEXT[snapshot.state], 0.78, state_color),
                (f"Eye status: {ocular.reason}", 0.44, (235, 235, 235)),
                (
                    "Opening L/R: "
                    f"{optional_number(ocular.normalized_left_eye_opening)} / "
                    f"{optional_number(ocular.normalized_right_eye_opening)}",
                    0.44,
                    (235, 235, 235),
                ),
                (
                    "State L/R: "
                    f"{ocular.left_eye_state.value} / {ocular.right_eye_state.value}",
                    0.44,
                    (235, 235, 235),
                ),
                (
                    f"All bilateral events: {ocular.bilateral_blink_count}",
                    0.44,
                    (235, 235, 235),
                ),
                (
                    "Provisional full: "
                    f"{ocular.blink_count} | Shallow: "
                    f"{ocular.incomplete_blink_count}",
                    0.41,
                    (235, 235, 235),
                ),
                (
                    "Winks L/R: "
                    f"{ocular.left_wink_count} / {ocular.right_wink_count}",
                    0.44,
                    (235, 235, 235),
                ),
                (
                    f"Uncertain events: {ocular.uncertain_event_count}",
                    0.44,
                    (235, 235, 235),
                ),
                (
                    "PERCLOS: "
                    f"{optional_number(ocular.perclos, 3)} | "
                    f"Closure: {ocular.current_closure_duration_sec:.2f}s",
                    0.44,
                    (235, 235, 235),
                ),
                (
                    "FPS: "
                    f"{optional_number(ocular.processed_fps, 1)} | "
                    f"Valid: {ocular.signal_coverage * 100:.0f}%",
                    0.44,
                    (235, 235, 235),
                ),
                (
                    f"Ready: {'yes' if ocular.data_ready else 'no'}",
                    0.44,
                    (235, 235, 235),
                ),
            ]

            y_position = 34
            for text, scale, color in lines:
                cv2.putText(
                    panel,
                    text,
                    (14, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                y_position += 34

            display_frame = np.hstack((frame, panel))

            cv2.imshow("Blink detector diagnostics", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break

    finally:
        monitor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
