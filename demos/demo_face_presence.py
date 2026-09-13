import cv2

from app_core.face_presence import FacePresenceSensor


def main() -> None:
    sensor = FacePresenceSensor(
        camera_index=0,
        min_detection_confidence=0.55,
    )

    print("Датчик присутствия лица запущен.")
    print("Видео и фотографии не сохраняются.")
    print("Для выхода нажми Q или Escape.")

    try:
        sensor.start()

        while True:
            snapshot = sensor.snapshot()

            if snapshot.frame is None:
                print(
                    "Не удалось получить кадр с камеры."
                )
                continue

            frame = snapshot.frame.copy()

            if snapshot.face_detected is True:
                status = "FACE DETECTED"
                status_color = (0, 220, 0)

            elif snapshot.face_detected is False:
                status = "NO FACE"
                status_color = (0, 0, 255)

            else:
                status = "LOW IMAGE QUALITY"
                status_color = (0, 180, 255)

            if snapshot.face_box is not None:
                x, y, width, height = snapshot.face_box

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    status_color,
                    2,
                )

            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Quality: "
                    f"{snapshot.data_quality:.2f}"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Confidence: "
                    f"{snapshot.detection_confidence:.2f}"
                ),
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Brightness: "
                    f"{snapshot.brightness:.0f}"
                ),
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Face presence test",
                frame,
            )

            pressed_key = (
                cv2.waitKey(1) & 0xFF
            )

            if pressed_key in {
                ord("q"),
                27,
            }:
                break

    finally:
        sensor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()