from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

try:
    import mediapipe as mp
except ModuleNotFoundError:  # Unit-тесты могут запускаться без камеры.
    mp = None


@dataclass(slots=True)
class FacePresenceSnapshot:
    """
    Результат анализа одного кадра.

    Кадр существует только в оперативной памяти
    и не сохраняется на диск.
    """

    face_detected: bool | None
    data_quality: float
    detection_confidence: float
    brightness: float
    sharpness: float
    reason: str
    frame: Any
    face_box: tuple[int, int, int, int] | None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def calculate_frame_quality(
    brightness: float,
    sharpness: float,
) -> float:
    """
    Приблизительно оценивает пригодность изображения.

    0.0 — кадр практически непригоден.
    1.0 — хорошее освещение и достаточная резкость.
    """

    dark_score = clamp(
        (brightness - 15.0) / 55.0,
        0.0,
        1.0,
    )

    bright_score = clamp(
        (245.0 - brightness) / 45.0,
        0.0,
        1.0,
    )

    brightness_score = min(
        dark_score,
        bright_score,
    )

    sharpness_score = clamp(
        (sharpness - 10.0) / 120.0,
        0.0,
        1.0,
    )

    return (
        brightness_score * 0.65
        + sharpness_score * 0.35
    )


class FacePresenceSensor:
    """
    Определяет присутствие лица с помощью камеры.

    Видео и изображения не записываются.
    Обрабатывается только текущий кадр.
    """

    def __init__(
        self,
        *,
        camera_index: int = 0,
        min_detection_confidence: float = 0.55,
        min_usable_quality: float = 0.18,
        frame_width: int = 640,
        frame_height: int = 480,
        run_face_detection: bool = True,
    ) -> None:
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError(
                "min_detection_confidence должен быть от 0 до 1."
            )

        if not 0.0 <= min_usable_quality <= 1.0:
            raise ValueError(
                "min_usable_quality должен быть от 0 до 1."
            )

        self.camera_index = camera_index
        self.min_usable_quality = min_usable_quality
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.run_face_detection = bool(run_face_detection)

        self._capture: cv2.VideoCapture | None = None

        self._face_detection = None
        if self.run_face_detection:
            if mp is None:
                raise RuntimeError(
                    "Для анализа лица не установлен mediapipe. "
                    "Установи зависимости из requirements.txt."
                )
            self._face_detection = (
                mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_detection_confidence,
                )
            )

    @property
    def running(self) -> bool:
        return (
            self._capture is not None
            and self._capture.isOpened()
        )

    def start(self) -> None:
        if self.running:
            return

        # Сначала пробуем DirectShow — обычно он быстрее
        # запускает встроенную камеру в Windows.
        capture = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW,
        )

        if not capture.isOpened():
            capture.release()

            # Запасной обычный способ открытия камеры.
            capture = cv2.VideoCapture(
                self.camera_index
            )

        if not capture.isOpened():
            capture.release()

            raise RuntimeError(
                "Не удалось открыть камеру. "
                "Возможно, она занята другой программой "
                "или доступ к ней запрещён Windows."
            )

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.frame_width,
        )
        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.frame_height,
        )
        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        self._capture = capture

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()

        self._capture = None

    def snapshot(self) -> FacePresenceSnapshot:
        if not self.running:
            raise RuntimeError(
                "Сначала нужно запустить датчик методом start()."
            )

        assert self._capture is not None

        success, frame = self._capture.read()

        if not success or frame is None:
            return FacePresenceSnapshot(
                face_detected=None,
                data_quality=0.0,
                detection_confidence=0.0,
                brightness=0.0,
                sharpness=0.0,
                reason="camera_frame_unavailable",
                frame=None,
                face_box=None,
            )

        # Зеркальное отображение, как в обычной веб-камере.
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(gray.mean())

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        data_quality = calculate_frame_quality(
            brightness=brightness,
            sharpness=sharpness,
        )

        confidence = 0.0
        face_box = None
        detections = []

        if self._face_detection is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._face_detection.process(rgb_frame)
            detections = result.detections if result.detections else []

            if detections:
                best_detection = max(
                    detections,
                    key=lambda detection: detection.score[0],
                )
                confidence = float(best_detection.score[0])
                relative_box = (
                    best_detection.location_data.relative_bounding_box
                )
                frame_height, frame_width = frame.shape[:2]
                x = max(0, int(relative_box.xmin * frame_width))
                y = max(0, int(relative_box.ymin * frame_height))
                width = min(
                    int(relative_box.width * frame_width),
                    frame_width - x,
                )
                height = min(
                    int(relative_box.height * frame_height),
                    frame_height - y,
                )
                face_box = (x, y, width, height)

        if data_quality < self.min_usable_quality:
            face_detected = None
            if brightness < 35:
                reason = "too_dark"
            elif brightness > 225:
                reason = "too_bright"
            else:
                reason = "image_quality_too_low"
        elif self._face_detection is None:
            # Присутствие лица определит тот же Face Mesh, который нужен
            # для глаз. Так мы не запускаем вторую модель на каждом кадре.
            face_detected = None
            reason = "frame_ready"
        elif detections:
            face_detected = True
            reason = "face_detected"
        else:
            face_detected = False
            reason = "face_not_detected"

        return FacePresenceSnapshot(
            face_detected=face_detected,
            data_quality=data_quality,
            detection_confidence=confidence,
            brightness=brightness,
            sharpness=sharpness,
            reason=reason,
            frame=frame,
            face_box=face_box,
        )

    def close(self) -> None:
        self.stop()
        if self._face_detection is not None:
            self._face_detection.close()

    def __enter__(self) -> FacePresenceSensor:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.close()