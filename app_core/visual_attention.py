from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

try:
    import mediapipe as mp
except ModuleNotFoundError:  # Геометрические тесты не требуют модели.
    mp = None


LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

# Три вертикальные пары устойчивее двух при небольшом шуме landmarks.
LEFT_EYE_VERTICAL_PAIRS = ((160, 144), (159, 145), (158, 153))
RIGHT_EYE_VERTICAL_PAIRS = ((387, 373), (386, 374), (385, 380))

LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)

LEFT_CHEEK = 234
RIGHT_CHEEK = 454
NOSE_TIP = 1
CHIN = 152


@dataclass(frozen=True, slots=True)
class VisualAttentionSnapshot:
    """Обезличенные признаки одного прохода Face Mesh."""

    gaze_on_screen: bool | None
    face_landmarks_detected: bool
    horizontal_gaze_ratio: float | None
    head_turn_ratio: float | None
    head_vertical_ratio: float | None
    reason: str

    left_eye_opening_ratio: float | None = None
    right_eye_opening_ratio: float | None = None
    left_eye_quality: float = 0.0
    right_eye_quality: float = 0.0
    head_pose_quality: float = 0.0
    face_box: tuple[int, int, int, int] | None = None

    @property
    def eye_opening_ratio(self) -> float | None:
        """Среднее оставлено только для диагностической совместимости."""
        values = [
            value
            for value in (
                self.left_eye_opening_ratio,
                self.right_eye_opening_ratio,
            )
            if value is not None
        ]
        return float(np.mean(values)) if values else None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


class VisualAttentionAnalyzer:
    """
    Из одного кадра получает положение головы, приблизительный взгляд и
    независимую геометрию каждого глаза. Камеру не открывает и кадры не хранит.
    """

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_window: int = 7,
    ) -> None:
        if smoothing_window < 1:
            raise ValueError("smoothing_window должен быть не меньше 1.")
        if mp is None:
            raise RuntimeError(
                "Для анализа глаз не установлен mediapipe. "
                "Установи зависимости из requirements.txt."
            )

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._attention_history: deque[bool] = deque(maxlen=smoothing_window)

    def analyze(
        self,
        frame: Any,
        *,
        data_quality: float,
    ) -> VisualAttentionSnapshot:
        if frame is None:
            self._attention_history.clear()
            return self._empty_snapshot("frame_unavailable")
        if data_quality < 0.18:
            self._attention_history.clear()
            return self._empty_snapshot("image_quality_too_low")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            self._attention_history.clear()
            return self._empty_snapshot("face_landmarks_not_detected")

        landmarks = result.multi_face_landmarks[0].landmark
        frame_height, frame_width = frame.shape[:2]

        def point(index: int) -> np.ndarray:
            item = landmarks[index]
            return np.array(
                [item.x * frame_width, item.y * frame_height],
                dtype=np.float64,
            )

        x_values = [item.x * frame_width for item in landmarks[:468]]
        y_values = [item.y * frame_height for item in landmarks[:468]]
        minimum_x = max(0, int(min(x_values)))
        minimum_y = max(0, int(min(y_values)))
        maximum_x = min(frame_width - 1, int(max(x_values)))
        maximum_y = min(frame_height - 1, int(max(y_values)))
        face_box = (
            minimum_x,
            minimum_y,
            max(0, maximum_x - minimum_x),
            max(0, maximum_y - minimum_y),
        )

        left_gaze = self._eye_horizontal_ratio(point, LEFT_EYE_CORNERS, LEFT_IRIS)
        right_gaze = self._eye_horizontal_ratio(point, RIGHT_EYE_CORNERS, RIGHT_IRIS)
        horizontal_gaze_ratio = float((left_gaze + right_gaze) / 2.0)

        left_eye_ratio, left_eye_width = self._eye_opening_measurement(
            point,
            LEFT_EYE_CORNERS,
            LEFT_EYE_VERTICAL_PAIRS,
        )
        right_eye_ratio, right_eye_width = self._eye_opening_measurement(
            point,
            RIGHT_EYE_CORNERS,
            RIGHT_EYE_VERTICAL_PAIRS,
        )

        left_cheek = point(LEFT_CHEEK)
        right_cheek = point(RIGHT_CHEEK)
        nose = point(NOSE_TIP)
        face_center_x = (left_cheek[0] + right_cheek[0]) / 2.0
        face_width = abs(right_cheek[0] - left_cheek[0])

        if face_width < 1.0:
            self._attention_history.clear()
            return VisualAttentionSnapshot(
                gaze_on_screen=None,
                face_landmarks_detected=True,
                horizontal_gaze_ratio=horizontal_gaze_ratio,
                head_turn_ratio=None,
                head_vertical_ratio=None,
                reason="face_geometry_unreliable",
                left_eye_opening_ratio=left_eye_ratio,
                right_eye_opening_ratio=right_eye_ratio,
                left_eye_quality=0.0,
                right_eye_quality=0.0,
                head_pose_quality=0.0,
                face_box=face_box,
            )

        head_turn_ratio = float((nose[0] - face_center_x) / face_width)
        left_eye_corner = point(LEFT_EYE_CORNERS[0])
        right_eye_corner = point(RIGHT_EYE_CORNERS[1])
        eye_line_y = (left_eye_corner[1] + right_eye_corner[1]) / 2.0
        chin = point(CHIN)
        eye_to_chin = chin[1] - eye_line_y

        if eye_to_chin < 1.0:
            head_vertical_ratio = None
        else:
            head_vertical_ratio = float((nose[1] - eye_line_y) / eye_to_chin)

        head_pose_quality = self._head_pose_quality(
            head_turn_ratio=head_turn_ratio,
            head_vertical_ratio=head_vertical_ratio,
        )
        left_eye_quality = self._eye_quality(
            eye_width_px=left_eye_width,
            head_turn_ratio=head_turn_ratio,
            head_vertical_ratio=head_vertical_ratio,
            is_left=True,
        )
        right_eye_quality = self._eye_quality(
            eye_width_px=right_eye_width,
            head_turn_ratio=head_turn_ratio,
            head_vertical_ratio=head_vertical_ratio,
            is_left=False,
        )

        head_frontal = abs(head_turn_ratio) <= 0.16
        head_vertical_ok = (
            head_vertical_ratio is not None
            and 0.25 <= head_vertical_ratio <= 0.78
        )
        gaze_horizontal_ok = 0.20 <= horizontal_gaze_ratio <= 0.80
        current_attention = head_frontal and head_vertical_ok and gaze_horizontal_ok
        self._attention_history.append(current_attention)
        positive_count = sum(self._attention_history)
        smoothed_attention = positive_count >= (
            len(self._attention_history) // 2 + 1
        )

        if smoothed_attention:
            reason = "looking_toward_screen"
        elif not head_frontal:
            reason = "head_turned_away"
        elif not gaze_horizontal_ok:
            reason = "eyes_turned_away"
        else:
            reason = "head_position_outside_range"

        return VisualAttentionSnapshot(
            gaze_on_screen=smoothed_attention,
            face_landmarks_detected=True,
            horizontal_gaze_ratio=horizontal_gaze_ratio,
            head_turn_ratio=head_turn_ratio,
            head_vertical_ratio=head_vertical_ratio,
            reason=reason,
            left_eye_opening_ratio=left_eye_ratio,
            right_eye_opening_ratio=right_eye_ratio,
            left_eye_quality=left_eye_quality,
            right_eye_quality=right_eye_quality,
            head_pose_quality=head_pose_quality,
            face_box=face_box,
        )

    @staticmethod
    def _empty_snapshot(reason: str) -> VisualAttentionSnapshot:
        return VisualAttentionSnapshot(
            gaze_on_screen=None,
            face_landmarks_detected=False,
            horizontal_gaze_ratio=None,
            head_turn_ratio=None,
            head_vertical_ratio=None,
            reason=reason,
            left_eye_opening_ratio=None,
            right_eye_opening_ratio=None,
            left_eye_quality=0.0,
            right_eye_quality=0.0,
            head_pose_quality=0.0,
        )

    @staticmethod
    def _eye_horizontal_ratio(
        point_function: Callable[[int], np.ndarray],
        corner_indices: tuple[int, int],
        iris_indices: tuple[int, ...],
    ) -> float:
        first_corner = point_function(corner_indices[0])
        second_corner = point_function(corner_indices[1])
        minimum_x = min(first_corner[0], second_corner[0])
        maximum_x = max(first_corner[0], second_corner[0])
        eye_width = maximum_x - minimum_x
        if eye_width < 1.0:
            return 0.5
        iris_points = [point_function(index) for index in iris_indices]
        iris_center_x = float(np.mean([item[0] for item in iris_points]))
        return clamp((iris_center_x - minimum_x) / eye_width, 0.0, 1.0)

    @staticmethod
    def _eye_opening_measurement(
        point_function: Callable[[int], np.ndarray],
        corner_indices: tuple[int, int],
        vertical_pairs: tuple[tuple[int, int], ...],
    ) -> tuple[float | None, float]:
        first_corner = point_function(corner_indices[0])
        second_corner = point_function(corner_indices[1])
        horizontal_distance = float(np.linalg.norm(second_corner - first_corner))
        if horizontal_distance < 3.0:
            return None, horizontal_distance

        vertical_distances = [
            float(
                np.linalg.norm(
                    point_function(upper_index) - point_function(lower_index)
                )
            )
            for upper_index, lower_index in vertical_pairs
        ]
        if not vertical_distances:
            return None, horizontal_distance

        # Медиана слабее реагирует на одну ошибочную landmark-пару.
        vertical_distance = float(np.median(vertical_distances))
        ratio = vertical_distance / horizontal_distance
        if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 0.80:
            return None, horizontal_distance
        return float(ratio), horizontal_distance

    @staticmethod
    def _eye_opening_ratio(
        point_function: Callable[[int], np.ndarray],
        corner_indices: tuple[int, int],
        vertical_pairs: tuple[tuple[int, int], ...],
    ) -> float | None:
        """Совместимый помощник для геометрических unit-тестов."""
        ratio, _ = VisualAttentionAnalyzer._eye_opening_measurement(
            point_function,
            corner_indices,
            vertical_pairs,
        )
        return ratio

    @staticmethod
    def _head_pose_quality(
        *,
        head_turn_ratio: float,
        head_vertical_ratio: float | None,
    ) -> float:
        """Грубая пригодность геометрии, а не "правильность" позы.

        Старый вариант сравнивал всех пользователей с одной вертикальной
        пропорцией 0.52. Это систематически браковало нормальный кадр при
        другой форме лица или высоте камеры. Здесь остаётся только широкий
        контроль сильного поворота и явно вырожденной геометрии. Для
        валидационных сессий используется более точная личная калибровка.
        """
        absolute_turn = abs(float(head_turn_ratio))
        if absolute_turn <= 0.10:
            turn_quality = 1.0
        else:
            turn_quality = clamp(
                1.0 - (absolute_turn - 0.10) / 0.28,
                0.0,
                1.0,
            )

        if head_vertical_ratio is None or not np.isfinite(head_vertical_ratio):
            return 0.0
        vertical = float(head_vertical_ratio)
        if 0.12 <= vertical <= 0.88:
            vertical_quality = 1.0
        elif vertical < 0.12:
            vertical_quality = clamp((vertical - 0.02) / 0.10, 0.0, 1.0)
        else:
            vertical_quality = clamp((0.98 - vertical) / 0.10, 0.0, 1.0)
        return float(min(turn_quality, vertical_quality))

    @staticmethod
    def _eye_quality(
        *,
        eye_width_px: float,
        head_turn_ratio: float,
        head_vertical_ratio: float | None,
        is_left: bool,
    ) -> float:
        size_quality = clamp((eye_width_px - 5.0) / 17.0, 0.0, 1.0)
        # При повороте дальний глаз обычно виден хуже. Знак зависит от
        # зеркального кадра, но асимметричная поправка нужна только как мягкий
        # quality gate, а не как геометрический вывод о закрытии.
        signed_turn = head_turn_ratio if is_left else -head_turn_ratio
        side_quality = clamp(
            1.0 - max(0.0, signed_turn - 0.08) / 0.34,
            0.0,
            1.0,
        )
        # Качество глаза описывает только разрешение глаза и перспективное
        # ухудшение дальней стороны. Вертикальная пропорция лица сюда больше
        # не входит: она индивидуальна и калибруется отдельно.
        return float(0.70 * size_quality + 0.30 * side_quality)

    @staticmethod
    def _combined_eye_opening_ratio(
        left_ratio: float | None,
        right_ratio: float | None,
        *,
        head_turn_ratio: float,
        head_vertical_ratio: float | None,
    ) -> float | None:
        """Старый диагностический метод; детектор больше не использует среднее."""
        if abs(head_turn_ratio) > 0.22:
            return None
        if head_vertical_ratio is None or not 0.18 <= head_vertical_ratio <= 0.85:
            return None
        if left_ratio is None or right_ratio is None:
            return None
        return float((left_ratio + right_ratio) / 2.0)

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> VisualAttentionAnalyzer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
