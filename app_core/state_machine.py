from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import monotonic


class UserState(StrEnum):
    """
    Текущее состояние пользователя относительно рабочего места.

    Эти состояния не являются оценкой усталости.
    Они только определяют, какие каналы данных сейчас можно анализировать.
    """

    UNKNOWN = "unknown"
    ACTIVE_WORK = "active_work"
    PASSIVE_WORK = "passive_work"
    AWAY = "away"
    BREAK = "break"
    RETURNING = "returning"


@dataclass(frozen=True, slots=True)
class PresenceSignals:
    """Обезличенные сигналы одного момента времени."""

    face_detected: bool | None
    input_idle_sec: float | None
    gaze_on_screen: bool | None
    data_quality: float
    screen_locked: bool = False
    manual_break: bool = False

    def __post_init__(self) -> None:
        if self.input_idle_sec is not None and self.input_idle_sec < 0:
            raise ValueError("input_idle_sec не может быть отрицательным.")

        if not 0.0 <= self.data_quality <= 1.0:
            raise ValueError("data_quality должен находиться в диапазоне 0..1.")


@dataclass(frozen=True, slots=True)
class StateDecision:
    state: UserState
    previous_state: UserState
    changed: bool
    reason: str
    fatigue_analysis_allowed: bool
    baseline_update_allowed: bool
    seconds_in_state: float


class UserStateManager:
    """
    Детерминированный автомат состояний пользователя.

    Важные правила:
    - отсутствие движения мыши само по себе не означает усталость;
    - недавний реальный ввод подтверждает активную работу, даже когда
      визуальный канал временно недоступен;
    - AWAY требует одновременно долгого отсутствия лица и долгого
      отсутствия ввода;
    - после возвращения действует защитный период RETURNING.
    """

    def __init__(
        self,
        *,
        active_input_threshold_sec: float = 8.0,
        away_input_threshold_sec: float = 20.0,
        away_face_threshold_sec: float = 12.0,
        returning_duration_sec: float = 60.0,
        low_quality_threshold: float = 0.45,
        transition_debounce_sec: float = 2.0,
    ) -> None:
        positive_values = {
            "active_input_threshold_sec": active_input_threshold_sec,
            "away_input_threshold_sec": away_input_threshold_sec,
            "away_face_threshold_sec": away_face_threshold_sec,
            "returning_duration_sec": returning_duration_sec,
            "transition_debounce_sec": transition_debounce_sec,
        }

        for name, value in positive_values.items():
            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным.")

        if away_input_threshold_sec < active_input_threshold_sec:
            raise ValueError(
                "away_input_threshold_sec не должен быть меньше "
                "active_input_threshold_sec."
            )

        if not 0.0 <= low_quality_threshold <= 1.0:
            raise ValueError(
                "low_quality_threshold должен находиться в диапазоне 0..1."
            )

        self.active_input_threshold_sec = float(active_input_threshold_sec)
        self.away_input_threshold_sec = float(away_input_threshold_sec)
        self.away_face_threshold_sec = float(away_face_threshold_sec)
        self.returning_duration_sec = float(returning_duration_sec)
        self.low_quality_threshold = float(low_quality_threshold)
        self.transition_debounce_sec = float(transition_debounce_sec)

        now = monotonic()
        self._state = UserState.UNKNOWN
        self._state_since = now
        self._candidate_state: UserState | None = None
        self._candidate_since: float | None = None
        self._face_missing_since: float | None = None
        self._returning_until: float | None = None

    @property
    def state(self) -> UserState:
        return self._state

    def reset(self, *, now: float | None = None) -> None:
        current_time = monotonic() if now is None else float(now)
        self._state = UserState.UNKNOWN
        self._state_since = current_time
        self._candidate_state = None
        self._candidate_since = None
        self._face_missing_since = None
        self._returning_until = None

    def update(
        self,
        signals: PresenceSignals,
        *,
        now: float | None = None,
    ) -> StateDecision:
        current_time = monotonic() if now is None else float(now)

        if signals.face_detected is False:
            if self._face_missing_since is None:
                self._face_missing_since = current_time
        else:
            self._face_missing_since = None

        previous_state = self._state
        target_state, reason, immediate = self._classify(signals, current_time)

        if immediate:
            self._commit_state(target_state, current_time)
        elif (
            previous_state in {UserState.AWAY, UserState.BREAK}
            and target_state in {UserState.ACTIVE_WORK, UserState.PASSIVE_WORK}
        ):
            self._returning_until = current_time + self.returning_duration_sec
            self._commit_state(UserState.RETURNING, current_time)
            reason = (
                "Пользователь вернулся. Идёт защитный период повторной "
                "калибровки."
            )
        elif self._state is UserState.RETURNING:
            if (
                self._returning_until is not None
                and current_time < self._returning_until
            ):
                target_state = UserState.RETURNING
                reason = "Идёт защитный период после возвращения."
            else:
                self._returning_until = None
                self._apply_debounced_state(target_state, current_time)
        else:
            self._apply_debounced_state(target_state, current_time)

        changed = self._state is not previous_state

        fatigue_allowed = self._state in {
            UserState.ACTIVE_WORK,
            UserState.PASSIVE_WORK,
        }

        # Этот общий флаг относится только к визуально подтверждённым
        # окнам. Клавиатурный baseline проверяет качество своего канала
        # отдельно и не зависит от качества камеры.
        baseline_allowed = (
            fatigue_allowed
            and signals.face_detected is True
            and signals.data_quality >= 0.80
            and not signals.screen_locked
            and not signals.manual_break
        )

        return StateDecision(
            state=self._state,
            previous_state=previous_state,
            changed=changed,
            reason=reason,
            fatigue_analysis_allowed=fatigue_allowed,
            baseline_update_allowed=baseline_allowed,
            seconds_in_state=max(0.0, current_time - self._state_since),
        )

    def _classify(
        self,
        signals: PresenceSignals,
        current_time: float,
    ) -> tuple[UserState, str, bool]:
        if signals.manual_break:
            return (
                UserState.BREAK,
                "Пользователь вручную включил перерыв.",
                True,
            )

        if signals.screen_locked:
            return (
                UserState.AWAY,
                "Экран заблокирован; анализ рабочего поведения остановлен.",
                True,
            )

        input_known = signals.input_idle_sec is not None
        input_recent = (
            input_known
            and signals.input_idle_sec <= self.active_input_threshold_sec
        )
        input_long_idle = (
            input_known
            and signals.input_idle_sec >= self.away_input_threshold_sec
        )

        face_missing_long = (
            signals.face_detected is False
            and self._face_missing_since is not None
            and current_time - self._face_missing_since
            >= self.away_face_threshold_sec
        )

        if face_missing_long and input_long_idle:
            return (
                UserState.AWAY,
                "Лицо отсутствует и давно нет ввода: пользователь отошёл.",
                False,
            )

        # Активный системный ввод — самостоятельный канал присутствия.
        # Плохая камера должна отключать визуальные признаки, но не
        # превращать реальную печать в UNKNOWN.
        if input_recent:
            if signals.face_detected is True:
                reason = "Есть недавний ввод с мыши или клавиатуры."
            else:
                reason = (
                    "Есть недавний ввод; активная работа подтверждена, "
                    "но визуальный канал временно недоступен."
                )
            return UserState.ACTIVE_WORK, reason, False

        if (
            signals.face_detected is True
            and signals.data_quality >= self.low_quality_threshold
        ):
            if signals.gaze_on_screen is True:
                return (
                    UserState.PASSIVE_WORK,
                    "Пользователь присутствует и смотрит на рабочую область, "
                    "но сейчас почти не использует мышь и клавиатуру.",
                    False,
                )

            if input_known and not input_long_idle:
                return (
                    UserState.PASSIVE_WORK,
                    "Пользователь присутствует; короткая пауза во вводе "
                    "может быть чтением или размышлением.",
                    False,
                )

            return (
                UserState.UNKNOWN,
                "Пользователь виден, но недостаточно данных, чтобы отличить "
                "пассивную работу от паузы.",
                False,
            )

        if (
            signals.face_detected is None
            and signals.input_idle_sec is None
        ):
            return (
                UserState.UNKNOWN,
                "Датчики присутствия ещё не предоставили данных.",
                False,
            )

        if signals.data_quality < self.low_quality_threshold:
            return (
                UserState.UNKNOWN,
                "Качество изображения слишком низкое для визуальной оценки, "
                "а недавний ввод не обнаружен.",
                False,
            )

        return (
            UserState.UNKNOWN,
            "Недостаточно согласованных данных о присутствии и активности.",
            False,
        )

    def _apply_debounced_state(
        self,
        target_state: UserState,
        current_time: float,
    ) -> None:
        if target_state is self._state:
            self._candidate_state = None
            self._candidate_since = None
            return

        if target_state is not self._candidate_state:
            self._candidate_state = target_state
            self._candidate_since = current_time
            return

        if self._candidate_since is None:
            self._candidate_since = current_time
            return

        if current_time - self._candidate_since >= self.transition_debounce_sec:
            self._commit_state(target_state, current_time)

    def _commit_state(
        self,
        new_state: UserState,
        current_time: float,
    ) -> None:
        if new_state is not self._state:
            self._state = new_state
            self._state_since = current_time

        self._candidate_state = None
        self._candidate_since = None