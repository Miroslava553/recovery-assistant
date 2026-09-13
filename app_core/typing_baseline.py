from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Iterator, Mapping

from app_core.baseline_engine import (
    BaselineObservation,
    BaselineUpdateResult,
    FeatureChannel,
    PersonalBaselineEngine,
    WorkContext,
)
from app_core.state_machine import UserState
from app_core.typing_activity import TypingMetrics


@dataclass(frozen=True, slots=True)
class TypingBaselineResult:
    """Результат одной попытки добавить окно печати в личную норму."""

    attempted: bool
    accepted: bool
    persisted: bool
    typing_quality: float
    calibration_day_count: int
    calibration_progress: float
    initial_calibration_complete: bool
    accepted_features: tuple[str, ...]
    reason: str


class TypingBaselineService:
    """
    Связывает обезличенные TypingMetrics с PersonalBaselineEngine.

    В базу попадают только агрегаты 60-секундного окна:
    скорость, интервалы, вариативность, паузы, серии печати и доля
    исправлений. Текст и названия клавиш не принимаются этим модулем.

    Сервис намеренно сохраняет окна не чаще одного раза в минуту,
    чтобы почти одинаковые скользящие снимки не раздували baseline.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        profile_id: int,
        baseline: PersonalBaselineEngine | None = None,
        sample_interval_sec: float = 60.0,
        min_observation_sec: float = 50.0,
        target_relevant_keys: int = 25,
    ) -> None:
        if profile_id < 1:
            raise ValueError("profile_id должен быть положительным.")
        if sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec должен быть больше нуля.")
        if min_observation_sec <= 0:
            raise ValueError("min_observation_sec должен быть больше нуля.")
        if target_relevant_keys < 1:
            raise ValueError("target_relevant_keys должен быть не меньше 1.")

        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_id = int(profile_id)
        self.baseline = baseline or PersonalBaselineEngine()
        self.sample_interval_sec = float(sample_interval_sec)
        self.min_observation_sec = float(min_observation_sec)
        self.target_relevant_keys = int(target_relevant_keys)

        self._last_attempt_at: float | None = None

        self._initialize_storage()
        self._restore_persisted_samples()

    def observe(
        self,
        metrics: TypingMetrics,
        *,
        state: UserState,
        workday: date,
        minutes_since_workday_start: float,
        captured_at: datetime | None = None,
        now_monotonic: float | None = None,
        self_report_fatigue_0_10: float | None = None,
        self_report_sleepiness_kss_1_9: float | None = None,
        allow_refresh: bool = False,
    ) -> TypingBaselineResult:
        """
        Пытается принять одно реальное окно печати в личную норму.

        Для typing-baseline подходит только ACTIVE_WORK. PASSIVE_WORK
        является чтением/просмотром и не должен создавать искусственные
        нулевые показатели печати.
        """

        current_monotonic = (
            monotonic() if now_monotonic is None else float(now_monotonic)
        )

        if not math.isfinite(current_monotonic):
            raise ValueError("now_monotonic должен быть конечным числом.")

        if state is not UserState.ACTIVE_WORK:
            return self._result(
                attempted=False,
                accepted=False,
                persisted=False,
                typing_quality=0.0,
                accepted_features=(),
                reason=(
                    "Окно не относится к активному набору текста; "
                    "скорость печати не записывается как нулевая."
                ),
            )

        if metrics.observation_sec < self.min_observation_sec:
            return self._result(
                attempted=False,
                accepted=False,
                persisted=False,
                typing_quality=0.0,
                accepted_features=(),
                reason="Окно печати ещё слишком короткое для личной нормы.",
            )

        if not metrics.data_ready:
            return self._result(
                attempted=False,
                accepted=False,
                persisted=False,
                typing_quality=0.0,
                accepted_features=(),
                reason="В окне недостаточно обезличенных событий печати.",
            )

        if (
            self._last_attempt_at is not None
            and current_monotonic - self._last_attempt_at
            < self.sample_interval_sec
        ):
            seconds_left = self.sample_interval_sec - (
                current_monotonic - self._last_attempt_at
            )
            return self._result(
                attempted=False,
                accepted=False,
                persisted=False,
                typing_quality=0.0,
                accepted_features=(),
                reason=(
                    "Следующее независимое окно печати ещё формируется "
                    f"({math.ceil(seconds_left)} с)."
                ),
            )

        self._last_attempt_at = current_monotonic

        typing_quality = self._typing_quality(metrics)
        features = self.features_from_metrics(metrics)

        observation = BaselineObservation(
            workday=workday,
            minutes_since_workday_start=float(minutes_since_workday_start),
            context=WorkContext.TYPING,
            features=features,
            user_present=True,
            returning=False,
            severe_ocular_event=False,
            self_report_fatigue_0_10=self_report_fatigue_0_10,
            self_report_sleepiness_kss_1_9=(
                self_report_sleepiness_kss_1_9
            ),
            channel_quality={
                FeatureChannel.TYPING: typing_quality,
            },
            liveness_score=None,
        )

        update = self.baseline.observe(
            observation,
            allow_refresh=allow_refresh,
        )

        persisted = False
        if update.accepted_features:
            event_time = captured_at or datetime.now(timezone.utc)
            if event_time.tzinfo is None:
                raise ValueError(
                    "captured_at должен содержать часовой пояс."
                )

            accepted_values = {
                name: float(features[name])
                for name in update.accepted_features
                if features.get(name) is not None
            }

            persisted = self._save_samples(
                captured_at=event_time,
                workday=workday,
                context=WorkContext.TYPING,
                features=accepted_values,
            )

        return self._result_from_update(
            update,
            persisted=persisted,
            typing_quality=typing_quality,
        )

    @staticmethod
    def features_from_metrics(
        metrics: TypingMetrics,
    ) -> dict[str, float | None]:
        """Преобразует TypingMetrics в признаки личной нормы."""

        observation_sec = float(metrics.observation_sec)

        if observation_sec > 0:
            long_pause_rate = (
                float(metrics.long_pause_count) * 60.0 / observation_sec
            )
            burst_rate = (
                float(metrics.burst_count) * 60.0 / observation_sec
            )
        else:
            long_pause_rate = None
            burst_rate = None

        return {
            "typing_keys_per_minute": float(metrics.keys_per_minute),
            "typing_mean_interval_sec": metrics.mean_interval_sec,
            "typing_median_interval_sec": metrics.median_interval_sec,
            "typing_rhythm_cv": metrics.rhythm_cv,
            "typing_correction_ratio": float(metrics.correction_ratio),
            "typing_long_pause_rate_per_min": long_pause_rate,
            "typing_burst_rate_per_min": burst_rate,
        }

    def stored_sample_count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM personal_baseline_samples
                WHERE profile_id = ?
                """,
                (self.profile_id,),
            ).fetchone()
        return int(row["count"])

    def _typing_quality(self, metrics: TypingMetrics) -> float:
        duration_quality = min(
            1.0,
            max(0.0, float(metrics.observation_sec) / 60.0),
        )
        event_quality = min(
            1.0,
            max(
                0.0,
                float(metrics.relevant_key_count)
                / float(self.target_relevant_keys),
            ),
        )

        # Берём более слабую сторону. Длинное окно с двумя клавишами
        # и короткое окно с плотной печатью одинаково ненадёжны.
        return min(duration_quality, event_quality)

    def _initialize_storage(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS personal_baseline_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    workday TEXT NOT NULL,
                    context TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_personal_baseline_sample_unique
                ON personal_baseline_samples(
                    profile_id,
                    captured_at,
                    context,
                    feature_name
                );

                CREATE INDEX IF NOT EXISTS
                    idx_personal_baseline_profile_context
                ON personal_baseline_samples(
                    profile_id,
                    context,
                    feature_name,
                    workday
                );
                """
            )

    def _restore_persisted_samples(self) -> None:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT workday, context, feature_name, feature_value
                FROM personal_baseline_samples
                WHERE profile_id = ?
                ORDER BY captured_at ASC, id ASC
                """,
                (self.profile_id,),
            ).fetchall()

        for row in rows:
            context = WorkContext(str(row["context"]))
            feature_name = str(row["feature_name"])
            feature_value = float(row["feature_value"])
            workday_value = date.fromisoformat(str(row["workday"]))
            channel = self.baseline.feature_channel(feature_name)

            quality: Mapping[FeatureChannel, float] = {channel: 1.0}
            liveness = (
                1.0
                if channel in {FeatureChannel.OCULAR, FeatureChannel.POSTURE}
                else None
            )

            self.baseline.observe(
                BaselineObservation(
                    workday=workday_value,
                    minutes_since_workday_start=0.0,
                    context=context,
                    features={feature_name: feature_value},
                    channel_quality=quality,
                    liveness_score=liveness,
                ),
                allow_refresh=True,
            )

    def _save_samples(
        self,
        *,
        captured_at: datetime,
        workday: date,
        context: WorkContext,
        features: Mapping[str, float],
    ) -> bool:
        if not features:
            return False

        captured_at_utc = captured_at.astimezone(timezone.utc).isoformat(
            timespec="milliseconds"
        )

        rows = []
        for feature_name, feature_value in features.items():
            value = float(feature_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"Признак {feature_name!r} должен быть конечным числом."
                )
            rows.append(
                (
                    self.profile_id,
                    captured_at_utc,
                    workday.isoformat(),
                    context.value,
                    feature_name,
                    value,
                )
            )

        with self._connection() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO personal_baseline_samples(
                    profile_id,
                    captured_at,
                    workday,
                    context,
                    feature_name,
                    feature_value
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return connection.total_changes > before

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _result_from_update(
        self,
        update: BaselineUpdateResult,
        *,
        persisted: bool,
        typing_quality: float,
    ) -> TypingBaselineResult:
        return TypingBaselineResult(
            attempted=True,
            accepted=update.accepted,
            persisted=persisted,
            typing_quality=typing_quality,
            calibration_day_count=update.calibration_day_count,
            calibration_progress=self.baseline.calibration_progress,
            initial_calibration_complete=(
                update.initial_calibration_complete
            ),
            accepted_features=update.accepted_features,
            reason=update.reason,
        )

    def _result(
        self,
        *,
        attempted: bool,
        accepted: bool,
        persisted: bool,
        typing_quality: float,
        accepted_features: tuple[str, ...],
        reason: str,
    ) -> TypingBaselineResult:
        return TypingBaselineResult(
            attempted=attempted,
            accepted=accepted,
            persisted=persisted,
            typing_quality=typing_quality,
            calibration_day_count=self.baseline.calibration_day_count,
            calibration_progress=self.baseline.calibration_progress,
            initial_calibration_complete=(
                self.baseline.initial_calibration_complete
            ),
            accepted_features=accepted_features,
            reason=reason,
        )