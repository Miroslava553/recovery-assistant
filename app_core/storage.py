from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping


SCHEMA_VERSION = 2


def utc_now_iso() -> str:
    """Возвращает текущее время в UTC в ISO-формате."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppDatabase:
    """
    Локальное хранилище профилей, рабочих сессий, окон метрик и базовых линий.

    Хранилище намеренно не содержит введённый текст, названия клавиш,
    скриншоты, видеозаписи или содержимое документов.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
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

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    context TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    FOREIGN KEY (profile_id) REFERENCES profiles(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS metric_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    duration_sec REAL NOT NULL CHECK (duration_sec > 0),
                    context TEXT NOT NULL,
                    data_quality REAL NOT NULL
                        CHECK (data_quality >= 0 AND data_quality <= 1),
                    fatigue_flag INTEGER NOT NULL DEFAULT 0
                        CHECK (fatigue_flag IN (0, 1)),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS metric_values (
                    window_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    PRIMARY KEY (window_id, metric_name),
                    FOREIGN KEY (window_id) REFERENCES metric_windows(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS baselines (
                    profile_id INTEGER NOT NULL,
                    context TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    reference_value REAL NOT NULL,
                    median_value REAL NOT NULL,
                    q25_value REAL NOT NULL,
                    q75_value REAL NOT NULL,
                    mad_value REAL NOT NULL,
                    sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
                    method TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (profile_id, context, metric_name),
                    FOREIGN KEY (profile_id) REFERENCES profiles(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_profile_context
                    ON sessions(profile_id, context);

                CREATE INDEX IF NOT EXISTS idx_windows_session_started
                    ON metric_windows(session_id, started_at);

                CREATE INDEX IF NOT EXISTS idx_metric_values_name
                    ON metric_values(metric_name);

                CREATE TABLE IF NOT EXISTS user_state_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    session_id INTEGER,
                    occurred_at TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES profiles(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_state_events_profile_time
                    ON user_state_events(profile_id, occurred_at);
                """
            )

            connection.execute(
                """
                INSERT INTO app_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def schema_version(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM app_meta WHERE key = 'schema_version'"
            ).fetchone()
        if row is None:
            raise RuntimeError("В базе отсутствует версия схемы.")
        return int(row["value"])

    def get_or_create_profile(self, name: str = "Основной пользователь") -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Имя профиля не может быть пустым.")

        with self._connection() as connection:
            row = connection.execute(
                "SELECT id FROM profiles WHERE name = ?",
                (clean_name,),
            ).fetchone()
            if row is not None:
                return int(row["id"])

            cursor = connection.execute(
                "INSERT INTO profiles(name, created_at) VALUES (?, ?)",
                (clean_name, utc_now_iso()),
            )
            return int(cursor.lastrowid)

    def start_session(self, profile_id: int, context: str) -> int:
        clean_context = context.strip().lower()
        if not clean_context:
            raise ValueError("Контекст рабочей сессии не может быть пустым.")

        with self._connection() as connection:
            profile = connection.execute(
                "SELECT id FROM profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            if profile is None:
                raise ValueError(f"Профиль с id={profile_id} не найден.")

            cursor = connection.execute(
                """
                INSERT INTO sessions(profile_id, started_at, context, status)
                VALUES (?, ?, ?, 'running')
                """,
                (profile_id, utc_now_iso(), clean_context),
            )
            return int(cursor.lastrowid)

    def end_session(self, session_id: int, status: str = "completed") -> None:
        clean_status = status.strip().lower()
        if clean_status not in {"completed", "cancelled", "failed"}:
            raise ValueError("Статус должен быть completed, cancelled или failed.")

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET ended_at = ?, status = ?
                WHERE id = ? AND ended_at IS NULL
                """,
                (utc_now_iso(), clean_status, session_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("Активная сессия не найдена или уже завершена.")

    @staticmethod
    def _validate_metrics(metrics: Mapping[str, float | int]) -> dict[str, float]:
        if not metrics:
            raise ValueError("Нельзя сохранить пустой набор метрик.")

        validated: dict[str, float] = {}
        for raw_name, raw_value in metrics.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Название метрики не может быть пустым.")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError(
                    f"Метрика '{name}' должна быть числом, получено: "
                    f"{type(raw_value).__name__}."
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"Метрика '{name}' содержит недопустимое значение: {value}."
                )
            validated[name] = value
        return validated

    def add_metric_window(
        self,
        session_id: int,
        *,
        started_at: str,
        duration_sec: float,
        context: str,
        data_quality: float,
        fatigue_flag: bool,
        metrics: Mapping[str, float | int],
    ) -> int:
        if duration_sec <= 0:
            raise ValueError("duration_sec должен быть больше нуля.")
        if not 0.0 <= data_quality <= 1.0:
            raise ValueError("data_quality должен находиться в диапазоне 0..1.")

        clean_context = context.strip().lower()
        if not clean_context:
            raise ValueError("Контекст окна метрик не может быть пустым.")

        validated_metrics = self._validate_metrics(metrics)

        with self._connection() as connection:
            session = connection.execute(
                "SELECT id, ended_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError(f"Сессия с id={session_id} не найдена.")
            if session["ended_at"] is not None:
                raise ValueError("Нельзя добавлять данные в завершённую сессию.")

            cursor = connection.execute(
                """
                INSERT INTO metric_windows(
                    session_id, started_at, duration_sec, context,
                    data_quality, fatigue_flag
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    started_at,
                    float(duration_sec),
                    clean_context,
                    float(data_quality),
                    int(bool(fatigue_flag)),
                ),
            )
            window_id = int(cursor.lastrowid)

            connection.executemany(
                """
                INSERT INTO metric_values(window_id, metric_name, metric_value)
                VALUES (?, ?, ?)
                """,
                [
                    (window_id, name, value)
                    for name, value in validated_metrics.items()
                ],
            )
            return window_id

    def get_metric_samples(
        self,
        profile_id: int,
        *,
        context: str,
        metric_name: str,
        limit: int = 1000,
        min_quality: float = 0.80,
        exclude_fatigued: bool = True,
    ) -> list[float]:
        if limit <= 0:
            raise ValueError("limit должен быть больше нуля.")
        if not 0.0 <= min_quality <= 1.0:
            raise ValueError("min_quality должен находиться в диапазоне 0..1.")

        clean_context = context.strip().lower()
        clean_metric_name = metric_name.strip()
        if not clean_context or not clean_metric_name:
            raise ValueError("Контекст и имя метрики не могут быть пустыми.")

        fatigue_filter = "AND w.fatigue_flag = 0" if exclude_fatigued else ""
        query = f"""
            SELECT v.metric_value
            FROM metric_values AS v
            JOIN metric_windows AS w ON w.id = v.window_id
            JOIN sessions AS s ON s.id = w.session_id
            WHERE s.profile_id = ?
              AND w.context = ?
              AND v.metric_name = ?
              AND w.data_quality >= ?
              {fatigue_filter}
            ORDER BY w.started_at DESC
            LIMIT ?
        """

        with self._connection() as connection:
            rows = connection.execute(
                query,
                (
                    profile_id,
                    clean_context,
                    clean_metric_name,
                    float(min_quality),
                    int(limit),
                ),
            ).fetchall()

        return [float(row["metric_value"]) for row in reversed(rows)]

    def upsert_baseline(
        self,
        profile_id: int,
        *,
        context: str,
        metric_name: str,
        reference_value: float,
        median_value: float,
        q25_value: float,
        q75_value: float,
        mad_value: float,
        sample_count: int,
        method: str,
    ) -> None:
        if sample_count < 0:
            raise ValueError("sample_count не может быть отрицательным.")

        values_to_check = {
            "reference_value": reference_value,
            "median_value": median_value,
            "q25_value": q25_value,
            "q75_value": q75_value,
            "mad_value": mad_value,
        }
        for name, value in values_to_check.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} должно быть конечным числом.")

        clean_context = context.strip().lower()
        clean_metric_name = metric_name.strip()
        clean_method = method.strip()
        if not clean_context or not clean_metric_name or not clean_method:
            raise ValueError(
                "Контекст, имя метрики и метод расчёта не могут быть пустыми."
            )

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO baselines(
                    profile_id, context, metric_name, reference_value,
                    median_value, q25_value, q75_value, mad_value,
                    sample_count, method, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, context, metric_name)
                DO UPDATE SET
                    reference_value = excluded.reference_value,
                    median_value = excluded.median_value,
                    q25_value = excluded.q25_value,
                    q75_value = excluded.q75_value,
                    mad_value = excluded.mad_value,
                    sample_count = excluded.sample_count,
                    method = excluded.method,
                    updated_at = excluded.updated_at
                """,
                (
                    profile_id,
                    clean_context,
                    clean_metric_name,
                    float(reference_value),
                    float(median_value),
                    float(q25_value),
                    float(q75_value),
                    float(mad_value),
                    int(sample_count),
                    clean_method,
                    utc_now_iso(),
                ),
            )

    def get_baseline(
        self,
        profile_id: int,
        *,
        context: str,
        metric_name: str,
    ) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    profile_id, context, metric_name, reference_value,
                    median_value, q25_value, q75_value, mad_value,
                    sample_count, method, updated_at
                FROM baselines
                WHERE profile_id = ? AND context = ? AND metric_name = ?
                """,
                (
                    profile_id,
                    context.strip().lower(),
                    metric_name.strip(),
                ),
            ).fetchone()
        return dict(row) if row is not None else None


def record_state_event(
    self,
    profile_id: int,
    *,
    previous_state: str,
    new_state: str,
    reason: str,
    session_id: int | None = None,
    occurred_at: str | None = None,
) -> int:
    clean_previous = previous_state.strip()
    clean_new = new_state.strip()
    clean_reason = reason.strip()

    if not clean_previous or not clean_new or not clean_reason:
        raise ValueError(
            "Состояния и причина перехода не могут быть пустыми."
        )

    with self._connection() as connection:
        profile = connection.execute(
            "SELECT id FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()

        if profile is None:
            raise ValueError(f"Профиль с id={profile_id} не найден.")

        if session_id is not None:
            session = connection.execute(
                "SELECT id FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError(f"Сессия с id={session_id} не найдена.")

        cursor = connection.execute(
            """
            INSERT INTO user_state_events(
                profile_id,
                session_id,
                occurred_at,
                previous_state,
                new_state,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                session_id,
                occurred_at or utc_now_iso(),
                clean_previous,
                clean_new,
                clean_reason,
            ),
        )

        return int(cursor.lastrowid)

def list_state_events(
    self,
    profile_id: int,
    *,
    limit: int = 100,
) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError("limit должен быть больше нуля.")

    with self._connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                profile_id,
                session_id,
                occurred_at,
                previous_state,
                new_state,
                reason
            FROM user_state_events
            WHERE profile_id = ?
            ORDER BY occurred_at DESC, id DESC
            LIMIT ?
            """,
            (profile_id, int(limit)),
        ).fetchall()

    return [dict(row) for row in rows]
