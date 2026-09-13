from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RecoveryEventStore:
    """Локальный журнал рекомендаций, обратной связи и самооценок."""

    def __init__(self, db_path: str | Path, *, profile_id: int) -> None:
        if profile_id < 1:
            raise ValueError("profile_id должен быть положительным.")
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_id = int(profile_id)
        self._initialize()

    def log_event(
        self,
        event_type: str,
        *,
        recommendation_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        clean_type = event_type.strip().lower()
        if not clean_type:
            raise ValueError("event_type не может быть пустым.")
        payload_json = json.dumps(
            dict(payload or {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recovery_events(
                    profile_id, occurred_at, event_type,
                    recommendation_id, payload_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.profile_id,
                    _utc_now_iso(),
                    clean_type,
                    recommendation_id,
                    payload_json,
                ),
            )
            return int(cursor.lastrowid)

    def add_self_report(
        self,
        *,
        fatigue_sp_1_7: float,
        sleepiness_kss_1_9: float,
    ) -> int:
        fatigue = float(fatigue_sp_1_7)
        sleepiness = float(sleepiness_kss_1_9)
        if not math.isfinite(fatigue) or not fatigue.is_integer() or not 1.0 <= fatigue <= 7.0:
            raise ValueError("fatigue_sp_1_7 должен быть целым числом 1..7.")
        if not math.isfinite(sleepiness) or not sleepiness.is_integer() or not 1.0 <= sleepiness <= 9.0:
            raise ValueError("sleepiness_kss_1_9 должен быть целым числом 1..9.")

        # Старые базы Step 4–6 содержат обязательные поля fatigue_0_10 и
        # sleepiness_1_9. Сохраняем их для совместимости, но источником истины
        # в Step 7 становятся валидированные шкалы SP 1–7 и KSS 1–9.
        legacy_fatigue_0_10 = (fatigue - 1.0) / 6.0 * 10.0
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recovery_self_reports(
                    profile_id, occurred_at, fatigue_0_10, sleepiness_1_9,
                    fatigue_sp_1_7, sleepiness_kss_1_9
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.profile_id,
                    _utc_now_iso(),
                    legacy_fatigue_0_10,
                    sleepiness,
                    fatigue,
                    sleepiness,
                ),
            )
            return int(cursor.lastrowid)

    def event_count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM recovery_events WHERE profile_id = ?",
                (self.profile_id,),
            ).fetchone()
        return int(row["count"])

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS recovery_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    recommendation_id INTEGER,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_recovery_events_profile_time
                    ON recovery_events(profile_id, occurred_at);

                CREATE TABLE IF NOT EXISTS recovery_self_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    fatigue_0_10 REAL NOT NULL,
                    sleepiness_1_9 REAL NOT NULL,
                    fatigue_sp_1_7 REAL,
                    sleepiness_kss_1_9 REAL
                );

                CREATE INDEX IF NOT EXISTS idx_recovery_reports_profile_time
                    ON recovery_self_reports(profile_id, occurred_at);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(recovery_self_reports)")
            }
            if "fatigue_sp_1_7" not in columns:
                connection.execute(
                    "ALTER TABLE recovery_self_reports ADD COLUMN fatigue_sp_1_7 REAL"
                )
            if "sleepiness_kss_1_9" not in columns:
                connection.execute(
                    "ALTER TABLE recovery_self_reports ADD COLUMN sleepiness_kss_1_9 REAL"
                )

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
