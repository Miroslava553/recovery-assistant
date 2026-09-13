from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app_core.recovery_storage import RecoveryEventStore


class RecoveryEventStoreTests(unittest.TestCase):
    def test_events_and_self_reports_are_saved_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "assistant.sqlite3"
            store = RecoveryEventStore(db_path, profile_id=1)
            event_id = store.log_event(
                "recommendation_created",
                recommendation_id=7,
                payload={"kind": "microbreak"},
            )
            report_id = store.add_self_report(
                fatigue_sp_1_7=5,
                sleepiness_kss_1_9=7,
            )
            self.assertGreater(event_id, 0)
            self.assertGreater(report_id, 0)
            self.assertEqual(store.event_count(), 1)

            with closing(sqlite3.connect(db_path)) as connection, connection:
                row = connection.execute(
                    "SELECT fatigue_sp_1_7, sleepiness_kss_1_9 FROM recovery_self_reports"
                ).fetchone()
            self.assertEqual(row, (5.0, 7.0))

    def test_invalid_self_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RecoveryEventStore(Path(temp_dir) / "db.sqlite3", profile_id=1)
            with self.assertRaises(ValueError):
                store.add_self_report(fatigue_sp_1_7=8, sleepiness_kss_1_9=4)

    def test_old_schema_is_migrated_without_deleting_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "old.sqlite3"
            with closing(sqlite3.connect(db_path)) as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE recovery_self_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        profile_id INTEGER NOT NULL,
                        occurred_at TEXT NOT NULL,
                        fatigue_0_10 REAL NOT NULL,
                        sleepiness_1_9 REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO recovery_self_reports(profile_id, occurred_at, fatigue_0_10, sleepiness_1_9) VALUES (1, 'old', 4, 3)"
                )
            store = RecoveryEventStore(db_path, profile_id=1)
            store.add_self_report(fatigue_sp_1_7=4, sleepiness_kss_1_9=6)
            with closing(sqlite3.connect(db_path)) as connection, connection:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(recovery_self_reports)")
                }
                count = connection.execute("SELECT COUNT(*) FROM recovery_self_reports").fetchone()[0]
            self.assertIn("fatigue_sp_1_7", columns)
            self.assertIn("sleepiness_kss_1_9", columns)
            self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
