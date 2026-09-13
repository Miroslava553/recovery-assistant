"""Разбор метрик для панели технических показателей.

Панель собирала поля через `vars()`. У `TypingMetrics` и `OcularMetrics`
объявлен `slots=True`, словаря атрибутов у них нет, и вызов падал с
TypeError. Панель не открывалась ни разу, а исключение выходило наружу и
останавливало цикл опроса: на экране появлялось «Ошибка датчика», хотя
датчики были исправны.

Тест падает на старом коде.
"""

from __future__ import annotations

import unittest
from dataclasses import fields

from app_core.ocular_activity import OcularMetrics
from app_core.typing_activity import TypingMetrics
from ui.formatting import dataclass_items


class DataclassItemsTests(unittest.TestCase):
    def test_vars_would_still_fail_on_these_metrics(self):
        """Фиксируем причину: у метрик нет словаря атрибутов."""

        with self.assertRaises(TypeError):
            vars(TypingMetrics.__new__(TypingMetrics))

    def test_typing_metrics_are_readable(self):
        metrics = TypingMetrics(
            observation_sec=60.0,
            data_ready=True,
            text_key_count=120,
            correction_key_count=6,
            relevant_key_count=126,
            keys_per_minute=126.0,
            mean_interval_sec=0.48,
            median_interval_sec=0.45,
            interval_std_sec=0.12,
            rhythm_cv=0.25,
            longest_pause_sec=4.0,
            long_pause_count=1,
            burst_count=3,
            correction_ratio=0.05,
            current_idle_sec=0.3,
        )
        items = dict(dataclass_items(metrics))

        self.assertEqual(items["text_key_count"], 120)
        self.assertEqual(items["correction_key_count"], 6)

    def test_fields_come_back_sorted(self):
        """Порядок устойчивый: панель не должна прыгать между обновлениями."""

        metrics = TypingMetrics(
            observation_sec=1.0,
            data_ready=False,
            text_key_count=0,
            correction_key_count=0,
            relevant_key_count=0,
            keys_per_minute=0.0,
            mean_interval_sec=None,
            median_interval_sec=None,
            interval_std_sec=None,
            rhythm_cv=None,
            longest_pause_sec=None,
            long_pause_count=0,
            burst_count=0,
            correction_ratio=0.0,
            current_idle_sec=None,
        )
        names = [name for name, _ in dataclass_items(metrics)]

        self.assertEqual(names, sorted(names))

    def test_missing_metrics_do_not_raise(self):
        """Глазной канал может быть выключен, и метрик не будет вовсе."""

        self.assertEqual(dataclass_items(None), [("нет данных", "—")])

    def test_ocular_metrics_are_readable(self):
        """Тридцать три поля глазного канала тоже читаются без vars()."""

        metrics = OcularMetrics.__new__(OcularMetrics)
        for field in fields(OcularMetrics):
            object.__setattr__(metrics, field.name, None)

        items = dict(dataclass_items(metrics))

        self.assertIn("perclos", items)
        self.assertIn("blink_count", items)
