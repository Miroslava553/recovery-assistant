import unittest

from app_core.typing_activity import (
    GlobalKeyboardTimingSensor,
    TypingActivityCollector,
    TypingEventCategory,
    VK_A,
    VK_BACKSPACE,
    VK_DELETE,
    VK_ENTER,
    VK_SPACE,
)


class TypingActivityCollectorTests(unittest.TestCase):
    def make_collector(self) -> TypingActivityCollector:
        collector = TypingActivityCollector(
            window_sec=60.0,
            long_pause_sec=2.0,
            burst_gap_sec=1.5,
            min_observation_sec=10.0,
            min_relevant_keys=4,
        )

        collector.reset(now=0.0)
        return collector

    def test_empty_snapshot(self):
        collector = self.make_collector()
        metrics = collector.snapshot(now=10.0)

        self.assertEqual(metrics.relevant_key_count, 0)
        self.assertEqual(metrics.keys_per_minute, 0.0)
        self.assertIsNone(metrics.mean_interval_sec)
        self.assertFalse(metrics.data_ready)

    def test_rate_and_intervals(self):
        collector = self.make_collector()

        for timestamp in [1.0, 2.0, 3.0, 4.0]:
            collector.record_event(
                TypingEventCategory.TEXT,
                timestamp=timestamp,
            )

        metrics = collector.snapshot(now=10.0)

        self.assertEqual(metrics.text_key_count, 4)
        self.assertAlmostEqual(metrics.keys_per_minute, 24.0)
        self.assertAlmostEqual(metrics.mean_interval_sec, 1.0)
        self.assertAlmostEqual(metrics.median_interval_sec, 1.0)
        self.assertEqual(metrics.burst_count, 1)
        self.assertTrue(metrics.data_ready)

    def test_long_pause_creates_new_burst(self):
        collector = self.make_collector()

        for timestamp in [1.0, 1.3, 1.6, 5.0]:
            collector.record_event(
                TypingEventCategory.TEXT,
                timestamp=timestamp,
            )

        metrics = collector.snapshot(now=10.0)

        self.assertEqual(metrics.long_pause_count, 1)
        self.assertEqual(metrics.burst_count, 2)
        self.assertAlmostEqual(metrics.longest_pause_sec, 3.4)

    def test_correction_ratio(self):
        collector = self.make_collector()

        collector.record_event(
            TypingEventCategory.TEXT,
            timestamp=1.0,
        )
        collector.record_event(
            TypingEventCategory.TEXT,
            timestamp=2.0,
        )
        collector.record_event(
            TypingEventCategory.CORRECTION,
            timestamp=3.0,
        )
        collector.record_event(
            TypingEventCategory.CORRECTION,
            timestamp=4.0,
        )

        metrics = collector.snapshot(now=10.0)

        self.assertEqual(metrics.correction_key_count, 2)
        self.assertAlmostEqual(metrics.correction_ratio, 0.5)

    def test_old_event_leaves_window(self):
        collector = self.make_collector()

        collector.record_event(
            TypingEventCategory.TEXT,
            timestamp=1.0,
        )
        collector.record_event(
            TypingEventCategory.TEXT,
            timestamp=65.0,
        )

        metrics = collector.snapshot(now=65.0)

        self.assertEqual(metrics.relevant_key_count, 1)

    def test_out_of_order_event_is_rejected(self):
        collector = self.make_collector()

        collector.record_event(
            TypingEventCategory.TEXT,
            timestamp=5.0,
        )

        with self.assertRaises(ValueError):
            collector.record_event(
                TypingEventCategory.TEXT,
                timestamp=4.0,
            )


class VirtualKeyClassificationTests(unittest.TestCase):
    def test_letter_is_text(self):
        result = GlobalKeyboardTimingSensor._classify_virtual_key(
            VK_A
        )

        self.assertEqual(
            result,
            TypingEventCategory.TEXT,
        )

    def test_space_and_enter_are_text(self):
        self.assertEqual(
            GlobalKeyboardTimingSensor._classify_virtual_key(
                VK_SPACE
            ),
            TypingEventCategory.TEXT,
        )

        self.assertEqual(
            GlobalKeyboardTimingSensor._classify_virtual_key(
                VK_ENTER
            ),
            TypingEventCategory.TEXT,
        )

    def test_backspace_and_delete_are_corrections(self):
        self.assertEqual(
            GlobalKeyboardTimingSensor._classify_virtual_key(
                VK_BACKSPACE
            ),
            TypingEventCategory.CORRECTION,
        )

        self.assertEqual(
            GlobalKeyboardTimingSensor._classify_virtual_key(
                VK_DELETE
            ),
            TypingEventCategory.CORRECTION,
        )

    def test_shift_is_ignored(self):
        virtual_key_shift = 0x10

        result = GlobalKeyboardTimingSensor._classify_virtual_key(
            virtual_key_shift
        )

        self.assertEqual(
            result,
            TypingEventCategory.IGNORED,
        )


if __name__ == "__main__":
    unittest.main()