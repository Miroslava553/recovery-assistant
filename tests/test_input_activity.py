import unittest

from app_core.input_activity import (
    UINT32_MASK,
    calculate_elapsed_milliseconds,
)


class InputActivityTests(unittest.TestCase):
    def test_normal_difference(self):
        result = calculate_elapsed_milliseconds(
            current_tick=5000,
            previous_tick=3200,
        )

        self.assertEqual(result, 1800)

    def test_equal_ticks(self):
        result = calculate_elapsed_milliseconds(
            current_tick=1000,
            previous_tick=1000,
        )

        self.assertEqual(result, 0)

    def test_counter_wraparound(self):
        previous_tick = UINT32_MASK - 99
        current_tick = 150

        result = calculate_elapsed_milliseconds(
            current_tick=current_tick,
            previous_tick=previous_tick,
        )

        self.assertEqual(result, 250)

    def test_negative_value_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_elapsed_milliseconds(
                current_tick=-1,
                previous_tick=0,
            )


if __name__ == "__main__":
    unittest.main()