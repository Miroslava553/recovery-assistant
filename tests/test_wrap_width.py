"""Ширина переноса строки.

Текст обрезался с обеих сторон, потому что ширина переноса считалась
подобранными долями от ширины окна, а не по фактической разметке. Эти
тесты закрепляют новое правило: перенос никогда не шире того места,
которое метке действительно досталось.
"""

from __future__ import annotations

import unittest

from ui.formatting import wrap_width


class WrapWidthTests(unittest.TestCase):
    def test_unlaid_out_container_gives_no_answer(self):
        """До первой раскладки tkinter сообщает ширину 1."""

        self.assertIsNone(wrap_width(1))
        self.assertIsNone(wrap_width(0))

    def test_wrap_never_exceeds_the_container(self):
        self.assertLessEqual(wrap_width(300), 300)

    def test_neighbours_take_their_share(self):
        """Плашка канала стоит на той же строке и место отнимает."""

        alone = wrap_width(400)
        with_pill = wrap_width(400, occupied=(90,), gap=9)

        self.assertEqual(alone - with_pill, 99)

    def test_wider_pill_leaves_less_room(self):
        """«Самооценка» шире, чем «сессия», — доля на всех не годится."""

        narrow = wrap_width(400, occupied=(60,), gap=9)
        wide = wrap_width(400, occupied=(120,), gap=9)

        self.assertGreater(narrow, wide)

    def test_content_limit_caps_a_runaway_card(self):
        """Широкий текст не должен разрешать себе ещё большую ширину."""

        self.assertLessEqual(wrap_width(5000, limit=700), 700)

    def test_limit_is_ignored_when_not_known_yet(self):
        self.assertGreater(wrap_width(600, limit=0), 500)

    def test_scaling_is_divided_out(self):
        """CustomTkinter умножит значение на масштаб экрана сам."""

        self.assertAlmostEqual(wrap_width(600, scaling=1.5), int(598 / 1.5))

    def test_result_never_collapses_to_nothing(self):
        """Даже когда соседи съели всё, перенос по одной букве не нужен."""

        self.assertGreaterEqual(wrap_width(100, occupied=(200,), gap=9), 80)
