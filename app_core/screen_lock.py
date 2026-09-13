"""Определение блокировки экрана Windows.

Пока экран заблокирован, рабочее поведение наблюдать бессмысленно: человек
физически не работает, а накопленная непрерывная сессия не должна расти.
`UserStateManager` уже умеет обрабатывать флаг `screen_locked`, но до сих пор
в `SessionMonitor` вместо реального значения жёстко передавался `False`.

Способ определения. Windows не предоставляет прямого «заблокирован ли
экран». Стандартный приём — попытаться открыть текущий input desktop через
`OpenInputDesktop`. На экране блокировки активен отдельный защищённый
desktop (`Winlogon`), к которому обычный процесс доступа не получает,
поэтому вызов возвращает NULL. Заодно так же определяется экранная заставка
с паролем и переключение пользователя.

На не-Windows системах датчик возвращает `None`: «неизвестно», а не «не
заблокирован». Вызывающая сторона сама решает, как трактовать отсутствие
данных.
"""

from __future__ import annotations

import sys

DESKTOP_SWITCHDESKTOP = 0x0100


class ScreenLockSensor:
    """Отвечает на один вопрос: заблокирован ли сейчас экран.

    Датчик не хранит состояние и безопасен при частом опросе. Если
    определить состояние нельзя (другая ОС или ошибка вызова), возвращается
    `None`.
    """

    def __init__(self) -> None:
        self._user32 = None
        self._available = False
        if sys.platform != "win32":
            return
        try:
            import ctypes

            self._user32 = ctypes.windll.user32
            self._available = True
        except (ImportError, AttributeError, OSError):
            self._user32 = None
            self._available = False

    @property
    def available(self) -> bool:
        """Поддерживается ли определение на этой системе."""

        return self._available

    def is_locked(self) -> bool | None:
        """`True` — экран заблокирован, `False` — нет, `None` — неизвестно."""

        if not self._available or self._user32 is None:
            return None

        try:
            handle = self._user32.OpenInputDesktop(0, False, DESKTOP_SWITCHDESKTOP)
        except OSError:
            return None

        if not handle:
            return True

        try:
            self._user32.CloseDesktop(handle)
        except OSError:
            pass
        return False
