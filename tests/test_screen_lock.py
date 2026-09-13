from __future__ import annotations

import unittest

from app_core.screen_lock import ScreenLockSensor


class FakeUser32:
    """Подставной user32 без обращения к настоящему Windows API."""

    def __init__(self, *, handle: int, raises: bool = False) -> None:
        self.handle = handle
        self.raises = raises
        self.closed: list[int] = []

    def OpenInputDesktop(self, flags, inherit, access):  # noqa: N802 - имя WinAPI
        if self.raises:
            raise OSError("desktop unavailable")
        return self.handle

    def CloseDesktop(self, handle):  # noqa: N802 - имя WinAPI
        self.closed.append(handle)
        return True


def sensor_with(user32: FakeUser32) -> ScreenLockSensor:
    sensor = ScreenLockSensor()
    sensor._user32 = user32
    sensor._available = True
    return sensor


class ScreenLockSensorTests(unittest.TestCase):
    def test_null_handle_means_locked(self) -> None:
        """На экране блокировки активен защищённый desktop Winlogon."""

        sensor = sensor_with(FakeUser32(handle=0))
        self.assertIs(sensor.is_locked(), True)

    def test_valid_handle_means_unlocked(self) -> None:
        sensor = sensor_with(FakeUser32(handle=12345))
        self.assertIs(sensor.is_locked(), False)

    def test_open_desktop_handle_is_released(self) -> None:
        """Незакрытый handle — утечка ресурса при опросе раз в секунду."""

        user32 = FakeUser32(handle=999)
        sensor = sensor_with(user32)
        sensor.is_locked()
        sensor.is_locked()
        self.assertEqual(user32.closed, [999, 999])

    def test_api_error_is_unknown_not_locked(self) -> None:
        sensor = sensor_with(FakeUser32(handle=1, raises=True))
        self.assertIsNone(sensor.is_locked())

    def test_unsupported_platform_returns_unknown(self) -> None:
        sensor = ScreenLockSensor()
        sensor._available = False
        sensor._user32 = None
        self.assertIsNone(sensor.is_locked())
        self.assertFalse(sensor.available)


if __name__ == "__main__":
    unittest.main()
