from __future__ import annotations

import ctypes
import platform
import time
from ctypes import wintypes
from dataclasses import dataclass


UINT32_MASK = 0xFFFFFFFF


class InputActivityError(RuntimeError):
    """Ошибка чтения локальной активности мыши и клавиатуры."""


@dataclass(frozen=True, slots=True)
class InputActivitySnapshot:
    """
    Обезличенный снимок активности ввода.

    idle_seconds:
        Количество секунд с последнего действия мыши или клавиатуры.

    recent_input:
        Было ли действие ввода недавно.

    captured_at:
        Локальная монотонная отметка времени.
    """

    idle_seconds: float
    recent_input: bool
    captured_at: float


class _LastInputInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("dwTime", wintypes.DWORD),
    ]


def calculate_elapsed_milliseconds(
    current_tick: int,
    previous_tick: int,
) -> int:
    """
    Вычисляет разницу между двумя 32-битными счётчиками Windows.

    Маска нужна потому, что 32-битный счётчик примерно раз в 49,7 дня
    переходит через ноль.
    """

    if current_tick < 0 or previous_tick < 0:
        raise ValueError("Значения счётчика не могут быть отрицательными.")

    return (current_tick - previous_tick) & UINT32_MASK


class WindowsInputActivitySensor:
    """
    Читает время последнего системного ввода через WinAPI.

    Датчик не устанавливает клавиатурные перехватчики, не сохраняет
    клавиши, координаты мыши, текст, клики или содержимое программ.
    """

    def __init__(self, recent_threshold_sec: float = 8.0) -> None:
        if recent_threshold_sec < 0:
            raise ValueError(
                "recent_threshold_sec не может быть отрицательным."
            )

        if platform.system() != "Windows":
            raise OSError(
                "WindowsInputActivitySensor работает только в Windows."
            )

        self.recent_threshold_sec = float(recent_threshold_sec)

        self._user32 = ctypes.WinDLL(
            "user32",
            use_last_error=True,
        )
        self._kernel32 = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )

        self._user32.GetLastInputInfo.argtypes = [
            ctypes.POINTER(_LastInputInfo)
        ]
        self._user32.GetLastInputInfo.restype = wintypes.BOOL

        self._kernel32.GetTickCount64.argtypes = []
        self._kernel32.GetTickCount64.restype = ctypes.c_ulonglong

    def get_idle_seconds(self) -> float:
        """
        Возвращает число секунд с последнего действия мыши или клавиатуры.
        """

        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)

        success = self._user32.GetLastInputInfo(
            ctypes.byref(info)
        )

        if not success:
            error_code = ctypes.get_last_error()
            raise InputActivityError(
                "Windows не предоставила время последнего ввода. "
                f"Код ошибки: {error_code}."
            )

        current_tick_64 = int(
            self._kernel32.GetTickCount64()
        )

        # LASTINPUTINFO хранит только младшие 32 бита времени.
        current_tick_32 = current_tick_64 & UINT32_MASK
        previous_tick_32 = int(info.dwTime)

        idle_milliseconds = calculate_elapsed_milliseconds(
            current_tick=current_tick_32,
            previous_tick=previous_tick_32,
        )

        return idle_milliseconds / 1000.0

    def snapshot(self) -> InputActivitySnapshot:
        """
        Создаёт один обезличенный снимок активности.
        """

        idle_seconds = self.get_idle_seconds()

        return InputActivitySnapshot(
            idle_seconds=idle_seconds,
            recent_input=(
                idle_seconds <= self.recent_threshold_sec
            ),
            captured_at=time.monotonic(),
        )