from __future__ import annotations

import ctypes
import platform
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


IS_WINDOWS = platform.system() == "Windows"

WH_KEYBOARD_LL = 13
HC_ACTION = 0

WM_KEYUP = 0x0101
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012
PM_NOREMOVE = 0x0000

LLKHF_INJECTED = 0x00000010

VK_BACKSPACE = 0x08
VK_TAB = 0x09
VK_ENTER = 0x0D
VK_SPACE = 0x20
VK_DELETE = 0x2E

VK_0 = 0x30
VK_9 = 0x39
VK_A = 0x41
VK_Z = 0x5A

VK_NUMPAD0 = 0x60
VK_NUMPAD9 = 0x69
VK_MULTIPLY = 0x6A
VK_ADD = 0x6B
VK_SEPARATOR = 0x6C
VK_SUBTRACT = 0x6D
VK_DECIMAL = 0x6E
VK_DIVIDE = 0x6F

OEM_TEXT_KEYS = {
    0xBA,
    0xBB,
    0xBC,
    0xBD,
    0xBE,
    0xBF,
    0xC0,
    0xDB,
    0xDC,
    0xDD,
    0xDE,
    0xDF,
    0xE2,
}


if IS_WINDOWS:
    from ctypes import wintypes

    LRESULT = ctypes.c_ssize_t
    WPARAM = ctypes.c_size_t
    LPARAM = ctypes.c_ssize_t
    ULONG_PTR = ctypes.c_size_t
    HHOOK = wintypes.HANDLE

    LowLevelKeyboardProcedure = ctypes.WINFUNCTYPE(
        LRESULT,
        ctypes.c_int,
        WPARAM,
        LPARAM,
    )

    class KeyboardHookData(ctypes.Structure):
        _fields_ = [
            ("vkCode", wintypes.DWORD),
            ("scanCode", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

else:
    LowLevelKeyboardProcedure = None

    class KeyboardHookData(ctypes.Structure):
        pass


class TypingEventCategory(StrEnum):
    TEXT = "text"
    CORRECTION = "correction"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class TypingMetrics:
    observation_sec: float
    data_ready: bool

    text_key_count: int
    correction_key_count: int
    relevant_key_count: int

    keys_per_minute: float

    mean_interval_sec: float | None
    median_interval_sec: float | None
    interval_std_sec: float | None
    rhythm_cv: float | None

    longest_pause_sec: float | None
    long_pause_count: int
    burst_count: int

    correction_ratio: float
    current_idle_sec: float | None


@dataclass(frozen=True, slots=True)
class _TypingEvent:
    timestamp: float
    category: TypingEventCategory


class TypingActivityCollector:
    """
    Обезличенный анализ ритма печати.

    Хранятся только:
    - время события;
    - категория события: текст или исправление.

    Символы, названия клавиш, слова и текст не сохраняются.
    """

    def __init__(
        self,
        *,
        window_sec: float = 60.0,
        long_pause_sec: float = 2.0,
        burst_gap_sec: float = 1.5,
        min_observation_sec: float = 15.0,
        min_relevant_keys: int = 20,
    ) -> None:
        numeric_values = {
            "window_sec": window_sec,
            "long_pause_sec": long_pause_sec,
            "burst_gap_sec": burst_gap_sec,
            "min_observation_sec": min_observation_sec,
        }

        for name, value in numeric_values.items():
            if value <= 0:
                raise ValueError(
                    f"{name} должен быть больше нуля."
                )

        if min_relevant_keys < 1:
            raise ValueError(
                "min_relevant_keys должен быть не меньше 1."
            )

        if min_observation_sec > window_sec:
            raise ValueError(
                "min_observation_sec не должен превышать window_sec."
            )

        self.window_sec = float(window_sec)
        self.long_pause_sec = float(long_pause_sec)
        self.burst_gap_sec = float(burst_gap_sec)
        self.min_observation_sec = float(min_observation_sec)
        self.min_relevant_keys = int(min_relevant_keys)

        self._events: deque[_TypingEvent] = deque()
        self._lock = threading.Lock()
        self._started_at = time.monotonic()

    def reset(
        self,
        *,
        now: float | None = None,
    ) -> None:
        current_time = (
            time.monotonic()
            if now is None
            else float(now)
        )

        with self._lock:
            self._events.clear()
            self._started_at = current_time

    def record_event(
        self,
        category: TypingEventCategory,
        *,
        timestamp: float | None = None,
    ) -> None:
        if category is TypingEventCategory.IGNORED:
            return

        event_time = (
            time.monotonic()
            if timestamp is None
            else float(timestamp)
        )

        with self._lock:
            if (
                self._events
                and event_time < self._events[-1].timestamp
            ):
                raise ValueError(
                    "События должны поступать "
                    "в правильном порядке времени."
                )

            self._events.append(
                _TypingEvent(
                    timestamp=event_time,
                    category=category,
                )
            )

            self._discard_old_events(event_time)

    def snapshot(
        self,
        *,
        now: float | None = None,
    ) -> TypingMetrics:
        current_time = (
            time.monotonic()
            if now is None
            else float(now)
        )

        with self._lock:
            self._discard_old_events(current_time)
            events = list(self._events)
            started_at = self._started_at

        elapsed_since_start = max(
            0.0,
            current_time - started_at,
        )

        observation_sec = min(
            self.window_sec,
            elapsed_since_start,
        )

        text_count = sum(
            event.category is TypingEventCategory.TEXT
            for event in events
        )

        correction_count = sum(
            event.category
            is TypingEventCategory.CORRECTION
            for event in events
        )

        relevant_count = text_count + correction_count

        if observation_sec > 0:
            keys_per_minute = (
                relevant_count
                * 60.0
                / observation_sec
            )
        else:
            keys_per_minute = 0.0

        timestamps = [
            event.timestamp
            for event in events
        ]

        intervals = [
            later - earlier
            for earlier, later in zip(
                timestamps,
                timestamps[1:],
            )
        ]

        if intervals:
            mean_interval = statistics.fmean(
                intervals
            )

            median_interval = statistics.median(
                intervals
            )

            interval_std = (
                statistics.pstdev(intervals)
                if len(intervals) > 1
                else 0.0
            )

            rhythm_cv = (
                interval_std / mean_interval
                if mean_interval > 0
                else 0.0
            )

            longest_pause = max(intervals)

            long_pause_count = sum(
                interval >= self.long_pause_sec
                for interval in intervals
            )

            burst_count = 1 + sum(
                interval >= self.burst_gap_sec
                for interval in intervals
            )

        else:
            mean_interval = None
            median_interval = None
            interval_std = None
            rhythm_cv = None
            longest_pause = None
            long_pause_count = 0
            burst_count = 1 if relevant_count else 0

        correction_ratio = (
            correction_count / relevant_count
            if relevant_count
            else 0.0
        )

        current_idle_sec = (
            max(
                0.0,
                current_time - timestamps[-1],
            )
            if timestamps
            else None
        )

        data_ready = (
            observation_sec >= self.min_observation_sec
            and relevant_count >= self.min_relevant_keys
        )

        return TypingMetrics(
            observation_sec=observation_sec,
            data_ready=data_ready,
            text_key_count=text_count,
            correction_key_count=correction_count,
            relevant_key_count=relevant_count,
            keys_per_minute=keys_per_minute,
            mean_interval_sec=mean_interval,
            median_interval_sec=median_interval,
            interval_std_sec=interval_std,
            rhythm_cv=rhythm_cv,
            longest_pause_sec=longest_pause,
            long_pause_count=long_pause_count,
            burst_count=burst_count,
            correction_ratio=correction_ratio,
            current_idle_sec=current_idle_sec,
        )

    def _discard_old_events(
        self,
        current_time: float,
    ) -> None:
        cutoff = current_time - self.window_sec

        while (
            self._events
            and self._events[0].timestamp < cutoff
        ):
            self._events.popleft()


class GlobalKeyboardTimingSensor:
    """
    Системный обезличенный датчик клавиатуры Windows.

    Конкретный код клавиши используется только на мгновение,
    чтобы определить категорию события.

    В сборщик передаётся только:
    - время;
    - обычное нажатие или исправление.
    """

    def __init__(
        self,
        collector: TypingActivityCollector | None = None,
    ) -> None:
        if not IS_WINDOWS:
            raise OSError(
                "Датчик клавиатуры работает только в Windows."
            )

        self.collector = (
            collector
            or TypingActivityCollector()
        )

        self._user32 = ctypes.WinDLL(
            "user32",
            use_last_error=True,
        )

        self._kernel32 = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )

        self._configure_windows_api()

        self._hook: Any | None = None
        self._hook_callback: Any | None = None
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None

        self._ready_event = threading.Event()
        self._start_error: BaseException | None = None

        self._state_lock = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    def start(self) -> None:
        with self._state_lock:
            if self._running:
                return

            self._ready_event.clear()
            self._start_error = None

            self._thread = threading.Thread(
                target=self._run_message_loop,
                name="keyboard-timing-sensor",
                daemon=True,
            )

            self._thread.start()

        if not self._ready_event.wait(timeout=4.0):
            raise RuntimeError(
                "Датчик клавиатуры не успел запуститься."
            )

        if self._start_error is not None:
            raise RuntimeError(
                "Не удалось запустить датчик клавиатуры."
            ) from self._start_error

    def stop(self) -> None:
        with self._state_lock:
            thread = self._thread
            thread_id = self._thread_id

        if thread is None:
            return

        if thread_id is not None:
            self._user32.PostThreadMessageW(
                thread_id,
                WM_QUIT,
                0,
                0,
            )

        thread.join(timeout=4.0)

        if thread.is_alive():
            raise RuntimeError(
                "Не удалось корректно остановить "
                "датчик клавиатуры."
            )

        with self._state_lock:
            self._thread = None
            self._thread_id = None
            self._running = False

    def snapshot(self) -> TypingMetrics:
        return self.collector.snapshot()

    def _run_message_loop(self) -> None:
        try:
            self._thread_id = int(
                self._kernel32.GetCurrentThreadId()
            )

            message = wintypes.MSG()

            # Создаём очередь сообщений для потока.
            self._user32.PeekMessageW(
                ctypes.byref(message),
                None,
                0,
                0,
                PM_NOREMOVE,
            )

            self._hook_callback = (
                LowLevelKeyboardProcedure(
                    self._keyboard_procedure
                )
            )

            module_handle = (
                self._kernel32.GetModuleHandleW(None)
            )

            self._hook = (
                self._user32.SetWindowsHookExW(
                    WH_KEYBOARD_LL,
                    self._hook_callback,
                    module_handle,
                    0,
                )
            )

            if not self._hook:
                error_code = ctypes.get_last_error()

                raise OSError(
                    error_code,
                    "Windows не разрешила подключить "
                    "датчик клавиатуры.",
                )

            with self._state_lock:
                self._running = True

            self._ready_event.set()

            while True:
                result = self._user32.GetMessageW(
                    ctypes.byref(message),
                    None,
                    0,
                    0,
                )

                if result == -1:
                    error_code = ctypes.get_last_error()

                    raise OSError(
                        error_code,
                        "Ошибка системного цикла сообщений.",
                    )

                if result == 0:
                    break

                self._user32.TranslateMessage(
                    ctypes.byref(message)
                )

                self._user32.DispatchMessageW(
                    ctypes.byref(message)
                )

        except BaseException as error:
            self._start_error = error
            self._ready_event.set()

        finally:
            if self._hook:
                self._user32.UnhookWindowsHookEx(
                    self._hook
                )

            self._hook = None
            self._hook_callback = None

            with self._state_lock:
                self._running = False

    def _keyboard_procedure(
        self,
        code: int,
        message_type: int,
        event_pointer: int,
    ) -> int:
        try:
            if (
                code == HC_ACTION
                and int(message_type)
                in {WM_KEYUP, WM_SYSKEYUP}
            ):
                event = ctypes.cast(
                    event_pointer,
                    ctypes.POINTER(
                        KeyboardHookData
                    ),
                ).contents

                # Автоматически созданные программами
                # события не учитываем.
                if not (
                    int(event.flags)
                    & LLKHF_INJECTED
                ):
                    category = (
                        self._classify_virtual_key(
                            int(event.vkCode)
                        )
                    )

                    if (
                        category
                        is not TypingEventCategory.IGNORED
                    ):
                        self.collector.record_event(
                            category
                        )

        except Exception:
            # Исключение нельзя выпускать наружу
            # из системной callback-функции.
            pass

        return int(
            self._user32.CallNextHookEx(
                self._hook,
                code,
                message_type,
                event_pointer,
            )
        )

    @staticmethod
    def _classify_virtual_key(
        virtual_key: int,
    ) -> TypingEventCategory:
        if virtual_key in {
            VK_BACKSPACE,
            VK_DELETE,
        }:
            return TypingEventCategory.CORRECTION

        if virtual_key in {
            VK_TAB,
            VK_ENTER,
            VK_SPACE,
        }:
            return TypingEventCategory.TEXT

        if VK_0 <= virtual_key <= VK_9:
            return TypingEventCategory.TEXT

        if VK_A <= virtual_key <= VK_Z:
            return TypingEventCategory.TEXT

        if (
            VK_NUMPAD0
            <= virtual_key
            <= VK_NUMPAD9
        ):
            return TypingEventCategory.TEXT

        if virtual_key in {
            VK_MULTIPLY,
            VK_ADD,
            VK_SEPARATOR,
            VK_SUBTRACT,
            VK_DECIMAL,
            VK_DIVIDE,
        }:
            return TypingEventCategory.TEXT

        if virtual_key in OEM_TEXT_KEYS:
            return TypingEventCategory.TEXT

        return TypingEventCategory.IGNORED

    def _configure_windows_api(self) -> None:
        self._user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int,
            LowLevelKeyboardProcedure,
            wintypes.HINSTANCE,
            wintypes.DWORD,
        ]

        self._user32.SetWindowsHookExW.restype = HHOOK

        self._user32.CallNextHookEx.argtypes = [
            HHOOK,
            ctypes.c_int,
            WPARAM,
            LPARAM,
        ]

        self._user32.CallNextHookEx.restype = LRESULT

        self._user32.UnhookWindowsHookEx.argtypes = [
            HHOOK
        ]

        self._user32.UnhookWindowsHookEx.restype = (
            wintypes.BOOL
        )

        self._user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]

        self._user32.GetMessageW.restype = (
            wintypes.BOOL
        )

        self._user32.PeekMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]

        self._user32.PeekMessageW.restype = (
            wintypes.BOOL
        )

        self._user32.TranslateMessage.argtypes = [
            ctypes.POINTER(wintypes.MSG)
        ]

        self._user32.TranslateMessage.restype = (
            wintypes.BOOL
        )

        self._user32.DispatchMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG)
        ]

        self._user32.DispatchMessageW.restype = (
            LRESULT
        )

        self._user32.PostThreadMessageW.argtypes = [
            wintypes.DWORD,
            wintypes.UINT,
            WPARAM,
            LPARAM,
        ]

        self._user32.PostThreadMessageW.restype = (
            wintypes.BOOL
        )

        self._kernel32.GetCurrentThreadId.argtypes = []
        self._kernel32.GetCurrentThreadId.restype = (
            wintypes.DWORD
        )

        self._kernel32.GetModuleHandleW.argtypes = [
            wintypes.LPCWSTR
        ]

        self._kernel32.GetModuleHandleW.restype = (
            wintypes.HMODULE
        )

    def __enter__(
        self,
    ) -> GlobalKeyboardTimingSensor:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.stop()