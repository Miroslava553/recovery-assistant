"""Палитра и шрифты нового интерфейса.

Цвета собраны в одном месте намеренно: подбор оттенков — задача UX, и он
меняется отдельно от логики. Менять внешний вид здесь можно свободно, на
работу движка решений это не влияет.
"""

from __future__ import annotations

# --- основа --------------------------------------------------------------
BG = "#1C1C1A"
CARD = "#2C2C2A"
CARD_MUTED = "#242422"
BORDER = "#444441"

TEXT = "#F1EFE8"
TEXT_SECONDARY = "#B4B2A9"
TEXT_DIM = "#888780"
TEXT_FAINT = "#5F5E5A"

# --- пять уровней нагрузки ----------------------------------------------
# Для каждого: фон карточки, цвет заполненных сегментов, подпись, заголовок.
LEVEL_COLORS: dict[int, dict[str, str]] = {
    1: {"bg": "#04342C", "accent": "#5DCAA5", "label": "#9FE1CB", "title": "#E1F5EE"},
    2: {"bg": "#1F3A10", "accent": "#A3C544", "label": "#C9DE8B", "title": "#EAF3D5"},
    3: {"bg": "#3A3406", "accent": "#D9BA2A", "label": "#EBD57B", "title": "#F6EFCE"},
    4: {"bg": "#412402", "accent": "#EF9F27", "label": "#FAC775", "title": "#FAEEDA"},
    5: {"bg": "#4A1210", "accent": "#E24B4A", "label": "#F0999B", "title": "#F7E2E1"},
}

LEVEL_EMPTY = "#5F5E5A"

# --- качество сигнала камеры --------------------------------------------
SIGNAL_COLORS: dict[str, str] = {
    "good": "#5DCAA5",
    "warn": "#EF9F27",
    "bad": "#E24B4A",
    "off": "#888780",
}

SIGNAL_HINT_COLORS: dict[str, str] = {
    "good": TEXT_DIM,
    "warn": "#FAC775",
    "bad": "#F7C1C1",
    "off": TEXT_DIM,
}

# --- метки каналов наблюдения -------------------------------------------
CHANNEL_COLORS: dict[str, dict[str, str]] = {
    "session": {"bg": "#633806", "text": "#FAEEDA"},
    "typing": {"bg": "#0C447C", "text": "#E6F1FB"},
    "self_report": {"bg": "#3E2A5E", "text": "#DCC9F0"},
    "ocular": {"bg": "#444441", "text": "#888780"},
}

# --- рекомендация --------------------------------------------------------
ACTION_BG = "#04342C"
ACTION_LABEL = "#9FE1CB"
ACTION_TITLE = "#E1F5EE"
ACTION_PRIMARY = "#5DCAA5"
ACTION_PRIMARY_HOVER = "#7FD8BA"
ACTION_PRIMARY_TEXT = "#04342C"
ACTION_BORDER = "#0F6E56"

BADGE_ON_BG = "#085041"
BADGE_ON_TEXT = "#9FE1CB"
BADGE_OFF_BG = "#3A3A38"
BADGE_OFF_TEXT = "#B4B2A9"

# --- шрифты --------------------------------------------------------------
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"

CAMERA_BORDER = "#8A5A16"
CAMERA_BG = "#171715"

RADIUS_CARD = 12
RADIUS_PILL = 999
RADIUS_BUTTON = 8
