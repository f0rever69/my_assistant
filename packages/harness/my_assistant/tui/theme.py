"""Restrained colour + symbol palette for the TUI.

A black and charcoal base keeps the terminal visually quiet, with distinct
accent hues for speakers and tool state. Rich-compatible hex colours let the
same constants drive both Rich renderables and Textual CSS variables.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    bg: str = "#080a0d"
    panel: str = "#111419"
    border: str = "#292e36"
    text: str = "#d7dbe2"
    dim: str = "#777f8a"
    muted: str = "#9aa1ab"

    primary: str = "#7dcfff"  # headings / app accent
    user: str = "#7aa2f7"  # user speaker
    assistant: str = "#c0caf5"  # assistant speaker
    tool: str = "#bb9af7"  # tool activity
    accent: str = "#9ece6a"  # success / ok
    warning: str = "#e0af68"  # running / caution
    error: str = "#f7768e"  # errors


THEME = Theme()

SYMBOLS = {
    "user": "›",
    "assistant": "●",
    "tool": "⚙",
    "running": "◐",
    "ok": "✓",
    "error": "✗",
    "system": "·",
    "spinner": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
}
