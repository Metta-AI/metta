"""Shared styling helpers for the miniscope renderer."""

from __future__ import annotations

from typing import Iterable, Literal

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# Palette tuned to evoke Claude Code's cool, soft contrast aesthetic.
PRIMARY_ACCENT = "#d5c3ff"
PRIMARY_ACCENT_DIM = "#b8a6ff"
ACCENT_TWILIGHT = "#8f9bff"
SURFACE_BASE = "#11121b"
SURFACE_ALT = "#15182a"
SURFACE_DARK = "#0d0e16"
BORDER_PRIMARY = "#2d3044"
BORDER_ALT = "#3a3d53"


CLAUDE_THEME = Theme(
    {
        "text": "#e7ecff",
        "muted": "#9197b3",
        "muted.dim": "#6f758d",
        "accent": f"bold {PRIMARY_ACCENT}",
        "accent.dim": f"bold {PRIMARY_ACCENT_DIM}",
        "accent.alt": f"bold {ACCENT_TWILIGHT}",
        "chip.primary": f"bold {SURFACE_DARK} on {PRIMARY_ACCENT}",
        "chip.secondary": f"bold {PRIMARY_ACCENT} on {SURFACE_ALT}",
        "chip.negative": "bold #0f1017 on #f29db4",
        "surface": f"on {SURFACE_BASE}",
        "surface.alt": f"on {SURFACE_ALT}",
        "surface.dark": f"on {SURFACE_DARK}",
        "border": BORDER_PRIMARY,
        "border.alt": BORDER_ALT,
        "divider": "#2b2d3f",
        "good": "#7ee7b5",
        "warn": "#fbc77c",
        "bad": "#f29db4",
    }
)


def gradient_title(label: str, *, colors: tuple[str, str] = (PRIMARY_ACCENT, ACCENT_TWILIGHT)) -> Text:
    """Create a gradient title text element."""
    text = Text(label, style="bold")
    apply = getattr(text, "apply_gradient", None)
    if callable(apply):
        apply(*colors)
    else:
        # Gracefully degrade to a solid accent color when gradients are unsupported.
        text.stylize(f"bold {colors[0]}")
    return text


def chip_markup(label: str, *, variant: Literal["primary", "secondary", "negative"] = "primary") -> str:
    """Return markup for a soft pill-shaped shortcut chip."""
    style_name = f"chip.{variant}"
    return f"[{style_name}] {label} [/{style_name}]"


def join_chips(items: Iterable[tuple[str, str]], *, spacer: str = "  ") -> Text:
    """Build a control legend row with chips and muted captions."""
    line = Text()
    first = True
    for chip_label, caption in items:
        if not first:
            line.append(spacer)
        line += Text.from_markup(chip_markup(chip_label))
        if caption:
            line.append(f" {caption}", style="muted")
        first = False
    return line


def surface_panel(
    renderable: RenderableType,
    *,
    title: Text | str | None = None,
    subtitle: Text | str | None = None,
    variant: Literal["base", "alt", "dark"] = "base",
    border_variant: Literal["primary", "alt"] = "primary",
    padding: tuple[int, int] = (0, 1),
    width: int | None = None,
) -> Panel:
    """Wrap content in a softly bordered panel."""

    background_style = {
        "base": "surface",
        "alt": "surface.alt",
        "dark": "surface.dark",
    }[variant]
    border_style = "border" if border_variant == "primary" else "border.alt"

    return Panel(
        renderable,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        style=background_style,
        padding=padding,
        width=width,
    )
