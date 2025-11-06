"""Easy preset coordination helpers."""

from __future__ import annotations

from ._preset import PRESET

train = PRESET.train_coordination
play = PRESET.play

__all__ = ["train", "play"]
