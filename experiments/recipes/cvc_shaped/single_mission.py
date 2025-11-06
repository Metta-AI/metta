"""Shaped preset single-mission helpers."""

from __future__ import annotations

from ._preset import PRESET

train = PRESET.train_single_mission
play = PRESET.play

__all__ = ["train", "play"]
