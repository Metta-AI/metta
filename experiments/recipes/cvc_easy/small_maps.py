"""Small-map easy preset entry points."""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.recipes.cvc_easy import DEFAULT_VARIANTS, play as preset_play, train_small_maps as preset_train


def train(*, variants: Optional[Sequence[str]] = None, **kwargs):
    return preset_train(variants=variants, **kwargs)


def play(*, variants: Optional[Sequence[str]] = None, **kwargs):
    return preset_play(variants=variants, **kwargs)

__all__ = ["train", "play"]
