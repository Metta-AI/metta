"""ABES recipe wrapper for the sliding transformer variant."""

from .. import abes_sliding_transformer as _base

make_mettagrid = _base.make_mettagrid
make_curriculum = _base.make_curriculum
make_evals = _base.make_evals
train = _base.train
play = _base.play
replay = _base.replay
evaluate = _base.evaluate
evaluate_in_sweep = _base.evaluate_in_sweep

__all__ = [
    "make_mettagrid",
    "make_curriculum",
    "make_evals",
    "train",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
]
