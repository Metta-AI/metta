"""Sliding-transformer recipe reusing the ABES transformer scaffolding."""

from __future__ import annotations

from typing import Optional

from metta.agent.policies.transformer import (
    TransformerBackboneVariant,
    TransformerPolicyConfig,
)
from metta.agent.policy import PolicyArchitecture
from metta.tools.train import TrainTool

from . import trxl as _trxl

make_mettagrid = _trxl.make_mettagrid
make_curriculum = _trxl.make_curriculum
make_evals = _trxl.make_evals
play = _trxl.play
replay = _trxl.replay
evaluate = _trxl.evaluate
evaluate_in_sweep = _trxl.evaluate_in_sweep


def train(
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Return a training tool configured with the sliding transformer backbone."""

    policy_architecture = policy_architecture or TransformerPolicyConfig(
        variant=TransformerBackboneVariant.SLIDING,
        use_aux_tokens=True,
    )

    return _trxl.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )


__all__ = [
    "make_mettagrid",
    "make_curriculum",
    "make_evals",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "train",
]
