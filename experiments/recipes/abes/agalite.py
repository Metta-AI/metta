"""Arena Basic Easy Shaped recipe targeting AGaLiTe policy variants."""

from __future__ import annotations

from typing import Callable, Dict

from experiments.recipes import arena_basic_easy_shaped as base
from metta.agent.policies.agalite import AGaLiTeConfig, AGaLiTeImprovedConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.trainer_config import TorchProfilerConfig
from metta.tools.train import TrainTool

make_mettagrid = base.make_mettagrid
make_curriculum = base.make_curriculum
make_evals = base.make_evals
play = base.play
replay = base.replay
evaluate = base.evaluate
evaluate_in_sweep = base.evaluate_in_sweep
sweep_async_progressive = base.sweep_async_progressive

_POLICY_PRESETS: Dict[str, Callable[[], PolicyArchitecture]] = {
    "agalite": AGaLiTeConfig,
    "agalite_improved": AGaLiTeImprovedConfig,
}


def _policy_from_name(name: str) -> PolicyArchitecture:
    try:
        return _POLICY_PRESETS[name]()
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(sorted(_POLICY_PRESETS))
        raise ValueError(f"Unknown policy '{name}'. Available: {available}") from exc


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    agent: str | None = None,
) -> TrainTool:
    if policy_architecture is None:
        if agent is not None:
            policy_architecture = _policy_from_name(agent)
        else:
            policy_architecture = AGaLiTeImprovedConfig()

    tool = base.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    optimizer = tool.trainer.optimizer
    optimizer.learning_rate = 8e-4

    tool.trainer.batch_size = 131072
    tool.trainer.minibatch_size = 4096
    tool.training_env.forward_pass_minibatch_target_size = 1024

    tool.torch_profiler = TorchProfilerConfig(
        interval_epochs=1,
        profile_dir="${run_dir}/torch_traces",
    )

    return tool


__all__ = [
    "make_mettagrid",
    "make_curriculum",
    "make_evals",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep_async_progressive",
    "train",
]
