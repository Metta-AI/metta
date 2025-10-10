import logging
from typing import Optional, TYPE_CHECKING

from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep_async_progressive,
    train as base_train,
)
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import TorchProfilerConfig
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    pass


DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_BATCH_SIZE = 131_072
DEFAULT_MINIBATCH_SIZE = 4_096
DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE = 1_024
DEFAULT_SSM_LAYER = "Mamba2"


def _supports_mem_eff_path() -> bool:
    """Return True if the fused causal-conv1d kernels are available."""

    try:
        from causal_conv1d import causal_conv1d_fn  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        return False

    return callable(causal_conv1d_fn)


def _apply_overrides(
    tool: TrainTool,
    *,
    learning_rate: float,
    batch_size: int,
    minibatch_size: int,
    forward_pass_minibatch_target_size: int,
) -> None:
    trainer = tool.trainer
    trainer.optimizer.learning_rate = learning_rate
    trainer.batch_size = batch_size
    trainer.minibatch_size = minibatch_size

    tool.training_env.forward_pass_minibatch_target_size = (
        forward_pass_minibatch_target_size
    )
    tool.torch_profiler = TorchProfilerConfig(interval_epochs=0)


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    ssm_layer: str = DEFAULT_SSM_LAYER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    forward_pass_minibatch_target_size: int = DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE,
) -> TrainTool:
    try:
        from metta.agent.components.mamba import MambaBackboneConfig
        from metta.agent.policies.mamba_sliding import MambaSlidingConfig
    except ModuleNotFoundError as exc:
        if exc.name == "mamba_ssm":
            raise RuntimeError(
                "Mamba recipes require the `mamba-ssm` package (available on Linux with CUDA)."
                " Install it on a supported system before running this recipe."
            ) from exc
        raise

    if ssm_layer != DEFAULT_SSM_LAYER:
        msg = f"Unsupported SSM layer '{ssm_layer}'. Only '{DEFAULT_SSM_LAYER}' is available."
        raise ValueError(msg)

    policy = policy_architecture or MambaSlidingConfig()

    mem_eff_supported = _supports_mem_eff_path()
    if not mem_eff_supported:
        logger.warning(
            "[ABES Mamba] Detected missing causal-conv1d CUDA kernels; disabling memory-efficient path."
            " Install `flash-attn` and `causal-conv1d` for best performance."
        )

    for component in policy.components:
        if isinstance(component, MambaBackboneConfig):
            component.ssm_cfg = {**component.ssm_cfg, "layer": ssm_layer}
            component.use_mem_eff_path = bool(
                mem_eff_supported and component.use_mem_eff_path
            )
            if not mem_eff_supported:
                component.auto_align_stride = True

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy,
    )

    _apply_overrides(
        tool,
        learning_rate=learning_rate,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        forward_pass_minibatch_target_size=forward_pass_minibatch_target_size,
    )

    return tool


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep_async_progressive",
    "train",
]
