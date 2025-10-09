import logging
import sys
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

if TYPE_CHECKING:  # pragma: no cover - imports only for type checkers
    from metta.agent.components.mamba import MambaBackboneConfig
    from metta.agent.policies.mamba_sliding import MambaSlidingConfig

logger = logging.getLogger(__name__)

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


def _load_mamba_components() -> tuple["MambaBackboneConfig", "MambaSlidingConfig"]:
    """Import Mamba components lazily so the recipe can register without CUDA deps."""

    try:
        from metta.agent.components.mamba import MambaBackboneConfig
        from metta.agent.policies.mamba_sliding import MambaSlidingConfig
    except ModuleNotFoundError as exc:  # noqa: PERF203 - clarity over micro-optimisation
        if exc.name == "mamba_ssm":
            raise RuntimeError(
                "Mamba recipes require the `mamba-ssm` package (available on Linux with CUDA)."
                " Install it or run on a supported platform before invoking this recipe."
            ) from exc
        raise

    return MambaBackboneConfig, MambaSlidingConfig


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
    MambaBackboneConfig, MambaSlidingConfig = _load_mamba_components()

    policy = policy_architecture or MambaSlidingConfig()

    if ssm_layer != "Mamba2":
        msg = f"Unsupported SSM layer '{ssm_layer}'. Only 'Mamba2' is available."
        raise ValueError(msg)

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

    trainer = tool.trainer
    trainer.optimizer.learning_rate = learning_rate
    trainer.batch_size = batch_size
    trainer.minibatch_size = minibatch_size
    tool.training_env.forward_pass_minibatch_target_size = (
        forward_pass_minibatch_target_size
    )
    tool.torch_profiler = TorchProfilerConfig(interval_epochs=0)

    return tool


def train_mamba2(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    forward_pass_minibatch_target_size: int = DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE,
) -> TrainTool:
    _load_mamba_components()
    return train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
        ssm_layer="Mamba2",
        learning_rate=learning_rate,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        forward_pass_minibatch_target_size=forward_pass_minibatch_target_size,
    )


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
    "train_mamba2",
]


def _debug_recipe_registration() -> None:
    """Emit diagnostic information about recipe registration.

    When tool discovery fails, printing these lines helps identify whether the recipe
    module imported correctly and whether the registry registered our tool makers.
    """

    try:
        from metta.common.tool.recipe_registry import recipe_registry

        recipe = recipe_registry.get("experiments.recipes.abes.mamba")
        if recipe is None:
            print("[ABES Mamba DEBUG] registry returned None", file=sys.stderr)
            return

        maker_names = sorted(recipe.get_all_tool_maker_names())
        print(f"[ABES Mamba DEBUG] makers={maker_names}", file=sys.stderr)
        train_maker = recipe.get_tool_maker("train")
        print(f"[ABES Mamba DEBUG] has_train={bool(train_maker)}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - debugging-only handler
        print(f"[ABES Mamba DEBUG] registry check failed: {exc}", file=sys.stderr)


_debug_recipe_registration()
