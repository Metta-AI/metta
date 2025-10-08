import logging
from typing import Optional

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
from metta.agent.components.mamba import MambaBackboneConfig
from metta.agent.policies.mamba_sliding import MambaSlidingConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import TorchProfilerConfig
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)

DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_BATCH_SIZE = 131_072
DEFAULT_MINIBATCH_SIZE = 4_096
DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE = 1_024
DEFAULT_SSM_LAYER = "Mamba2"


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
    policy = policy_architecture or MambaSlidingConfig()

    if ssm_layer != "Mamba2":
        msg = f"Unsupported SSM layer '{ssm_layer}'. Only 'Mamba2' is available."
        raise ValueError(msg)

    for component in policy.components:
        if isinstance(component, MambaBackboneConfig):
            ssm_cfg = dict(component.ssm_cfg) if component.ssm_cfg else {}
            ssm_cfg["layer"] = "Mamba2"
            component.ssm_cfg = ssm_cfg

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
            logger.warning("[ABES Mamba] recipe_registry.get returned None during import")
            return

        maker_names = sorted(recipe.get_all_tool_maker_names())
        logger.info("[ABES Mamba] registered tool makers: %s", maker_names)
        train_maker = recipe.get_tool_maker("train")
        logger.info("[ABES Mamba] train tool maker present: %s", bool(train_maker))
    except Exception as exc:  # noqa: BLE001 - debugging-only handler
        logger.exception("[ABES Mamba] recipe registration check failed: %s", exc)


_debug_recipe_registration()
