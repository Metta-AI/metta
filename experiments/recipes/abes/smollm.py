"""Arena Basic Easy Shaped recipe targeting the SmolLLM policy."""

import importlib.util
import typing

import metta.cogworks.curriculum as cc
import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.smollm
import metta.agent.policy
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.tools.train
import metta.tools.sweep
import mettagrid

_FLASH_ATTENTION_AVAILABLE = importlib.util.find_spec("flash_attn") is not None


def _select_attn_implementation(
    attn_implementation: typing.Optional[str],
) -> typing.Optional[str]:
    """Prefer FlashAttention when installed; otherwise fall back to eager attention."""

    if attn_implementation is not None:
        return attn_implementation
    if _FLASH_ATTENTION_AVAILABLE:
        return "flash_attention_2"
    return None


def train(
    *,
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
    freeze_llm: bool = True,
    model_name: typing.Optional[str] = None,
    attn_implementation: typing.Optional[str] = None,
    policy_architecture: typing.Optional[metta.agent.policy.PolicyArchitecture] = None,
):
    if policy_architecture is None:
        config_kwargs = {
            "freeze_llm": freeze_llm,
            "attn_implementation": _select_attn_implementation(attn_implementation),
            "token_stride": 2,
            "actor_head_rank": 64,
            "value_head_rank": 8,
        }
        config_kwargs["model_name"] = model_name or "HuggingFaceTB/SmolLM2-135M"
        policy_architecture = metta.agent.policies.smollm.SmolLLMConfig(**config_kwargs)

    resolved_curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    tool = experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=resolved_curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    return _apply_smollm_defaults(tool)


def _apply_smollm_defaults(
    tool: metta.tools.train.TrainTool,
) -> metta.tools.train.TrainTool:
    """Clamp heavy training defaults to keep SmolLLM within memory limits."""

    trainer = tool.trainer
    env = tool.training_env

    trainer_updates: dict[str, object] = {}
    env_updates: dict[str, object] = {}

    if trainer.compile:
        trainer_updates["compile"] = False

    if trainer.batch_size > 131_072:
        trainer_updates["batch_size"] = 131_072

    if trainer.minibatch_size > 4_096:
        trainer_updates["minibatch_size"] = 4_096

    if trainer.bptt_horizon > 4:
        trainer_updates["bptt_horizon"] = 4

    if trainer_updates:
        tool.trainer = trainer.model_copy(update=trainer_updates)

    if env.forward_pass_minibatch_target_size > 2_048:
        env_updates["forward_pass_minibatch_target_size"] = 2_048
    if env.auto_workers:
        env_updates["auto_workers"] = False
    if env.num_workers != 1:
        env_updates["num_workers"] = 1
    if env.async_factor != 1:
        env_updates["async_factor"] = 1

    if env_updates:
        tool.training_env = env.model_copy(update=env_updates)

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
    "sweep",
    "train",
]


def make_curriculum(
    arena_env: typing.Optional[mettagrid.MettaGridConfig] = None,
    *,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumAlgorithmConfig
    ] = None,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    """SmolLLM-friendly arena curriculum that avoids legacy converter fields."""

    env = arena_env or experiments.recipes.arena_basic_easy_shaped.mettagrid()
    tasks = cc.bucketed(env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    if algorithm_config is None:
        algorithm_config = metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return tasks.to_curriculum(algorithm_config=algorithm_config)


def sweep(
    sweep_name: str,
    **kwargs: object,
) -> metta.tools.sweep.SweepTool:
    """Expose the canonical arena sweep for SmolLLM recipes."""

    return _delegate_sweep(sweep_name, **kwargs)


def sweep_async_progressive(
    sweep_name: str,
    **kwargs: object,
) -> metta.tools.sweep.SweepTool:
    """Backward-compatible alias retained for historical CLI invocations."""

    return _delegate_sweep(sweep_name, **kwargs)


def _delegate_sweep(sweep_name: str, **kwargs: object) -> metta.tools.sweep.SweepTool:
    return experiments.recipes.arena_basic_easy_shaped.sweep(sweep_name, **kwargs)



for _name in [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep",
]:
    if _name in globals():
        continue
    globals()[_name] = getattr(experiments.recipes.arena_basic_easy_shaped, _name)

del _name
