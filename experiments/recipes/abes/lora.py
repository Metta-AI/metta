"""Arena Basic Easy Shaped recipe that trains LoRA adapters on SmolLLM."""


import typing

import experiments.recipes.abes
import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.smollm
import metta.agent.policy
import metta.cogworks.curriculum.curriculum
import metta.tools.sweep
import metta.tools.train

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def train(
    *,
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
    model_name: typing.Optional[str] = None,
    attn_implementation: typing.Optional[str] = None,
    policy_architecture: typing.Optional[metta.agent.policy.PolicyArchitecture] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: typing.Optional[list[str]] = None,
) -> metta.tools.train.TrainTool:
    if policy_architecture is None:
        config_kwargs = {
            "freeze_llm": True,
            "use_lora": True,
            "attn_implementation": experiments.recipes.abes.smollm._select_attn_implementation(
                attn_implementation
            ),  # noqa: SLF001
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules or DEFAULT_LORA_TARGETS,
            "token_stride": 2,
            "actor_head_rank": None,
            "value_head_rank": None,
        }
        config_kwargs["model_name"] = model_name or "HuggingFaceTB/SmolLM2-135M"
        policy_architecture = metta.agent.policies.smollm.SmolLLMConfig(**config_kwargs)

    resolved_curriculum = curriculum or experiments.recipes.abes.smollm.make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    tool = experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=resolved_curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    return _apply_lora_defaults(tool)


def _apply_lora_defaults(
    tool: metta.tools.train.TrainTool,
) -> metta.tools.train.TrainTool:
    """Adjust training defaults for LoRA fine-tuning."""

    trainer = tool.trainer
    env = tool.training_env

    trainer_updates: dict[str, object] = {}
    env_updates: dict[str, object] = {}

    if trainer.compile:
        trainer_updates["compile"] = False

    bptt_horizon = min(trainer.bptt_horizon, 6)
    if bptt_horizon != trainer.bptt_horizon:
        trainer_updates["bptt_horizon"] = bptt_horizon

    minibatch_size = min(trainer.minibatch_size, 768)
    minibatch_size = max(bptt_horizon, minibatch_size - (minibatch_size % bptt_horizon))
    if minibatch_size != trainer.minibatch_size:
        trainer_updates["minibatch_size"] = minibatch_size

    batch_size = min(trainer.batch_size, 15_360)
    if minibatch_size > 0:
        batch_size = max(minibatch_size, batch_size - (batch_size % minibatch_size))
    if batch_size != trainer.batch_size:
        trainer_updates["batch_size"] = batch_size

    if trainer_updates:
        tool.trainer = trainer.model_copy(update=trainer_updates)

    if env.forward_pass_minibatch_target_size > 384:
        env_updates["forward_pass_minibatch_target_size"] = 384

    if env_updates:
        tool.training_env = env.model_copy(update=env_updates)

    return tool


__all__ = [
    "mettagrid",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep",
    "sweep_async_progressive",
    "train",
]


make_curriculum = experiments.recipes.abes.smollm.make_curriculum


def sweep(
    sweep_name: str,
    **kwargs: object,
) -> metta.tools.sweep.SweepTool:
    """Expose the canonical arena sweep for SmolLLM LoRA recipes."""

    return _delegate_sweep(sweep_name, **kwargs)


def sweep_async_progressive(
    sweep_name: str,
    **kwargs: object,
) -> metta.tools.sweep.SweepTool:
    """Backward-compatible alias maintained for historical CLI usage."""

    return _delegate_sweep(sweep_name, **kwargs)


def _delegate_sweep(sweep_name: str, **kwargs: object) -> metta.tools.sweep.SweepTool:
    return experiments.recipes.arena_basic_easy_shaped.sweep(sweep_name, **kwargs)
