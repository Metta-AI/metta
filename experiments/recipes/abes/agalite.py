"""Arena Basic Easy Shaped recipe targeting AGaLiTe policy variants."""


import typing

import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.agalite
import metta.agent.policy
import metta.cogworks.curriculum.curriculum
import metta.rl.trainer_config
import metta.tools.train

_POLICY_PRESETS: dict[
    str, typing.Callable[[], metta.agent.policy.PolicyArchitecture]
] = {
    "agalite": metta.agent.policies.agalite.AGaLiTeConfig,
}


def _policy_from_name(name: str) -> metta.agent.policy.PolicyArchitecture:
    try:
        return _POLICY_PRESETS[name]()
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(sorted(_POLICY_PRESETS))
        raise ValueError(f"Unknown policy '{name}'. Available: {available}") from exc


def train(
    *,
    curriculum: metta.cogworks.curriculum.curriculum.CurriculumConfig | None = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: metta.agent.policy.PolicyArchitecture | None = None,
    agent: str | None = None,
) -> metta.tools.train.TrainTool:
    if policy_architecture is None:
        if agent is not None:
            policy_architecture = _policy_from_name(agent)
        else:
            policy_architecture = metta.agent.policies.agalite.AGaLiTeConfig()

    tool = experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    hint = getattr(policy_architecture, "learning_rate_hint", None)
    optimizer = tool.trainer.optimizer
    default_lr = metta.rl.trainer_config.OptimizerConfig.model_fields[
        "learning_rate"
    ].default
    if hint is not None and optimizer.learning_rate == default_lr:
        optimizer.learning_rate = hint

    return tool


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep",
    "train",
]
