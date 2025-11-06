import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.gtrxl
import metta.agent.policies.transformer
import metta.agent.policy
import metta.rl.trainer_config

DEFAULT_LR = metta.rl.trainer_config.OptimizerConfig.model_fields[
    "learning_rate"
].default


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: metta.agent.policy.PolicyArchitecture | None = None,
):
    if policy_architecture is None:
        policy_architecture = metta.agent.policies.transformer.TransformerPolicyConfig(
            transformer=metta.agent.policies.gtrxl.GTrXLConfig()
        )

    tool = experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture,
    )

    if isinstance(
        policy_architecture, metta.agent.policies.transformer.TransformerPolicyConfig
    ):
        hint = policy_architecture.learning_rate_hint
        optimizer = tool.trainer.optimizer
        if hint is not None and optimizer.learning_rate == DEFAULT_LR:
            optimizer.learning_rate = hint

    return tool


if "mettagrid" not in globals():
    mettagrid = experiments.recipes.arena_basic_easy_shaped.mettagrid
if "make_curriculum" not in globals():
    make_curriculum = experiments.recipes.arena_basic_easy_shaped.make_curriculum
if "simulations" not in globals():
    simulations = experiments.recipes.arena_basic_easy_shaped.simulations
if "play" not in globals():
    play = experiments.recipes.arena_basic_easy_shaped.play
if "replay" not in globals():
    replay = experiments.recipes.arena_basic_easy_shaped.replay
if "evaluate" not in globals():
    evaluate = experiments.recipes.arena_basic_easy_shaped.evaluate
if "evaluate_in_sweep" not in globals():
    evaluate_in_sweep = experiments.recipes.arena_basic_easy_shaped.evaluate_in_sweep
if "sweep" not in globals():
    sweep = experiments.recipes.arena_basic_easy_shaped.sweep


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
