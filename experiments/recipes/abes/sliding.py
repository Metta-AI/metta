# ruff: noqa: E501
import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.vit_sliding_trans
import metta.agent.policy


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: metta.agent.policy.PolicyArchitecture | None = None,
):
    policy = (
        policy_architecture
        or metta.agent.policies.vit_sliding_trans.ViTSlidingTransConfig()
    )
    tool = experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy,
    )

    # Adjust optimizer + trainer settings for the smaller sliding transformer architecture
    optimizer = tool.trainer.optimizer
    optimizer.learning_rate = 0.001153637
    tool.trainer.batch_size = 131072
    tool.trainer.minibatch_size = 4096
    tool.training_env.forward_pass_minibatch_target_size = 1024

    # Disable periodic evaluation by default for this recipe
    tool.evaluator.epoch_interval = 0

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
