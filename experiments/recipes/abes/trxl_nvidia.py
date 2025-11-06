import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.trxl_nvidia
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
            transformer=metta.agent.policies.trxl_nvidia.TRXLNvidiaConfig()
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
