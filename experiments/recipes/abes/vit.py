import experiments.recipes.arena_basic_easy_shaped
import metta.agent.policies.vit
import metta.agent.policy


def train(
    *,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: metta.agent.policy.PolicyArchitecture | None = None,
):
    return experiments.recipes.arena_basic_easy_shaped.train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=policy_architecture
        or metta.agent.policies.vit.ViTDefaultConfig(),
    )


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
