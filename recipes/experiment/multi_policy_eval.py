from __future__ import annotations

from contextlib import ExitStack
from typing import Sequence

from metta.common.s3_policy_spec_loader import policy_spec_from_s3_submission
from metta.sim.runner import SimulationRunConfig
from metta.tools.multi_policy_eval import MultiPolicyEvalTool
from mettagrid.policy.policy import PolicySpec
from mettagrid.util.url_schemes import policy_spec_from_uri
from recipes.experiment import arena


def run(policy_specs: Sequence[PolicySpec] | None = None) -> MultiPolicyEvalTool:
    policy_specs = [
        PolicySpec(
            class_path="mettagrid.policy.random_agent.RandomMultiAgentPolicy",
            data_path=None,
        ),
        PolicySpec(
            class_path="noop",  # try shorthand
            data_path=None,
        ),
        *(policy_specs or []),
    ]
    basic_env = arena.mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return MultiPolicyEvalTool(
        policy_specs=policy_specs,
        simulations=[
            SimulationRunConfig(
                env=basic_env,
                num_episodes=1,
                proportions=[i + 1 for i in range(len(policy_specs))],
                episode_tags={"name": "basic", "category": "arena"},
            ),
            SimulationRunConfig(
                env=combat_env,
                num_episodes=2,
                episode_tags={"name": "combat", "category": "arena"},
            ),
        ],
    )


# ./tools/run.py recipes.experiment.multi_policy_eval.run_old_uris policy_uris="s3://softmax-public/policies/local.nishadsingh.20251114.124019/local.nishadsingh.20251114.124019:v74.mpt"
def run_old_uris(policy_uris: Sequence[str] | str | None = None) -> MultiPolicyEvalTool:
    if isinstance(policy_uris, str):
        policy_uris = [policy_uris]
    policy_specs = []
    for policy_uri in policy_uris or []:
        policy_specs.append(policy_spec_from_uri(policy_uri, device="cpu"))
    return run(policy_specs)


# ./tools/run.py recipes.experiment.multi_policy_eval.run_submission s3_paths="s3://observatory-private/cogames/submissions/hr1t9o9ool5j7bfhe5dz5dh6/629c6ef2-43ef-4164-bef3-e3c5b0bacc48.zip"
def run_submission(s3_paths: Sequence[str] | str | None = None) -> MultiPolicyEvalTool:
    if isinstance(s3_paths, str):
        s3_paths = [s3_paths]
    policy_specs = []
    # Keep all submission contexts open so their extracted code stays on sys.path
    with ExitStack() as stack:
        for s3_path in s3_paths or []:
            submission_spec = stack.enter_context(policy_spec_from_s3_submission(s3_path))
            policy_specs.append(submission_spec)

        return run(policy_specs)
