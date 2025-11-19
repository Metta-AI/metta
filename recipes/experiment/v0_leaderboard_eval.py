import logging
import uuid
from contextlib import ExitStack
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.s3_policy_spec_loader import policy_spec_from_s3_submission
from metta.common.tool import Tool
from metta.sim.handle_results import render_eval_summary
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulate_and_record import ObservatoryWriter, simulate_and_record
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri
from recipes.experiment import arena

logger = logging.getLogger(__name__)


class MultiPolicyVersionEvalTool(Tool):
    simulations: Sequence[SimulationRunConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    verbose: bool = Field(default=True, description="Whether to log verbose output")

    stats_server_uri: str
    policy_version_ids: list[str] = Field(description="Policy version ids to log to observatory")
    primary_policy_version_id: str = Field(description="Primary policy version id to log to observatory")

    def invoke(self, args: dict[str, str]) -> int | None:
        stats_client = StatsClient.create(self.stats_server_uri)
        observatory_writer = ObservatoryWriter(
            stats_client=stats_client,
            policy_version_ids=self.policy_version_ids,
            primary_policy_version_id=self.primary_policy_version_id,
        )
        policy_versions = [
            stats_client.get_policy_version(uuid.UUID(policy_version_id))
            for policy_version_id in self.policy_version_ids
        ]
        with ExitStack() as stack:
            policy_specs = []
            paths = [policy_version.s3_path for policy_version in policy_versions]
            if not all(paths):
                raise ValueError("All policy versions must have an s3 path")
            for p in paths:
                policy_spec = stack.enter_context(policy_spec_from_s3_submission(p))
                policy_specs.append(policy_spec)
            rollout_results = simulate_and_record(
                policy_specs=[policy_spec],
                simulations=self.simulations,
                replay_dir=self.replay_dir,
                seed=self.system.seed,
                observatory_writer=observatory_writer,
                wandb_writer=None,
                on_progress=logger.info,
            )
            render_eval_summary(
                rollout_results, policy_names=[spec.name for spec in policy_specs], verbose=self.verbose
            )
        return 0


def simulations() -> Sequence[SimulationRunConfig]:
    basic_env = arena.mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100
    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1
    return [
        SimulationRunConfig(env=basic_env, num_episodes=1),
        SimulationRunConfig(env=combat_env, num_episodes=1),
    ]


# ./tools/run.py recipes.experiment.v0_leaderboard_eval.run policy_version_id=8bf4f06c-fbd3-4678-af4f-0bf875c2edff
def run(policy_version_id: str, stats_server_uri: str | None = None) -> MultiPolicyVersionEvalTool:
    if (api_url := stats_server_uri or auto_stats_server_uri()) is None:
        raise ValueError("stats_server_uri is required")

    return MultiPolicyVersionEvalTool(
        simulations=simulations(),
        policy_version_ids=[
            policy_version_id,
            "3847e542-d57c-41a6-9e56-76c07866f2b7",  # nishad-test-baseline-scripted-3
            "d781b02a-a421-49ed-a66d-c781219af930",  # nishad-test-baseline-scripted-2
        ],
        primary_policy_version_id=policy_version_id,
        verbose=True,
        stats_server_uri=api_url,
    )
