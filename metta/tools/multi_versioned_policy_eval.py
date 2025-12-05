import logging
import uuid
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool.tool import ToolResult, ToolWithResult
from metta.sim.handle_results import render_eval_summary
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulate_and_record import ObservatoryWriter, simulate_and_record
from metta.tools.utils.auto_config import auto_replay_dir
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

logger = logging.getLogger(__name__)


class MultiPolicyVersionEvalTool(ToolWithResult):
    simulations: Sequence[SimulationRunConfig] = Field(description="Simulations to evaluate")
    replay_dir: str = Field(default_factory=auto_replay_dir)
    verbose: bool = Field(default=True, description="Whether to log verbose output")

    stats_server_uri: str
    policy_version_ids: list[str] = Field(description="Policy version ids to log to observatory")
    primary_policy_version_id: str = Field(description="Primary policy version id to log to observatory")

    def run_job(self) -> ToolResult:
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
        policy_specs = []
        paths = [policy_version.s3_path for policy_version in policy_versions]
        if not all(paths):
            raise ValueError("All policy versions must have an s3 path")
        for p in paths:
            assert p is not None
            policy_specs.append(policy_spec_from_uri(p, remove_downloaded_copy_on_exit=True))
        rollout_results = simulate_and_record(
            policy_specs=policy_specs,
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            observatory_writer=observatory_writer,
            wandb_writer=None,
            on_progress=logger.info,
        )
        render_eval_summary(rollout_results, policy_names=[spec.name for spec in policy_specs], verbose=self.verbose)
        return ToolResult(result="success")
