import contextlib
import logging
import math
import multiprocessing
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import PolicyVersionWithName
from metta.common.tool import Tool
from metta.common.tool.tool import ToolResult, ToolWithResult
from metta.common.wandb.context import WandbRunAppendContext
from metta.rl.metta_scheme_resolver import MettaSchemeResolver
from metta.sim.handle_results import render_eval_summary
from metta.sim.runner import SimulationRunConfig, SimulationRunResult
from metta.sim.simulate_and_record import (
    ObservatoryWriter,
    WandbWriter,
    simulate_and_record,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri, auto_wandb_config
from mettagrid.policy.policy import PolicySpec
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

logger = logging.getLogger(__name__)


class EvaluateTool(Tool):
    simulations: Sequence[SimulationConfig] | Sequence[SimulationRunConfig]
    policy_uris: str | list[str] = Field(description="Policy URIs to evaluate. The first URI is the primary policy.")

    replay_dir: str = Field(default_factory=auto_replay_dir)

    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)
    verbose: bool = False
    push_metrics_to_wandb: bool = False
    max_workers: int | None = None

    def invoke(self, args: dict[str, str]) -> int:
        """CLI entrypoint via run_tool. Runs eval and returns success exit code."""
        self.run_eval()
        return 0

    def run_eval(self) -> list[SimulationRunResult]:
        if not self.policy_uris:
            raise ValueError("policy_uris is required")

        if isinstance(self.policy_uris, str):
            policy_uris = [self.policy_uris]
        else:
            policy_uris = list(self.policy_uris)

        policy_specs = [policy_spec_from_uri(uri) for uri in policy_uris]

        observatory_writer: ObservatoryWriter | None = None
        wandb_writer: WandbWriter | None = None
        wandb_context = contextlib.nullcontext(None)
        primary_policy_version: PolicyVersionWithName | None = None

        stats_client: StatsClient | None = None
        if self.push_metrics_to_wandb or any(uri.startswith("metta://") for uri in policy_uris):
            if not self.stats_server_uri:
                raise ValueError("stats_server_uri is required when using metta:// policies or pushing metrics")
            stats_client = StatsClient.create(self.stats_server_uri)

        if stats_client:
            resolver = MettaSchemeResolver(self.stats_server_uri)
            policy_versions: list[PolicyVersionWithName | None] = []
            for uri in policy_uris:
                if uri.startswith("metta://"):
                    try:
                        policy_versions.append(resolver.get_policy_version(uri))
                    except Exception:
                        policy_versions.append(None)
                else:
                    policy_versions.append(None)
        else:
            policy_versions = [None for _ in policy_uris]
        primary_policy_version = policy_versions[0]

        if primary_policy_version and stats_client:
            policy_version_ids = [str(pv.id) for pv in policy_versions if pv]
            if not all(policy_version_ids) or len(policy_version_ids) != len(policy_uris):
                raise ValueError("All policy URIs must specify a policy registered in the stats server")
            observatory_writer = ObservatoryWriter(
                stats_client=stats_client,
                policy_version_ids=policy_version_ids,
                primary_policy_version_id=str(primary_policy_version.id),
            )

        if self.push_metrics_to_wandb:
            if not primary_policy_version:
                raise ValueError(
                    "The first policy_uri needs to specify a policy registered in the stats server in order for "
                    "metrics to be pushed to WandB; it's needed to find the wandb run to push stats to."
                )
            wandb_config = auto_wandb_config(primary_policy_version.name)
            wandb_context = WandbRunAppendContext(wandb_config)
            epoch = primary_policy_version.attributes.get("epoch")
            agent_step = primary_policy_version.attributes.get("agent_step")
            if epoch is None or agent_step is None:
                raise ValueError(
                    f"Cannot find the agent step or epoch associated with {primary_policy_version.name}. This is "
                    "needed to push metrics to WandB."
                )

        with wandb_context as wandb_run:
            if self.push_metrics_to_wandb:
                assert wandb_run is not None and epoch is not None and agent_step is not None
                wandb_writer = WandbWriter(
                    wandb_run=wandb_run,
                    epoch=epoch,
                    agent_step=agent_step,
                )

            if self.max_workers is not None:
                num_workers = self.max_workers
            else:
                cpu_count = multiprocessing.cpu_count()
                remainder = len(self.simulations) % cpu_count
                if remainder == 0 or len(self.simulations) < cpu_count:
                    num_workers = cpu_count
                else:
                    full_rounds = math.floor(len(self.simulations) / cpu_count)
                    num_workers = math.ceil(len(self.simulations) / full_rounds)
            logger.info("Using %d workers for evaluation", num_workers)
            sim_run_configs = [
                sim.to_simulation_run_config() if isinstance(sim, SimulationConfig) else sim
                for sim in self.simulations
            ]
            rollout_results = simulate_and_record(
                policy_specs=policy_specs,
                simulations=sim_run_configs,
                replay_dir=self.replay_dir,
                seed=self.system.seed,
                observatory_writer=observatory_writer,
                wandb_writer=wandb_writer,
                max_workers=num_workers,
                on_progress=logger.info if self.verbose else lambda x: None,
            )

        render_eval_summary(
            rollout_results,
            policy_names=[(spec.init_kwargs or {}).get("display_name") or spec.name for spec in policy_specs],
            verbose=self.verbose,
        )

        return rollout_results


class EvalWithResultTool(ToolWithResult, EvaluateTool):
    def run_job(self) -> ToolResult:
        try:
            self.run_eval()
            return ToolResult(result="success")
        except Exception as e:
            return ToolResult(result="failure", error=str(e))
