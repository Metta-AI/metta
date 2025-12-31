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


def _spec_display_name(policy_spec: PolicySpec) -> str:
    init_kwargs = policy_spec.init_kwargs or {}
    return init_kwargs.get("display_name") or policy_spec.name


class EvaluateTool(Tool):
    simulations: Sequence[SimulationConfig] | Sequence[SimulationRunConfig]
    policy_uris: str | list[str] = Field(description="Policy URIs to evaluate. The first URI is the primary policy.")

    replay_dir: str = Field(default_factory=auto_replay_dir)

    stats_server_uri: str | None = Field(default_factory=auto_stats_server_uri)
    verbose: bool = False
    push_metrics_to_wandb: bool = False
    max_workers: int | None = None

    def _compute_num_workers(self) -> int:
        if self.max_workers is not None:
            return self.max_workers

        cpu_count = multiprocessing.cpu_count()
        remainder = len(self.simulations) % cpu_count
        if remainder == 0 or len(self.simulations) < cpu_count:
            return cpu_count

        full_rounds = math.floor(len(self.simulations) / cpu_count)
        return math.ceil(len(self.simulations) / full_rounds)

    def _to_simulation_run_configs(self) -> list[SimulationRunConfig]:
        result = []
        for sim in self.simulations:
            if isinstance(sim, SimulationConfig):
                result.append(sim.to_simulation_run_config())
            else:
                result.append(sim)
        return result

    def _get_policy_version(self, uri: str) -> PolicyVersionWithName | None:
        if not uri.startswith("metta://"):
            return None
        try:
            resolver = MettaSchemeResolver(self.stats_server_uri)
            return resolver.get_policy_version(uri)
        except Exception:
            return None

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

        policy_versions = (
            [self._get_policy_version(uri) for uri in policy_uris] if stats_client else [None for _ in policy_uris]
        )
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

            num_workers = self._compute_num_workers()
            logger.info("Using %d workers for evaluation", num_workers)
            rollout_results = simulate_and_record(
                policy_specs=policy_specs,
                simulations=self._to_simulation_run_configs(),
                replay_dir=self.replay_dir,
                seed=self.system.seed,
                observatory_writer=observatory_writer,
                wandb_writer=wandb_writer,
                max_workers=num_workers,
                on_progress=logger.info if self.verbose else lambda x: None,
            )

        render_eval_summary(
            rollout_results,
            policy_names=[_spec_display_name(spec) for spec in policy_specs],
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
