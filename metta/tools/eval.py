import contextlib
import logging
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig, WandbRunAppendContext
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
from mettagrid.util.uri_resolvers.schemes import (
    policy_spec_from_uri,
    resolve_uri,
)

logger = logging.getLogger(__name__)


def _get_wandb_config(policy_name: str, group: str | None = None) -> WandbConfig:
    wandb = auto_wandb_config(policy_name)
    if group:
        wandb.group = group
    return wandb


def _spec_display_name(policy_spec: PolicySpec) -> str:
    init_kwargs = policy_spec.init_kwargs or {}
    return init_kwargs.get("display_name") or policy_spec.name


def _get_policy_version_id(uri: str) -> str | None:
    if not uri.startswith("metta://"):
        return None
    try:
        policy_version = MettaSchemeResolver()._get_policy_version(uri)
        return str(policy_version.id)
    except Exception:
        return None


class EvaluateTool(Tool):
    simulations: Sequence[SimulationConfig] | Sequence[SimulationRunConfig]
    policy_uris: list[str] = Field(description="Policy URIs to evaluate. The first URI is the primary policy.")

    replay_dir: str = Field(default_factory=auto_replay_dir)
    enable_replays: bool = True

    group: str | None = None
    stats_server_uri: str | None = auto_stats_server_uri()
    eval_task_id: str | None = None
    verbose: bool = False
    push_metrics_to_wandb: bool = False
    device: str = "cpu"
    max_workers: int | None = None

    def _to_simulation_run_configs(self) -> list[SimulationRunConfig]:
        result = []
        for sim in self.simulations:
            if isinstance(sim, SimulationConfig):
                result.append(sim.to_simulation_run_config())
            else:
                result.append(sim)
        return result

    def invoke(self, args: dict[str, str]) -> list[SimulationRunResult]:
        if not self.policy_uris:
            raise ValueError("policy_uris is required")
        if not self.stats_server_uri:
            raise ValueError("stats_server_uri is required")

        policy_uris = list(self.policy_uris)
        primary_uri = policy_uris[0]

        policy_specs = [policy_spec_from_uri(resolve_uri(uri), device=self.device) for uri in policy_uris]

        stats_client = StatsClient.create(self.stats_server_uri)

        policy_version_ids = [_get_policy_version_id(uri) for uri in policy_uris]
        primary_policy_version_id = policy_version_ids[0]

        observatory_writer: ObservatoryWriter | None = None
        wandb_writer: WandbWriter | None = None
        wandb_context = contextlib.nullcontext(None)

        if primary_policy_version_id:
            valid_policy_version_ids = [pid for pid in policy_version_ids if pid]
            observatory_writer = ObservatoryWriter(
                stats_client=stats_client,
                policy_version_ids=valid_policy_version_ids,
                primary_policy_version_id=primary_policy_version_id,
            )

            if self.push_metrics_to_wandb:
                try:
                    policy_version = MettaSchemeResolver()._get_policy_version(primary_uri)
                    epoch = policy_version.attributes.get("epoch")
                    agent_step = policy_version.attributes.get("agent_step")
                    if epoch and agent_step:
                        wandb_config = _get_wandb_config(policy_version.name, self.group)
                        wandb_context = WandbRunAppendContext(wandb_config)
                except Exception:
                    pass

        with wandb_context as wandb_run:
            if wandb_run:
                policy_version = MettaSchemeResolver()._get_policy_version(primary_uri)
                epoch = policy_version.attributes.get("epoch")
                agent_step = policy_version.attributes.get("agent_step")
                if epoch and agent_step:
                    wandb_writer = WandbWriter(
                        wandb_run=wandb_run,
                        epoch=epoch,
                        agent_step=agent_step,
                    )

            rollout_results = simulate_and_record(
                policy_specs=policy_specs,
                simulations=self._to_simulation_run_configs(),
                replay_dir=self.replay_dir,
                seed=self.system.seed,
                observatory_writer=observatory_writer,
                wandb_writer=wandb_writer,
                max_workers=self.max_workers,
                on_progress=logger.info if self.verbose else lambda x: None,
            )

        render_eval_summary(
            rollout_results,
            policy_names=[_spec_display_name(spec) for spec in policy_specs],
            verbose=self.verbose,
        )

        return rollout_results
