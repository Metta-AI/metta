import logging
import uuid
from typing import Sequence

from pydantic import BaseModel, Field

from metta.agent.policy import PolicyArchitecture
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.rl.checkpoint_manager import CheckpointManager
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

logger = logging.getLogger(__name__)


class MyPolicyMetadata(BaseModel):
    policy_name: str
    policy_version_id: str
    epoch: int
    agent_step: int


def _get_wandb_config(policy_name: str, group: str | None = None) -> WandbConfig:
    wandb = auto_wandb_config(policy_name)
    if group:
        wandb.group = group

    return wandb


def _spec_display_name(policy_spec: PolicySpec) -> str:
    init_kwargs = policy_spec.init_kwargs or {}
    return init_kwargs.get("display_name") or policy_spec.name


class EvaluateTool(Tool):
    # required params:
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uris: str | Sequence[str] | None = None  # list of policy uris to evaluate

    replay_dir: str = Field(default_factory=auto_replay_dir)
    enable_replays: bool = True

    group: str | None = None  # Separate group parameter like in train.py

    stats_server_uri: str | None = auto_stats_server_uri()
    eval_task_id: str | None = None
    verbose: bool = False
    push_metrics_to_wandb: bool = False
    policy_architecture: PolicyArchitecture | None = None

    def _build_policy_spec(self, normalized_uri: str) -> PolicySpec:
        spec = CheckpointManager.policy_spec_from_uri(normalized_uri, device="cpu")
        return spec

    def _get_policy_metadata(self, policy_uri: str, stats_client: StatsClient) -> MyPolicyMetadata | None:
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        result = stats_client.sql_query(
            f"""SELECT pv.id, pv.attributes->>'agent_step'
            FROM policy_versions pv
            JOIN policies p ON pv.policy_id = p.id
            WHERE p.name = '{metadata["run_name"]}' AND pv.attributes->>'epoch' = '{metadata["epoch"]}'"""
        )
        if result.rows is None or len(result.rows) == 0:
            return None
        return MyPolicyMetadata(
            policy_name=metadata["run_name"],
            policy_version_id=result.rows[0][0],
            epoch=metadata["epoch"],
            agent_step=int(result.rows[0][1]),
        )

    def handle_single_policy_uri(self, policy_uri: str) -> tuple[int, str, list[SimulationRunResult]]:
        normalized_uri = CheckpointManager.normalize_uri(policy_uri)
        policy_spec = self._build_policy_spec(normalized_uri)

        observatory_writer: ObservatoryWriter | None = None
        wandb_writer: WandbWriter | None = None

        if self.stats_server_uri is not None:
            stats_client = StatsClient.create(self.stats_server_uri)
            policy_metadata = self._get_policy_metadata(normalized_uri, stats_client)

            if policy_metadata is None:
                logger.info(
                    "Policy not found in Observatory database. Evaluation will proceed without Observatory integration."
                )
            else:
                observatory_writer = ObservatoryWriter(
                    stats_client=stats_client,
                    policy_version_ids=[policy_metadata.policy_version_id],
                    primary_policy_version_id=policy_metadata.policy_version_id,
                )

                if self.push_metrics_to_wandb:
                    wandb_config = _get_wandb_config(normalized_uri, self.group)
                    with WandbContext(wandb_config, self) as wandb_run:
                        if wandb_run:
                            wandb_writer = WandbWriter(
                                wandb_run=wandb_run,
                                epoch=policy_metadata.epoch,
                                agent_step=policy_metadata.agent_step,
                            )

        rollout_results = simulate_and_record(
            policy_specs=[policy_spec],
            simulations=[sim.to_simulation_run_config() for sim in self.simulations],
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            observatory_writer=observatory_writer,
            wandb_writer=wandb_writer,
        )
        render_eval_summary(rollout_results, policy_names=[_spec_display_name(policy_spec)], verbose=self.verbose)

        return 0, "Done", rollout_results

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.policy_uris is None:
            raise ValueError("policy_uris is required")

        if isinstance(self.policy_uris, str):
            self.policy_uris = [self.policy_uris]

        for policy_uri in self.policy_uris:
            self.handle_single_policy_uri(policy_uri)


class EvaluatePolicyVersionTool(Tool):
    simulations: Sequence[SimulationRunConfig]  # list of simulations to run
    policy_version_id: str  # policy version id to evaluate
    replay_dir: str = Field(default_factory=auto_replay_dir)
    stats_server_uri: str | None = auto_stats_server_uri()

    group: str | None = None  # Separate group parameter like in train.py
    write_to_wandb: bool = True
    device: str = "cpu"

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.stats_server_uri is None:
            raise ValueError("stats_server_uri is required")

        stats_client = StatsClient.create(self.stats_server_uri)
        policy_version = stats_client.get_policy_version(uuid.UUID(self.policy_version_id))
        policy_spec = PolicySpec.model_validate(policy_version.policy_spec)
        if policy_spec.init_kwargs.get("device") is not None:
            policy_spec.init_kwargs["device"] = self.device

        observatory_writer = ObservatoryWriter(
            stats_client=stats_client,
            policy_version_ids=[self.policy_version_id],
            primary_policy_version_id=self.policy_version_id,
        )

        wandb_writer: WandbWriter | None = None
        if self.write_to_wandb:
            epoch = policy_version.attributes.get("epoch")
            agent_step = policy_version.attributes.get("agent_step")

            if epoch and agent_step:
                wandb_config = _get_wandb_config(policy_version.name, self.group)
                with WandbContext(wandb_config, self) as wandb_run:
                    if wandb_run:
                        wandb_writer = WandbWriter(
                            wandb_run=wandb_run,
                            epoch=epoch,
                            agent_step=agent_step,
                        )

        rollout_results = simulate_and_record(
            policy_specs=[policy_spec],
            simulations=self.simulations,
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            observatory_writer=observatory_writer,
            wandb_writer=wandb_writer,
        )
        render_eval_summary(rollout_results, policy_names=[_spec_display_name(policy_spec)])
