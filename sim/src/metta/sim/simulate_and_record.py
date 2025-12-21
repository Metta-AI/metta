from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.wandb.context import WandbRun
from metta.sim.handle_results import send_eval_results_to_wandb, write_eval_results_to_observatory
from metta.sim.runner import SimulationRunConfig, SimulationRunResult, run_simulations
from mettagrid.policy.policy import PolicySpec


class WandbWriter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wandb_run: WandbRun
    epoch: int
    agent_step: int

    def write(self, rollout_results: list[SimulationRunResult]):
        send_eval_results_to_wandb(
            rollout_results=rollout_results,
            epoch=self.epoch,
            agent_step=self.agent_step,
            wandb_run=self.wandb_run,
            during_training=False,
        )


class ObservatoryWriter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stats_client: StatsClient
    policy_version_ids: list[str]
    primary_policy_version_id: str | None = None

    def write(self, rollout_results: list[SimulationRunResult]):
        write_eval_results_to_observatory(
            policy_version_ids=self.policy_version_ids,
            rollout_results=rollout_results,
            stats_client=self.stats_client,
            primary_policy_version_id=self.primary_policy_version_id,
        )


def simulate_and_record(
    policy_specs: Sequence[PolicySpec],
    simulations: Sequence[SimulationRunConfig],
    replay_dir: str,
    seed: int,
    max_workers: int | None = None,
    observatory_writer: ObservatoryWriter | None = None,
    wandb_writer: WandbWriter | None = None,
    on_progress: Callable[[str], None] = lambda x: None,
    device_override: str | None = None,
) -> list[SimulationRunResult]:
    rollout_results = run_simulations(
        policy_specs=policy_specs,
        simulations=simulations,
        replay_dir=replay_dir,
        seed=seed,
        max_workers=max_workers,
        on_progress=on_progress,
    )

    if observatory_writer is not None:
        observatory_writer.write(rollout_results)
    if wandb_writer is not None:
        wandb_writer.write(rollout_results)

    return rollout_results
