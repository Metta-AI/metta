import logging
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import HttpStatsClient, StatsClient
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.handle_results import render_eval_summary, send_eval_results_to_wandb
from metta.sim.runner import SimulationRunResult, run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


class EvaluateTool(Tool):
    # required params:
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uris: str | Sequence[str] | None = None  # list of policy uris to evaluate
    replay_dir: str = Field(default_factory=auto_replay_dir)
    enable_replays: bool = True

    group: str | None = None  # Separate group parameter like in train.py

    stats_server_uri: str | None = auto_stats_server_uri()
    register_missing_policies: bool = False
    eval_task_id: str | None = None
    push_metrics_to_wandb: bool = False

    def _get_wandb_config(self, policy_uri: str) -> WandbConfig | None:
        run_name = CheckpointManager.get_policy_metadata(policy_uri).get("run_name")
        if run_name is None:
            logger.info("Could not determine run name, skipping wandb logging")
            return

        wandb = auto_wandb_config(run_name)
        if self.group:
            wandb.group = self.group

        return wandb

    def _guess_epoch_and_agent_step(self, policy_uri: str) -> tuple[int, int] | None:
        stats_client: StatsClient | None = None
        if self.stats_server_uri is None:
            logger.info("Stats client is not set, skipping wandb logging")
            return None
        stats_client = HttpStatsClient.create(self.stats_server_uri)
        try:
            (epoch, attributes) = stats_client.sql_query(
                f"""SELECT e.end_training_epoch, e.attributes
                        FROM policies p join epochs e ON p.epoch_id = e.id
                        WHERE p.url = '{policy_uri}'"""
            ).rows[0]
            agent_step = attributes.get("agent_step")
            if agent_step is None:
                logger.info("Agent step is not set, skipping wandb logging")
                return None
            return epoch, agent_step
        except IndexError:
            # No rows returned; log with fallback step/epoch
            logger.info(
                "No epoch metadata for %s in stats DB; logging eval metrics to WandB with default step/epoch=0",
                policy_uri,
            )
            return 0, 0
        except Exception as e:
            logger.error(f"Error logging evaluation results to wandb: {e}", exc_info=True)
            # Best-effort fallback logging with default indices
            return 0, 0

    def eval_policy(self, normalized_uri: str) -> list[SimulationRunResult]:
        rollout_results = run_simulations(
            policy_specs=[CheckpointManager.policy_spec_from_uri(normalized_uri, device="cpu")],
            simulations=[sim.to_simulation_run_config() for sim in self.simulations],
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=self.enable_replays,
        )
        return rollout_results

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.policy_uris is None:
            raise ValueError("policy_uris is required")

        if isinstance(self.policy_uris, str):
            self.policy_uris = [self.policy_uris]

        for policy_uri in self.policy_uris:
            normalized_uri = CheckpointManager.normalize_uri(policy_uri)
            rollout_results = self.eval_policy(normalized_uri)
            render_eval_summary(rollout_results, policy_names=[policy_uri or "target policy"])
            if self.push_metrics_to_wandb:
                guess = self._guess_epoch_and_agent_step(normalized_uri)
                if guess is None:
                    logger.info("Could not determine epoch or agent step, skipping wandb logging")
                    continue
                epoch, agent_step = guess
                wandb_config = self._get_wandb_config(normalized_uri)
                if wandb_config is not None:
                    with WandbContext(wandb_config, self) as wandb_run:
                        if wandb_run:
                            send_eval_results_to_wandb(
                                rollout_results=rollout_results,
                                epoch=epoch,
                                agent_step=agent_step,
                                wandb_run=wandb_run,
                                during_training=False,
                                should_finish_run=True,
                            )
