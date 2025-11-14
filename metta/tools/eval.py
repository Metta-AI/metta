import logging
from typing import Sequence

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.app_backend.clients.stats_client import HttpStatsClient, StatsClient
from metta.common.tool import Tool
from metta.common.util.uri import ParsedURI
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.eval.eval_request_config import EvalResults
from metta.rl import stats as rl_stats
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.evaluate import upload_replay_html
from metta.sim.handle_results import render_eval_summary, to_eval_results
from metta.sim.runner import MultiAgentPolicyInitializer, SimulationRunResult, run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_stats_server_uri, auto_wandb_config
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

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

    def _log_to_wandb(self, policy_uri: str, eval_results: EvalResults, stats_client: StatsClient | None):
        if stats_client is None:
            logger.info("Stats client is not set, skipping wandb logging")
            return

        if not self.push_metrics_to_wandb:
            logger.info("Push metrics to wandb is not set, skipping wandb logging")
            return

        run_name = CheckpointManager.get_policy_metadata(policy_uri).get("run_name")
        if run_name is None:
            logger.info("Could not determine run name, skipping wandb logging")
            return

        # Resume the existing training run without overriding its group
        wandb = auto_wandb_config(run_name)
        if self.group:
            wandb.group = self.group

        if not wandb.enabled:
            logger.info("WandB is not enabled, skipping wandb logging")
            return

        wandb_context = WandbContext(wandb, self)
        with wandb_context as wandb_run:
            if not wandb_run:
                logger.info("Failed to initialize wandb run, skipping wandb logging")
                return

            logger.info(f"Initialized wandb run: {wandb_run.id}")

            try:
                (epoch, attributes) = stats_client.sql_query(
                    f"""SELECT e.end_training_epoch, e.attributes
                          FROM policies p join epochs e ON p.epoch_id = e.id
                          WHERE p.url = '{policy_uri}'"""
                ).rows[0]
                agent_step = attributes.get("agent_step")
                if agent_step is None:
                    logger.info("Agent step is not set, skipping wandb logging")
                    return

                rl_stats.process_policy_evaluator_stats(policy_uri, eval_results, wandb_run, epoch, agent_step, False)
            except IndexError:
                # No rows returned; log with fallback step/epoch
                logger.info(
                    "No epoch metadata for %s in stats DB; logging eval metrics to WandB with default step/epoch=0",
                    policy_uri,
                )
                try:
                    rl_stats.process_policy_evaluator_stats(policy_uri, eval_results, wandb_run, 0, 0, False)
                except Exception as e:
                    logger.error("Fallback WandB logging failed: %s", e, exc_info=True)
            except Exception as e:
                logger.error(f"Error logging evaluation results to wandb: {e}", exc_info=True)
                # Best-effort fallback logging with default indices
                try:
                    rl_stats.process_policy_evaluator_stats(policy_uri, eval_results, wandb_run, 0, 0, False)
                except Exception as e2:
                    logger.error("Fallback WandB logging failed: %s", e2, exc_info=True)

    def _guess_epoch_and_agent_step(self, policy_uri: str, stats_client: StatsClient | None) -> tuple[int, int]:
        if stats_client is None:
            logger.info("Stats client is not set, skipping wandb logging")
            return None

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

    def eval_policy(self, normalized_uri: str, stats_client: StatsClient | None) -> list[SimulationRunResult]:
        device = torch.device("cpu")

        def _materialize_policy(policy_uri: str) -> MultiAgentPolicyInitializer:
            def _m(policy_env_info: PolicyEnvInterface) -> Policy:
                artifact = CheckpointManager.load_artifact_from_uri(policy_uri)
                policy = artifact.instantiate(policy_env_info, device=device)
                policy = policy.to(device)
                policy.eval()
                return policy

            return _m

        policy_initializers = [_materialize_policy((normalized_uri))]
        rollout_results = run_simulations(
            policy_initializers=policy_initializers,
            simulations=[sim.to_simulation_run_config() for sim in self.simulations],
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )
        return rollout_results

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.policy_uris is None:
            raise ValueError("policy_uris is required")

        if isinstance(self.policy_uris, str):
            self.policy_uris = [self.policy_uris]

        for uri in self.policy_uris:
            parsed_uri = ParsedURI.parse(uri)
            if parsed_uri.scheme == "wandb":
                raise ValueError(
                    "Policy artifacts must be stored on local disk or S3. "
                    "Download the checkpoint and re-run with a file:// or s3:// URI."
                )

        stats_client: StatsClient | None = None
        if self.stats_server_uri is not None:
            stats_client = HttpStatsClient.create(self.stats_server_uri)

        for policy_uri in self.policy_uris:
            normalized_uri = CheckpointManager.normalize_uri(policy_uri)
            rollout_results = self.eval_policy(normalized_uri, stats_client)
            render_eval_summary(rollout_results, policy_names=[policy_uri or "target policy"])

            epoch, agent_step = self._guess_epoch_and_agent_step(normalized_uri, stats_client)
            if self.push_metrics_to_wandb:
                wandb_config = self._get_wandb_config(normalized_uri)
                if wandb_config is not None:
                    with WandbContext(wandb_config, self) as wandb_run:
                        if wandb_run:
                            eval_results = to_eval_results(rollout_results, num_policies=1, target_policy_idx=0)
                            rl_stats.process_policy_evaluator_stats(
                                policy_uri=policy_uri,
                                eval_results=eval_results,
                                wandb_run=wandb_run,
                                epoch=epoch,
                                agent_step=agent_step,
                                should_finish_run=False,
                            )
                            upload_replay_html(
                                replay_urls=eval_results.replay_urls,
                                agent_step=agent_step,
                                epoch=epoch,
                                wandb_run=wandb_run,
                                step_metric_key="metric/epoch",
                                epoch_metric_key="metric/epoch",
                            )
