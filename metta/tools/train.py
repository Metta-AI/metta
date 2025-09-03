import os
import platform
import uuid
from typing import Any, Optional

import torch

import gitta as git
from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import PolicyAgent
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.util.git_repo import REPO_SLUG
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.log_config import getRankAwareLogger, init_logging
from metta.common.wandb.wandb_context import WandbConfig, WandbContext, WandbRun
from metta.core.distributed import TorchDistributedConfig, cleanup_distributed, setup_torch_distributed
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.eval.eval_service import evaluate_policy
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.evaluate import evaluate_policy_remote_with_checkpoint_manager, upload_replay_html
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer import TrainingCallbacks, train
from metta.rl.trainer_config import TrainerConfig
from metta.rl.wandb import abort_requested, log_model_parameters, log_torch_trace_link, setup_wandb_metrics
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_run_name, auto_stats_server_uri, auto_wandb_config

logger = getRankAwareLogger(__name__)


class TrainTool(Tool):
    trainer: TrainerConfig = TrainerConfig()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    policy_architecture: Optional[AgentConfig] = None
    run: Optional[str] = None
    run_dir: Optional[str] = None
    stats_server_uri: Optional[str] = auto_stats_server_uri()

    # Policy configuration
    policy_uri: Optional[str] = None

    # Optional configurations
    map_preview_uri: str | None = None
    disable_macbook_optimize: bool = False

    consumed_args: list[str] = ["run", "group"]

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        # Handle run_id being passed via cmd line
        if "run" in args:
            assert self.run is None, "run cannot be set via args and config"
            self.run = args["run"]

        if self.run is None:
            self.run = auto_run_name(prefix="local")
        group_override = args.get("group")

        # Set run_dir based on run name if not explicitly set
        if self.run_dir is None:
            self.run_dir = f"{self.system.data_dir}/{self.run}"

        # Set policy_uri if not set
        if not self.policy_uri:
            self.policy_uri = CheckpointManager.normalize_uri(f"{self.run_dir}/checkpoints")

        # Set up checkpoint and replay directories
        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        # Initialize policy_architecture if not provided
        if self.policy_architecture is None:
            self.policy_architecture = AgentConfig()

        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.run)

        # Override group if provided via args (for sweep support)
        if group_override:
            self.wandb.group = group_override

        os.makedirs(self.run_dir, exist_ok=True)

        record_heartbeat()

        init_logging(run_dir=self.run_dir)

        torch_dist_cfg = setup_torch_distributed(self.system.device)

        if not self.trainer.checkpoint.checkpoint_dir:
            self.trainer.checkpoint.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        logger.info_master(
            f"Training {self.run} on "
            + f"{os.environ.get('NODE_INDEX', '0')}: "
            + f"{os.environ.get('LOCAL_RANK', '0')} ({self.system.device})",
        )

        logger.info_master(
            f"Training {self.run} on {self.system.device}",
        )
        if torch_dist_cfg.is_master:
            with WandbContext(self.wandb, self) as wandb_run:
                handle_train(self, torch_dist_cfg, wandb_run)
        else:
            handle_train(self, torch_dist_cfg, None)

        cleanup_distributed()

        return 0


def handle_train(cfg: TrainTool, torch_dist_cfg: TorchDistributedConfig, wandb_run: WandbRun | None) -> None:
    assert cfg.run_dir is not None
    assert cfg.run is not None
    run_dir = cfg.run_dir

    _configure_vecenv_settings(cfg)

    stats_client: StatsClient | None = None
    if cfg.stats_server_uri is not None:
        stats_client = StatsClient.create(cfg.stats_server_uri)

    _configure_evaluation_settings(cfg, stats_client)

    # Handle distributed training batch scaling
    if torch_dist_cfg.distributed:
        if cfg.trainer.scale_batches_by_world_size:
            cfg.trainer.forward_pass_minibatch_target_size = (
                cfg.trainer.forward_pass_minibatch_target_size // torch_dist_cfg.world_size
            )
            cfg.trainer.batch_size = cfg.trainer.batch_size // torch_dist_cfg.world_size

    checkpoint_manager = CheckpointManager(run=cfg.run, run_dir=cfg.run_dir)

    if platform.system() == "Darwin" and not cfg.disable_macbook_optimize:
        cfg = _minimize_config_for_debugging(cfg)

    # Save configuration
    if torch_dist_cfg.is_master:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            f.write(cfg.model_dump_json(indent=2))
            logger.info_master(f"Config saved to {os.path.join(run_dir, 'config.json')}")

    trainer_cfg = cfg.trainer
    agent_cfg = cfg.policy_architecture

    assert agent_cfg
    assert cfg.run

    device = torch.device(cfg.system.device)

    class MyCallbacks(TrainingCallbacks):
        def __init__(self):
            self.stats_run_id: uuid.UUID | None = None
            self.stats_epoch_id: uuid.UUID | None = None

            if stats_client is not None:
                # Extract wandb attributes with defaults
                name = url = "unknown"
                description: str | None = None
                tags: list[str] | None = None
                if wandb_run:
                    name = wandb_run.name or name
                    url = wandb_run.url
                    if wandb_run.tags:
                        tags = list(wandb_run.tags)
                    description = wandb_run.notes

                try:
                    self.stats_run_id = stats_client.create_training_run(
                        name=name, url=url, description=description, tags=tags
                    ).id
                except Exception as e:
                    logger.warning(f"Failed to create training run: {e}", exc_info=True)

        def on_start(self, policy: PolicyAgent):
            if torch_dist_cfg.is_master and wandb_run:
                setup_wandb_metrics(wandb_run)
                log_model_parameters(policy, wandb_run)

        def on_end(self, status: str):
            if stats_client and self.stats_run_id:
                try:
                    stats_client.update_training_run_status(self.stats_run_id, status)
                    logger.info(f"Training run status updated to {status}")
                except Exception as e:
                    logger.warning(f"Failed to update training run status to {status}: {e}", exc_info=True)

        def evaluate_policy(
            self,
            stats_epoch_start: int,
            stats_epoch_end: int,
            env_cfg: MettaGridConfig,
            agent_step: int,
            remote_uri: str | None,
        ) -> EvalResults | None:
            if cfg.trainer.evaluation:
                # Evaluation with CheckpointManager - use current policy directly
                if stats_client and self.stats_run_id:
                    self.stats_epoch_id = stats_client.create_epoch(
                        run_id=self.stats_run_id,
                        start_training_epoch=stats_epoch_start,
                        end_training_epoch=stats_epoch_end,
                    ).id

                sims = [
                    SimulationConfig(
                        name=f"train_task_{i}",
                        env=env_cfg,
                    )
                    for i in range(cfg.trainer.evaluation.num_training_tasks)
                ]
                sims.extend(cfg.trainer.evaluation.simulations)

                evaluate_local = cfg.trainer.evaluation.evaluate_local
                if remote_uri:
                    policy_uri = remote_uri
                else:
                    checkpoint_uris = checkpoint_manager.select_checkpoints("latest", count=1)
                    policy_uri = checkpoint_uris[0] if checkpoint_uris else None
                if cfg.trainer.evaluation.evaluate_remote:
                    try:
                        # Get the most recent checkpoint URI for remote evaluation
                        # Prefer wandb artifact if available, otherwise use local file
                        if policy_uri:
                            logger.info(f"Evaluating policy remotely from {policy_uri}")
                            evaluate_policy_remote_with_checkpoint_manager(
                                policy_uri=policy_uri,
                                simulations=sims,
                                stats_epoch_id=self.stats_epoch_id,
                                stats_client=stats_client,
                                wandb_run=wandb_run,
                                trainer_cfg=trainer_cfg,
                            )
                        else:
                            logger.warning("No checkpoint available for remote evaluation")
                    except Exception as e:
                        logger.error(f"Failed to evaluate policy remotely: {e}", exc_info=True)
                        logger.error("Falling back to local evaluation")
                        evaluate_local = True
                if evaluate_local:
                    if policy_uri:
                        evaluation_results = evaluate_policy(
                            checkpoint_uri=policy_uri,
                            simulations=sims,
                            device=device,
                            vectorization=cfg.system.vectorization,
                            replay_dir=trainer_cfg.evaluation.replay_dir if trainer_cfg.evaluation else None,
                            stats_epoch_id=self.stats_epoch_id,
                            stats_client=stats_client,
                        )
                        logger.info("Simulation complete")
                        if wandb_run is not None and evaluation_results.replay_urls:
                            upload_replay_html(
                                replay_urls=evaluation_results.replay_urls,
                                agent_step=agent_step,
                                epoch=stats_epoch_end,
                                wandb_run=wandb_run,
                                metric_prefix="training_eval",
                                step_metric_key="metric/epoch",
                                epoch_metric_key="metric/epoch",
                            )
                        return evaluation_results
                    else:
                        logger.warning("No checkpoint available for local evaluation")
                        evaluation_results = EvalResults(scores=EvalRewardSummary(), replay_urls={})
                        return evaluation_results

        def log_stats(
            self,
            agent_step: int,
            stats: dict[str, Any],
        ):
            if wandb_run:
                wandb_run.log(stats, step=agent_step)

        def save_checkpoint(
            self,
            epoch: int,
            policy: PolicyAgent,
            agent_step: int,
            timer: Stopwatch,
            eval_scores: EvalRewardSummary,
            optimizer: torch.optim.Optimizer,
            upload_remotely: bool,
        ) -> str | None:
            # Extract the actual agent from distributed wrapper if needed
            agent_to_save = policy.module if torch.distributed.is_initialized() else policy

            # Build metadata from evaluation scores
            metadata = {
                "agent_step": agent_step,
                "total_time": timer.get_elapsed(),
                "total_train_time": timer.get_all_elapsed().get("_rollout", 0)
                + timer.get_all_elapsed().get("_train", 0),
            }

            # Add evaluation scores if available
            if eval_scores.category_scores or eval_scores.simulation_scores:
                metadata.update(
                    {
                        "score": eval_scores.avg_simulation_score,
                        "avg_reward": eval_scores.avg_category_score,
                        "category_scores": eval_scores.category_scores,
                        "simulation_scores": {
                            f"{cat}/{sim}": score for (cat, sim), score in eval_scores.simulation_scores.items()
                        },
                    }
                )

            # Save agent and trainer state
            # Only upload to wandb if we're at the right interval

            metadata["upload_to_wandb"] = upload_remotely

            wandb_uri = checkpoint_manager.save_agent(
                agent_to_save, epoch, metadata, wandb_run=wandb_run if upload_remotely else None
            )
            checkpoint_manager.save_trainer_state(optimizer, epoch, agent_step, timer.save_state())

            return wandb_uri

        def check_for_abort(self, agent_step: int) -> bool:
            if wandb_run and abort_requested(wandb_run, min_interval_sec=60):
                logger.info("Abort tag detected. Stopping the run.")
                trainer_cfg.total_timesteps = int(agent_step)
                wandb_run.config.update({"trainer.total_timesteps": trainer_cfg.total_timesteps}, allow_val_change=True)
                return True
            return False

    torch_profiler = TorchProfiler(
        master=torch_dist_cfg.is_master,
        profiler_config=trainer_cfg.profiler,
        log_trace_link=lambda url, epoch: log_torch_trace_link(url, epoch, wandb_run) if wandb_run else None,
        run_dir=run_dir,
    )

    # Use the functional train interface directly
    train(
        run=cfg.run,
        run_dir=run_dir,
        system_cfg=cfg.system,
        agent_cfg=agent_cfg,
        device=device,
        trainer_cfg=cfg.trainer,
        checkpoint_manager=checkpoint_manager,
        torch_dist_cfg=torch_dist_cfg,
        torch_profiler=torch_profiler,
        callbacks=MyCallbacks(),
    )


def _configure_vecenv_settings(cfg: TrainTool) -> None:
    """Calculate default number of workers based on hardware."""
    if cfg.system.vectorization == "serial":
        cfg.trainer.rollout_workers = 1
        cfg.trainer.async_factor = 1
        return

    num_gpus = torch.cuda.device_count() or 1  # fallback to 1 to avoid division by zero
    cpu_count = os.cpu_count() or 1  # fallback to 1 to avoid division by None
    ideal_workers = (cpu_count // 2) // num_gpus
    cfg.trainer.rollout_workers = max(1, ideal_workers)


def _configure_evaluation_settings(cfg: TrainTool, stats_client: StatsClient | None) -> None:
    if cfg.trainer.evaluation is None:
        return

    if cfg.trainer.evaluation.replay_dir is None:
        cfg.trainer.evaluation.replay_dir = auto_replay_dir()
        logger.info_master(f"Setting replay_dir to {cfg.trainer.evaluation.replay_dir}")

    # Determine git hash for remote simulations
    if cfg.trainer.evaluation.evaluate_remote:
        if not stats_client:
            cfg.trainer.evaluation.evaluate_remote = False
            logger.info_master("Not connected to stats server, disabling remote evaluations")
        elif not cfg.trainer.evaluation.evaluate_interval:
            cfg.trainer.evaluation.evaluate_remote = False
            logger.info_master("Evaluate interval set to 0, disabling remote evaluations")
        elif not cfg.trainer.evaluation.git_hash:
            cfg.trainer.evaluation.git_hash = git.get_git_hash_for_remote_task(
                target_repo=REPO_SLUG,
                skip_git_check=cfg.trainer.evaluation.skip_git_check,
                skip_cmd="trainer.evaluation.skip_git_check=true",
            )
            if cfg.trainer.evaluation.git_hash:
                logger.info_master(f"Git hash for remote evaluations: {cfg.trainer.evaluation.git_hash}")
            else:
                logger.info_master("No git hash available for remote evaluations")


def _minimize_config_for_debugging(cfg: TrainTool) -> TrainTool:
    cfg.trainer.minibatch_size = min(cfg.trainer.minibatch_size, 1024)
    cfg.trainer.batch_size = min(cfg.trainer.batch_size, 1024)
    cfg.trainer.async_factor = 1
    cfg.trainer.forward_pass_minibatch_target_size = min(cfg.trainer.forward_pass_minibatch_target_size, 4)
    cfg.trainer.checkpoint.checkpoint_interval = min(cfg.trainer.checkpoint.checkpoint_interval, 10)
    cfg.trainer.checkpoint.wandb_checkpoint_interval = min(cfg.trainer.checkpoint.wandb_checkpoint_interval, 10)
    cfg.trainer.bptt_horizon = min(cfg.trainer.bptt_horizon, 8)
    if cfg.trainer.evaluation:
        cfg.trainer.evaluation.evaluate_interval = min(cfg.trainer.evaluation.evaluate_interval, 10)
    return cfg
