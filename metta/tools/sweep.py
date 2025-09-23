"""SweepTool for hyperparameter optimization using the new orchestrator."""

import logging
import os
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Optional

from cogweb.cogweb_client import CogwebClient
from metta.common.tool import Tool
from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.common.util.log_config import init_logging
from metta.common.wandb.context import WandbConfig
from metta.sweep import JobTypes, LocalDispatcher, SweepController, SweepControllerConfig, SweepStatus
from metta.sweep.dispatcher.routing import RoutingDispatcher
from metta.sweep.dispatcher.skypilot import SkypilotDispatcher
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.protocols import Dispatcher, Optimizer, Scheduler, Store
from metta.sweep.schedulers.batched_synced import BatchedSyncedOptimizingScheduler, BatchedSyncedSchedulerConfig
from metta.sweep.stores.wandb import WandbStore
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


def orchestrate_sweep(
    config: SweepControllerConfig,
    scheduler: Scheduler,
    optimizer: Optimizer,
    dispatcher: Dispatcher,
    store: Store,
) -> None:
    """Entry point for running a sweep."""
    cogweb_client = CogwebClient.get_client(base_url=config.sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    sweep_info = sweep_client.get_sweep(config.sweep_name)
    if not sweep_info.exists:
        logger.info(f"[Orchestrator] Registering sweep {config.sweep_name}")
        sweep_client.create_sweep(config.sweep_name, config.wandb.project, config.wandb.entity, config.sweep_name)
        sweep_status = SweepStatus.CREATED
    else:
        sweep_status = SweepStatus.RESUMED

    # Create the sweep controller (stateless)
    controller = SweepController(
        sweep_id=config.sweep_name,
        scheduler=scheduler,
        optimizer=optimizer,
        dispatcher=dispatcher,
        store=store,
        protein_config=config.protein_config,
        sweep_status=sweep_status,
        max_parallel_jobs=config.max_parallel_jobs,
        monitoring_interval=config.monitoring_interval,
    )

    try:
        controller.run()
    finally:
        logger.info("[Orchestrator] Sweep Completed")


class DispatcherType(StrEnum):
    """Available dispatcher types for job execution."""

    LOCAL = "local"  # All jobs run locally
    SKYPILOT = "skypilot"  # All jobs run on Skypilot


class SweepTool(Tool):
    tool_name: ClassVar[str] = "sweep"
    """Tool for running hyperparameter sweeps."""

    # Sweep identity - optional, will be generated if not provided
    sweep_name: Optional[str] = None
    sweep_dir: Optional[str] = None

    # Core sweep configuration - always required with defaults
    protein_config: ProteinConfig = ProteinConfig(
        metric="evaluator/eval_arena/score",
        goal="maximize",
        parameters={
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-3,
                distribution="log_normal",
                mean=1e-4,  # Geometric mean
                scale="auto",
            )
        },
    )

    # Scheduler configuration
    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train_shaped"
    eval_entrypoint: str = "evaluate"

    # Orchestrator settings
    max_parallel_jobs: int = 1
    monitoring_interval: int = 5
    sweep_server_uri: str = "https://api.observatory.softmax-research.net"
    gpus: int = 1  # Number of GPUs per training job
    nodes: int = 1  # Number of nodes per training job

    # Override configurations
    train_overrides: dict[str, Any] = {}  # Overrides to apply to all training jobs

    # Infrastructure configuration
    wandb: WandbConfig = WandbConfig.Unconfigured()
    stats_server_uri: str = PROD_STATS_SERVER_URI  # Stats server for remote evaluations

    # Dispatcher configuration
    dispatcher_type: DispatcherType = DispatcherType.SKYPILOT  # Default: train on Skypilot, evaluate locally
    capture_output: bool = True  # Capture and stream subprocess output (local only)

    def invoke(self, args: dict[str, str]) -> int | None:
        """Execute the sweep."""

        # Handle sweep_name being passed via cmd line
        if "sweep_name" in args:
            assert self.sweep_name is None, "sweep_name cannot be set via args and config"
            self.sweep_name = args["sweep_name"]

        # Handle run parameter from dispatcher (ignored - only consumed to prevent unused args error)
        if "run" in args:
            # The run parameter is added by dispatchers for training jobs
            # We consume it here but don't use it for sweep orchestration
            pass

        # Generate sweep name if not provided (similar to TrainTool's run name)
        if self.sweep_name is None:
            self.sweep_name = f"sweep.{os.getenv('USER', 'unknown')}.{str(uuid.uuid4())[:8]}"

        # Handle max_trials from args
        if "max_trials" in args:
            self.max_trials = int(args["max_trials"])

        # Handle recipe_module from args
        if "recipe_module" in args:
            self.recipe_module = args["recipe_module"]

        # Handle train_entrypoint from args
        if "train_entrypoint" in args:
            self.train_entrypoint = args["train_entrypoint"]

        # Handle eval_entrypoint from args
        if "eval_entrypoint" in args:
            self.eval_entrypoint = args["eval_entrypoint"]

        # Set sweep_dir based on sweep name if not explicitly set
        if self.sweep_dir is None:
            self.sweep_dir = f"{self.system.data_dir}/sweeps/{self.sweep_name}"

        # Auto-configure wandb if not set (similar to TrainTool)
        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.sweep_name)

        # Create sweep directory
        os.makedirs(self.sweep_dir, exist_ok=True)

        # Initialize logging
        init_logging(run_dir=Path(self.sweep_dir))

        logger.info("[SweepOrchestrator] " + "=" * 60)
        logger.info(f"[SweepOrchestrator] Starting sweep: {self.sweep_name}")
        logger.info(f"[SweepOrchestrator] Recipe: {self.recipe_module}.{self.train_entrypoint}")
        logger.info(f"[SweepOrchestrator] Max trials: {self.max_trials}")
        logger.info(f"[SweepOrchestrator] Max parallel jobs: {self.max_parallel_jobs}")
        logger.info(f"[SweepOrchestrator] Monitoring interval: {self.monitoring_interval}s")
        logger.info(f"[SweepOrchestrator] Dispatcher type: {self.dispatcher_type}")
        logger.info(f"[SweepOrchestrator] Output capture: {self.capture_output}")
        logger.info("[SweepOrchestrator] " + "=" * 60)

        # Build the orchestrator config
        sweep_controller_config = SweepControllerConfig(
            sweep_name=self.sweep_name,
            sweep_server_uri=self.sweep_server_uri,
            wandb=self.wandb,
            protein_config=self.protein_config,
            max_parallel_jobs=self.max_parallel_jobs,
            monitoring_interval=self.monitoring_interval,
        )

        # Create components
        store = WandbStore(entity=self.wandb.entity, project=self.wandb.project)

        # Create dispatcher based on type
        if self.dispatcher_type == DispatcherType.LOCAL:
            dispatcher = LocalDispatcher(capture_output=self.capture_output)

        elif self.dispatcher_type == DispatcherType.SKYPILOT:
            # Train on Skypilot, evaluate locally through the CLI
            dispatcher = RoutingDispatcher(
                routes={
                    JobTypes.LAUNCH_TRAINING: SkypilotDispatcher(),
                    JobTypes.LAUNCH_EVAL: LocalDispatcher(capture_output=self.capture_output),
                }
            )
            logger.info("[SweepOrchestrator] Using hybrid mode: training on Skypilot, evaluation locally")

        else:
            raise ValueError(f"Unsupported dispatcher type: {self.dispatcher_type}")

        # Create optimizer
        optimizer = ProteinOptimizer(self.protein_config)

        # Create scheduler with configuration
        scheduler_config = BatchedSyncedSchedulerConfig(
            max_trials=self.max_trials,
            recipe_module=self.recipe_module,
            train_entrypoint=self.train_entrypoint,
            eval_entrypoint=self.eval_entrypoint,
            train_overrides=self.train_overrides,  # Pass train overrides to scheduler
            stats_server_uri=self.stats_server_uri,  # Pass stats server for remote evals
            gpus=self.gpus,  # Pass GPU configuration
            nodes=self.nodes,
            batch_size=self.max_parallel_jobs,
        )
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config, optimizer)

        # Save configuration (similar to TrainTool saving config.json)
        config_path = os.path.join(self.sweep_dir, "sweep_config.json")
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
            logger.info(f"[SweepOrchestrator] Config saved to {config_path}")

        try:
            logger.info("[SweepOrchestrator] Starting orchestrator control loop...")

            # Use the orchestrate_sweep entry point
            orchestrate_sweep(
                config=sweep_controller_config,
                scheduler=scheduler,
                optimizer=optimizer,
                dispatcher=dispatcher,
                store=store,
            )

        except KeyboardInterrupt:
            logger.info("[SweepOrchestrator] Sweep interrupted by user")
        except Exception as e:
            logger.error(f"[SweepOrchestrator] Sweep failed with error: {e}")
            raise
        finally:
            # Final summary
            final_runs = store.fetch_runs(filters={"group": self.sweep_name})

            logger.info("[SweepOrchestrator] " + "=" * 60)
            logger.info("[SweepOrchestrator] SWEEP SUMMARY")
            logger.info("[SweepOrchestrator] " + "=" * 60)
            logger.info(f"[SweepOrchestrator] Sweep name: {self.sweep_name}")
            logger.info(f"[SweepOrchestrator] Total runs: {len(final_runs)}")

            # Show detailed status table
            if final_runs:
                logger.info("[SweepOrchestrator] ")
                logger.info("[SweepOrchestrator] Final Run Status Table:")
                logger.info(f"[SweepOrchestrator] {'=' * 80}")
                logger.info(f"[SweepOrchestrator] {'Run ID':<35} {'Status':<25} {'Score':<20}")
                logger.info(f"[SweepOrchestrator] {'-' * 80}")

                for run in final_runs:
                    score_str = f"{run.observation.score:.6f}" if run.observation else "N/A"
                    logger.info(f"[SweepOrchestrator] {run.run_id:<35} {str(run.status):<25} {score_str:<20}")

                logger.info(f"[SweepOrchestrator] {'=' * 80}")

            # Count by status
            completed_count = sum(1 for run in final_runs if run.observation is not None)
            failed_count = sum(1 for run in final_runs if run.has_failed)
            in_progress_count = sum(
                1 for run in final_runs if run.has_started_training and not run.has_completed_training
            )

            logger.info("[SweepOrchestrator] ")
            logger.info("[SweepOrchestrator] Summary:")
            logger.info(f"[SweepOrchestrator] - Completed with observations: {completed_count}")
            logger.info(f"[SweepOrchestrator] - Failed: {failed_count}")
            logger.info(f"[SweepOrchestrator] - In progress: {in_progress_count}")

            # Show best result if available
            observations = [run for run in final_runs if run.observation is not None]
            if observations:
                if self.protein_config.goal == "maximize":
                    best_run = max(observations, key=lambda r: r.observation.score if r.observation else 0.0)
                else:
                    best_run = min(observations, key=lambda r: r.observation.score if r.observation else 0.0)

                logger.info("[SweepOrchestrator] Best result:")
                logger.info(f"[SweepOrchestrator]    Run: {best_run.run_id}")
                logger.info(
                    f"[SweepOrchestrator]    Score: {(best_run.observation.score if best_run.observation else 0.0):.4f}"
                )
                if best_run.observation and best_run.observation.suggestion:
                    logger.info(f"[SweepOrchestrator]    Config: {best_run.observation.suggestion}")

            logger.info("[SweepOrchestrator] " + "=" * 60)

        return 0
