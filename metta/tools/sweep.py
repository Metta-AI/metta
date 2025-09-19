"""SweepTool for Bayesian hyperparameter optimization using adaptive experiments."""

import logging
import os
import uuid
from enum import StrEnum
from typing import Any, Optional

from cogweb.cogweb_client import CogwebClient
from metta.adaptive import AdaptiveConfig, AdaptiveController
from metta.adaptive.dispatcher import LocalDispatcher, SkypilotDispatcher
from metta.sweep.schedulers.batched_synced import BatchedSyncedOptimizingScheduler, BatchedSyncedSchedulerConfig
from metta.adaptive.stores import WandbStore
from metta.common.tool import Tool
from metta.common.util.log_config import init_logging
from metta.common.wandb.context import WandbConfig
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


def create_on_eval_completed_hook(metric_path: str):
    """Create an on_eval_completed hook that extracts the specified metric.

    Args:
        metric_path: The path to the metric in the summary (e.g., "evaluator/eval_arena/score")

    Returns:
        A hook function that extracts the metric and updates the observation.
    """

    def on_eval_completed(run, store, all_runs):
        """Update run summary with sweep-specific observation data for the optimizer."""
        # Extract the summary
        summary = run.summary or {}

        # Look for the specific metric we're optimizing - fail hard if not found
        if metric_path not in summary:
            error_msg = (
                f"[SweepTool] CRITICAL: Metric '{metric_path}' not found in run {run.run_id} summary. "
                f"The sweep cannot optimize without this metric. Please verify your evaluation "
                f"is producing the expected metric."
            )
            logger.error(error_msg)
            raise KeyError(error_msg)

        score = summary[metric_path]

        # Use the existing cost field from RunInfo (defaults to 0 if not set)
        cost = run.cost

        # Update the run summary with sweep data for the optimizer
        store.update_run_summary(
            run.run_id,
            {
                "sweep/score": float(score),
                "sweep/cost": float(cost),
            },
        )
        logger.info(f"[SweepTool] Updated sweep observation for {run.run_id}: score={score:.6f}, cost={cost:.2f}")

    return on_eval_completed


def on_job_dispatch_hook(job, store):
    """Record the suggestion that was used for this training job.

    Only training jobs have suggestions in their metadata. Eval jobs
    reuse the same run_id and don't have new suggestions.
    """
    from metta.adaptive.models import JobTypes

    # Only process training jobs - eval jobs don't have suggestions
    if job.type == JobTypes.LAUNCH_TRAINING:
        # Get the suggestion from job metadata (set by the scheduler)
        suggestion = job.metadata.get("adaptive/suggestion", {})

        if suggestion:
            # Store the suggestion in the run summary for the optimizer to read later
            store.update_run_summary(job.run_id, {"sweep/suggestion": suggestion})
            logger.info(f"[SweepTool] Recorded suggestion for {job.run_id}: {suggestion}")
        else:
            logger.warning(f"[SweepTool] Training job {job.run_id} dispatched without suggestion metadata")


class DispatcherType(StrEnum):
    """Available dispatcher types for job execution."""

    LOCAL = "local"  # All jobs run locally
    SKYPILOT = "skypilot"  # All jobs run on Skypilot


class SweepTool(Tool):
    """Tool for Bayesian hyperparameter optimization using adaptive experiments.

    This tool is specialized for hyperparameter tuning using Bayesian optimization.
    For other experiment types (GPU sweeps, architecture comparisons), use the
    AdaptiveController directly in Python code.
    """

    # Sweep identity - optional, will be generated if not provided
    sweep_name: Optional[str] = None
    sweep_dir: Optional[str] = None

    # Core sweep configuration - Bayesian optimization config
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
    batch_size: int = 4  # Number of suggestions per batch
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"

    # Controller settings
    max_parallel_jobs: int = 1
    monitoring_interval: int = 60
    sweep_server_uri: str = "https://api.observatory.softmax-research.net"
    gpus: int = 1  # Number of GPUs per training job
    nodes: int = 1  # Number of nodes per training job

    # Override configurations
    train_overrides: dict[str, Any] = {}  # Overrides to apply to all training jobs
    eval_overrides: dict[str, Any] = {}  # Overrides to apply to all evaluation jobs

    # Infrastructure configuration
    wandb: WandbConfig = WandbConfig.Unconfigured()
    stats_server_uri: Optional[str] = auto_stats_server_uri()  # Stats server for remote evaluations

    # Dispatcher configuration
    dispatcher_type: DispatcherType = DispatcherType.SKYPILOT  # SKYPILOT or LOCAL
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
        init_logging(run_dir=self.sweep_dir)

        logger.info("[SweepTool] " + "=" * 60)
        logger.info(f"[SweepTool] Starting Bayesian optimization sweep: {self.sweep_name}")
        logger.info(f"[SweepTool] Recipe: {self.recipe_module}.{self.train_entrypoint}")
        logger.info(f"[SweepTool] Max trials: {self.max_trials}")
        logger.info(f"[SweepTool] Batch size: {self.batch_size}")
        logger.info(f"[SweepTool] Max parallel jobs: {self.max_parallel_jobs}")
        logger.info(f"[SweepTool] Monitoring interval: {self.monitoring_interval}s")
        logger.info(f"[SweepTool] Dispatcher type: {self.dispatcher_type}")
        logger.info("[SweepTool] " + "=" * 60)

        # Check for resumption using cogweb
        resume = False
        if self.sweep_server_uri:
            try:
                cogweb_client = CogwebClient.get_client(base_url=self.sweep_server_uri)
                sweep_client = cogweb_client.sweep_client()
                sweep_info = sweep_client.get_sweep(self.sweep_name)

                if not sweep_info.exists:
                    logger.info(f"[SweepTool] Registering new sweep: {self.sweep_name}")
                    sweep_client.create_sweep(self.sweep_name, self.wandb.project, self.wandb.entity, self.sweep_name)
                    resume = False
                else:
                    logger.info(f"[SweepTool] Resuming existing sweep: {self.sweep_name}")
                    resume = True
            except Exception as e:
                logger.warning(f"[SweepTool] Could not check sweep status via cogweb: {e}")
                resume = False

        # Create components
        store = WandbStore(entity=self.wandb.entity, project=self.wandb.project)

        # Create dispatcher based on type
        if self.dispatcher_type == DispatcherType.LOCAL:
            dispatcher = LocalDispatcher(capture_output=self.capture_output)

        elif self.dispatcher_type == DispatcherType.SKYPILOT:
            dispatcher = SkypilotDispatcher()

        else:
            raise ValueError(f"Unsupported dispatcher type: {self.dispatcher_type}")

        # Create scheduler configuration for Bayesian optimization
        scheduler_config = BatchedSyncedSchedulerConfig(
            max_trials=self.max_trials,
            batch_size=self.batch_size,
            recipe_module=self.recipe_module,
            train_entrypoint=self.train_entrypoint,
            eval_entrypoint=self.eval_entrypoint,
            train_overrides=self.train_overrides,
            eval_overrides=self.eval_overrides,
            stats_server_uri=self.stats_server_uri,
            gpus=self.gpus,
            nodes=self.nodes,
            experiment_id=self.sweep_name,
            protein_config=self.protein_config,
        )

        # Create scheduler with Bayesian optimization
        scheduler = BatchedSyncedOptimizingScheduler(scheduler_config)

        # Create adaptive config
        adaptive_config = AdaptiveConfig(
            max_parallel=self.max_parallel_jobs, monitoring_interval=self.monitoring_interval, resume=resume
        )

        # Save configuration
        config_path = os.path.join(self.sweep_dir, "sweep_config.json")
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
            logger.info(f"[SweepTool] Config saved to {config_path}")

        # Create the adaptive controller
        controller = AdaptiveController(
            experiment_id=self.sweep_name,
            scheduler=scheduler,
            dispatcher=dispatcher,
            store=store,
            config=adaptive_config,
        )

        try:
            logger.info("[SweepTool] Starting adaptive controller with sweep hooks...")
            logger.info(f"[SweepTool] Optimizing metric: {self.protein_config.metric}")

            # Create the on_eval_completed hook with the specific metric we're optimizing
            on_eval_completed = create_on_eval_completed_hook(self.protein_config.metric)

            # Pass hooks to run method for sweep-specific observation tracking
            controller.run(
                on_eval_completed=on_eval_completed,
                on_job_dispatch=on_job_dispatch_hook,
            )

        except KeyboardInterrupt:
            logger.info("[SweepTool] Sweep interrupted by user")
        except Exception as e:
            logger.error(f"[SweepTool] Sweep failed with error: {e}")
            raise
        finally:
            # Final summary
            final_runs = store.fetch_runs(filters={"group": self.sweep_name})

            logger.info("[SweepTool] " + "=" * 60)
            logger.info("[SweepTool] SWEEP SUMMARY")
            logger.info("[SweepTool] " + "=" * 60)
            logger.info(f"[SweepTool] Sweep name: {self.sweep_name}")
            logger.info(f"[SweepTool] Total runs: {len(final_runs)}")

            # Show detailed status table
            if final_runs:
                from metta.adaptive.utils import make_monitor_table

                table_lines = make_monitor_table(
                    runs=final_runs,
                    title="Final Run Status",
                    logger_prefix="[SweepTool]",
                    include_score=True,
                    truncate_run_id=True,
                )
                for line in table_lines:
                    logger.info(line)

            # Show best result if available
            # Filter runs that have sweep scores (i.e., completed evaluations with scores)
            completed_runs = [run for run in final_runs if run.summary and run.summary.get("sweep/score") is not None]

            if completed_runs:
                # Find the best run based on the score
                if self.protein_config.goal == "maximize":
                    best_run = max(completed_runs, key=lambda r: r.summary.get("sweep/score", float("-inf")))  # type: ignore[union-attr]
                else:
                    best_run = min(completed_runs, key=lambda r: r.summary.get("sweep/score", float("inf")))  # type: ignore[union-attr]

                logger.info("[SweepTool] ")
                logger.info("[SweepTool] Best result:")
                logger.info(f"[SweepTool]    Run: {best_run.run_id}")

                # Get the score and suggestion from the summary
                score = best_run.summary.get("sweep/score")  # type: ignore[union-attr]
                suggestion = best_run.summary.get("sweep/suggestion", {})  # type: ignore[union-attr]

                logger.info(f"[SweepTool]    Score: {score:.4f}")
                if suggestion:
                    logger.info(f"[SweepTool]    Config: {suggestion}")

            logger.info("[SweepTool] " + "=" * 60)

        return 0
