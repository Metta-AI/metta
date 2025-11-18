"""SweepTool for Bayesian hyperparameter optimization using adaptive experiments."""

import logging
import os
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

from cogweb.cogweb_client import CogwebClient
from metta.sweep.stores import WandbStore
from metta.common.tool import Tool
from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.common.util.log_config import init_logging
from metta.common.wandb.context import WandbConfig
from metta.sweep.core import ParameterConfig
from metta.sweep.dispatchers import LocalDispatcher, SkypilotDispatcher
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.sweep import ProteinSweep
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class DispatcherType(StrEnum):
    """Available dispatcher types for job execution."""

    LOCAL = "local"  # All jobs run locally
    SKYPILOT = "skypilot"  # All jobs run on Skypilot
    REMOTE_QUEUE = "remote_queue"  # Jobs queued to PostgreSQL for remote workers


class SweepTool(Tool):
    """Tool for Bayesian hyperparameter optimization using adaptive experiments.

    This tool is specialized for hyperparameter tuning using Bayesian optimization.
    For other experiment types (GPU sweeps, architecture comparisons), use the
    SweepOrchestrator directly in Python code.
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

    # Sweep configuration
    max_trials: int = 10
    batch_size: int = 4  # Number of suggestions per batch
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"

    # Resource settings
    max_parallel: int = 6  # Maximum parallel trials
    poll_interval: float = 60.0  # Seconds between state syncs
    initial_wait: float = 5.0  # Initial wait before first sync
    sweep_server_uri: str = PROD_STATS_SERVER_URI
    gpus: int = 1  # Number of GPUs per training job
    nodes: int = 1  # Number of nodes per training job

    # local test is similar to dry runs
    local_test: bool = False

    # Force re-dispatch of evaluation jobs currently in IN_EVAL state on relaunch
    force_eval: bool = False

    # Override configurations
    train_overrides: dict[str, Any] = {}  # Overrides to apply to all training jobs
    eval_overrides: dict[str, Any] = {}  # Overrides to apply to all evaluation jobs

    # Infrastructure configuration
    wandb: WandbConfig = WandbConfig.Unconfigured()
    stats_server_uri: str = PROD_STATS_SERVER_URI  # Stats server for remote evaluations

    # Dispatcher configuration
    dispatcher_type: DispatcherType = DispatcherType.SKYPILOT  # SKYPILOT, LOCAL, or REMOTE_QUEUE
    capture_output: bool = True  # Capture and stream subprocess output (local only)
    db_url: Optional[str] = None  # PostgreSQL URL for REMOTE_QUEUE dispatcher

    # Cost tracking configuration
    cost_key: Optional[str] = None  # WandB summary key to use for cost tracking (e.g. "overview/gpu_hours")

    def invoke(self, args: dict[str, str]) -> int | None:
        """Execute the sweep."""

        if self.local_test:
            # Local testing configuration
            self.dispatcher_type = DispatcherType.LOCAL
            self.train_overrides["trainer.total_timesteps"] = 50000  # Quick 50k timesteps for testing

            # We let the batch size be set in training for the quick run
            # Use pop() to safely remove keys without raising KeyError if they don't exist
            # The keys include the full path "trainer.batch_size" not just "batch_size"
            self.protein_config.parameters.pop("trainer.batch_size", None)
            self.protein_config.parameters.pop("trainer.minibatch_size", None)
            self.protein_config.parameters.pop("trainer.total_timesteps", None)

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

        logger.info("[SweepTool] " + "=" * 60)
        logger.info(f"[SweepTool] Starting Bayesian optimization sweep: {self.sweep_name}")
        logger.info(f"[SweepTool] Recipe: {self.recipe_module}.{self.train_entrypoint}")
        logger.info(f"[SweepTool] Max trials: {self.max_trials}")
        logger.info(f"[SweepTool] Batch size: {self.batch_size}")
        logger.info(f"[SweepTool] Max parallel: {self.max_parallel}")
        logger.info(f"[SweepTool] Poll interval: {self.poll_interval}s")
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
        # Derive evaluator prefix from the configured metric if possible
        # Example: metric "evaluator/eval_sweep/score" -> prefix "evaluator/eval_sweep"
        evaluator_prefix = None
        try:
            metric_path = self.protein_config.metric
            if isinstance(metric_path, str) and "/" in metric_path:
                evaluator_prefix = metric_path.rsplit("/", 1)[0]
        except Exception:
            evaluator_prefix = None

        store = WandbStore(entity=self.wandb.entity, project=self.wandb.project, evaluator_prefix=evaluator_prefix)

        # Create dispatcher based on type
        if self.dispatcher_type == DispatcherType.LOCAL:
            dispatcher = LocalDispatcher(capture_output=self.capture_output)

        elif self.dispatcher_type == DispatcherType.SKYPILOT:
            dispatcher = SkypilotDispatcher()

        elif self.dispatcher_type == DispatcherType.REMOTE_QUEUE:
            raise NotImplementedError("[SweepTool] Using RemoteQueueDispatcher for distributed execution")

        else:
            raise ValueError(f"Unsupported dispatcher type: {self.dispatcher_type}")

        # Create ProteinSweep orchestrator
        sweep = ProteinSweep(
            experiment_id=self.sweep_name,
            dispatcher=dispatcher,
            store=store,
            protein_config=self.protein_config,
            recipe_module=self.recipe_module,
            train_entrypoint=self.train_entrypoint,
            eval_entrypoint=self.eval_entrypoint,
            max_trials=self.max_trials,
            batch_size=self.batch_size,
            gpus=self.gpus,
            nodes=self.nodes,
            train_overrides=self.train_overrides,
            eval_overrides=self.eval_overrides,
            max_parallel=self.max_parallel,
            poll_interval=self.poll_interval,
            initial_wait=self.initial_wait,
            metric_key=self.protein_config.metric,
            cost_key=self.cost_key,  # Pass the cost key if specified
            skip_evaluation=False,  # We always evaluate
            stop_on_error=False,  # Continue on errors
            resume=resume,  # Pass resume flag for state recovery
        )

        # Save configuration
        config_path = os.path.join(self.sweep_dir, "sweep_config.json")
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
            logger.info(f"[SweepTool] Config saved to {config_path}")

        try:
            logger.info("[SweepTool] Starting ProteinSweep orchestrator...")
            logger.info(f"[SweepTool] Optimizing metric: {self.protein_config.metric}")

            # Run the sweep
            sweep.run()

        except KeyboardInterrupt:
            logger.info("[SweepTool] Sweep interrupted by user")
        except Exception as e:
            logger.error(f"[SweepTool] Sweep failed with error: {e}", exc_info=True)
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
                from metta.sweep.utils import make_monitor_table

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
