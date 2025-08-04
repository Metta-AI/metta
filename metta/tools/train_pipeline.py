"""Pipeline-based implementation of TrainTool using tAXIOM.

This demonstrates how a monolithic Tool can be decomposed into an explicit
pipeline with clear stages and I/O operations. Key improvements:

1. **Explicit data flow**: Each stage's inputs/outputs are visible
2. **Clear I/O boundaries**: External operations are marked as .io()
3. **Guards for cross-cutting concerns**:
   - @wandb_context handles context manager elegantly
   - @master_process_only prevents duplicate operations
   - @platform_specific enables conditional execution
4. **Testability**: Each stage can be tested independently
5. **Observability**: Hooks/checks can be added between any stages

The guards pattern solves the context manager problem beautifully - instead
of awkward setup/cleanup stages, we have a clean decorator that wraps the
training stage with the WandB context.
"""

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Optional

import torch

from metta.agent.agent_config import AgentConfig
from metta.agent.policy_store import PolicyStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.logging_helpers import init_file_logging, init_logging
from metta.common.wandb.wandb_context import WandbConfig, WandbRun
from metta.core.distributed import TorchDistributedConfig, setup_torch_distributed
from metta.rl.trainer import train
from metta.rl.trainer_config import TrainerConfig
from metta.sweep.axiom import Context, Pipeline
from metta.sweep.axiom.training_guards import (
    master_process_only,
    platform_specific,
    wandb_context,
    with_logging,
)
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Immutable configuration that flows through the pipeline."""

    tool: "TrainJobPipeline"
    args: dict[str, str]
    overrides: list[str]


class WandbMixin:
    """Mixin that adds WandB context support to state objects."""

    wandb_run: Optional[WandbRun] = None


@dataclass
class TrainingState(WandbMixin):
    """Mutable state that accumulates through the pipeline.

    Inherits from WandbMixin to support wandb_context guard.
    """

    config: TrainingConfig

    # State that gets set during pipeline execution
    run: Optional[str] = None
    run_dir: Optional[str] = None
    torch_dist_cfg: Optional[TorchDistributedConfig] = None
    policy_store: Optional[PolicyStore] = None
    stats_client: Optional[StatsClient] = None


class TrainJobPipeline(Tool):
    """Training tool implemented as a tAXIOM pipeline.

    This demonstrates how a Tool can be decomposed into explicit pipeline stages,
    making the training protocol transparent and modifiable.
    """

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

    consumed_args: list[str] = ["run"]

    def get_pipeline(self, initial_state: TrainingState) -> Pipeline:
        """Build the training pipeline."""
        return (
            Pipeline(initial_state)
            .stage("initialize", self._initialize)
            .io("setup_logging", self._setup_logging)
            .io("setup_distributed", self._setup_distributed)
            .through(TrainingState, hooks=[lambda s, c: record_heartbeat()])
            .stage("configure", self._configure_training)
            .io("create_policy_store", self._create_policy_store)
            .io("save_config", self._save_configuration)
            .stage("platform_adjustments", self._apply_platform_adjustments)
            .io("train", self._execute_training, timeout=3600 * 24)
            .io("cleanup", self._cleanup_distributed)
        )

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Execute the training pipeline."""
        # Create initial state with typed configuration
        config = TrainingConfig(tool=self, args=args, overrides=overrides)
        initial_state = TrainingState(config=config)

        # Create context
        ctx = Context()

        # Build and run the pipeline
        pipeline = self.get_pipeline(initial_state)
        pipeline.run(ctx)

        return 0

    # ========== Stage Implementations ==========

    def _initialize(self, state: TrainingState = None) -> TrainingState:
        """Initialize run configuration, directories, and defaults."""
        # If no state provided (first stage), get it from context or create it
        if state is None:
            # This happens when called as first pipeline stage
            # Create the config and state here
            config = TrainingConfig(tool=self, args={}, overrides=[])
            state = TrainingState(config=config)

        tool = state.config.tool
        args = state.config.args

        # Process arguments
        if "run" in args:
            assert tool.run is None, "run cannot be set via args and config"
            tool.run = args["run"]
        if tool.run is None:
            tool.run = f"local.{os.getenv('USER', 'unknown')}.{str(uuid.uuid4())}"
        state.run = tool.run

        # Setup directories
        if tool.run_dir is None:
            tool.run_dir = f"{tool.system.data_dir}/{state.run}"
        os.makedirs(tool.run_dir, exist_ok=True)
        state.run_dir = tool.run_dir

        # Configure defaults
        if not tool.policy_uri:
            tool.policy_uri = f"file://{state.run_dir}/checkpoints"
        if not tool.trainer.checkpoint.checkpoint_dir:
            tool.trainer.checkpoint.checkpoint_dir = f"{state.run_dir}/checkpoints/"
        if tool.policy_architecture is None:
            tool.policy_architecture = AgentConfig()
        if tool.wandb == WandbConfig.Unconfigured():
            tool.wandb = auto_wandb_config(state.run)

        return state

    @with_logging("Training")
    def _setup_logging(self, state: TrainingState) -> TrainingState:
        assert state.run_dir is not None
        init_file_logging(run_dir=state.run_dir)
        init_logging(run_dir=state.run_dir)
        return state

    def _setup_distributed(self, state: TrainingState) -> TrainingState:
        tool = state.config.tool
        torch_dist_cfg = setup_torch_distributed(tool.system.device)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                f"Training {state.run} on "
                + f"{os.environ.get('NODE_INDEX', '0')}: "
                + f"{os.environ.get('LOCAL_RANK', '0')} ({tool.system.device})"
            )

        state.torch_dist_cfg = torch_dist_cfg
        return state

    def _configure_training(self, state: TrainingState) -> TrainingState:
        """Configure all training-related settings."""
        tool = state.config.tool

        # Vecenv settings
        if tool.system.vectorization == "serial":
            tool.trainer.rollout_workers = 1
            tool.trainer.async_factor = 1
        else:
            ideal_workers = (os.cpu_count() // 2) // torch.cuda.device_count()
            tool.trainer.rollout_workers = max(1, ideal_workers)

        # Evaluation settings
        if tool.trainer.evaluation is not None:
            if tool.trainer.evaluation.replay_dir is None:
                logger.info(f"Setting replay_dir to s3://softmax-public/replays/{state.run}")
                tool.trainer.evaluation.replay_dir = f"s3://softmax-public/replays/{state.run}"
            if tool.stats_server_uri is not None:
                state.stats_client = StatsClient.create(tool.stats_server_uri)

        # Scale batch sizes for distributed
        if state.torch_dist_cfg and state.torch_dist_cfg.distributed and tool.trainer.scale_batches_by_world_size:
            tool.trainer.forward_pass_minibatch_target_size //= state.torch_dist_cfg.world_size
            tool.trainer.batch_size //= state.torch_dist_cfg.world_size

        return state

    def _create_policy_store(self, state: TrainingState) -> TrainingState:
        tool = state.config.tool
        state.policy_store = PolicyStore.create(
            device=tool.system.device,
            data_dir=tool.system.data_dir,
            wandb_config=tool.wandb,
            wandb_run=None,  # Will be updated in wandb context
        )
        return state

    @master_process_only()
    def _save_configuration(self, state: TrainingState) -> TrainingState:
        tool = state.config.tool
        config_path = os.path.join(state.run_dir, "config.json")
        with open(config_path, "w") as f:
            f.write(tool.model_dump_json(indent=2))
        logger.info(f"Config saved to {config_path}")
        return state

    @platform_specific("Darwin")
    def _apply_platform_adjustments(self, state: TrainingState) -> TrainingState:
        tool = state.config.tool
        logger.info(f"Applying Darwin platform adjustments")
        logger.info(f"Before: total_timesteps={tool.trainer.total_timesteps}, batch_size={tool.trainer.batch_size}")

        # Quick test settings for macOS development
        tool.trainer.checkpoint.checkpoint_interval = 100
        # Ensure wandb_checkpoint_interval is >= checkpoint_interval
        if tool.trainer.checkpoint.wandb_checkpoint_interval < 100:
            tool.trainer.checkpoint.wandb_checkpoint_interval = 100
        # Ensure evaluate_interval is >= checkpoint_interval
        if tool.trainer.evaluation and tool.trainer.evaluation.evaluate_interval < 100:
            tool.trainer.evaluation.evaluate_interval = 100
        # Limit timesteps for quick testing
        tool.trainer.total_timesteps = min(tool.trainer.total_timesteps, 1000)

        # For very quick tests, reduce batch sizes to avoid memory issues
        if tool.trainer.total_timesteps <= 1000:
            logger.info("Applying quick test batch size adjustments")
            tool.trainer.batch_size = min(tool.trainer.batch_size, 65536)
            tool.trainer.minibatch_size = min(tool.trainer.minibatch_size, 4096)
            tool.trainer.forward_pass_minibatch_target_size = min(tool.trainer.forward_pass_minibatch_target_size, 512)
            tool.trainer.async_factor = 1  # No async for quick tests
            tool.trainer.bptt_horizon = min(tool.trainer.bptt_horizon, 16)
            tool.trainer.rollout_workers = min(tool.trainer.rollout_workers, 1)

        logger.info(f"After: total_timesteps={tool.trainer.total_timesteps}, batch_size={tool.trainer.batch_size}")
        return state

    @wandb_context(master_only=True)
    def _execute_training(self, state: TrainingState) -> TrainingState:
        tool = state.config.tool
        train(
            run=state.run,
            run_dir=state.run_dir,
            system_cfg=tool.system,
            agent_cfg=tool.policy_architecture,
            device=torch.device(tool.system.device),
            trainer_cfg=tool.trainer,
            wandb_run=state.wandb_run,
            policy_store=state.policy_store,
            stats_client=state.stats_client,
            torch_dist_cfg=state.torch_dist_cfg,
        )
        return state

    def _cleanup_distributed(self, state: TrainingState) -> TrainingState:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return state

    # Below we summarize the intended workflow, as it pertains to Axiom.
