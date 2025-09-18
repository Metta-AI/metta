import contextlib
import os
import platform
from typing import Optional

import torch
from pydantic import Field

from metta.agent.policies.fast import FastConfig
from metta.agent.policy import PolicyArchitecture
from metta.common.tool import Tool
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.log_config import getRankAwareLogger, init_logging
from metta.common.wandb.wandb_context import WandbConfig, WandbContext
from metta.rl.system_config import guess_device
from metta.rl.trainer import Trainer
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import (
    DistributedHelper,
    EvaluatorConfig,
    PolicyUploaderConfig,
    TrainerCheckpointerConfig,
)
from metta.rl.training.torch_profiler_component import TorchProfilerConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig, VectorizedTrainingEnvironment
from metta.rl.training.wandb_logger import WandbLoggerComponent
from metta.tools.utils.auto_config import auto_run_name, auto_stats_server_uri, auto_wandb_config

logger = getRankAwareLogger(__name__)


class TrainTool(Tool):
    run: Optional[str] = None
    run_dir: Optional[str] = None
    device: str = guess_device()

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    training_env: TrainingEnvironmentConfig
    policy_architecture: PolicyArchitecture = Field(default_factory=FastConfig)
    initial_policy_uri: Optional[str] = None
    policy_uploader: PolicyUploaderConfig = Field(default_factory=PolicyUploaderConfig)

    stats_server_uri: Optional[str] = auto_stats_server_uri()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    torch_profiler: TorchProfilerConfig = Field(default_factory=TorchProfilerConfig)

    checkpointer: TrainerCheckpointerConfig = Field(default_factory=TrainerCheckpointerConfig)

    # Optional configurations xcxc
    map_preview_uri: str | None = None
    disable_macbook_optimize: bool = False

    def invoke(self, args: dict[str, str]) -> int | None:
        init_logging(run_dir=self.run_dir)

        # Handle run_id being passed via cmd line
        if platform.system() == "Darwin" and not self.disable_macbook_optimize:
            self._minimize_config_for_debugging()
        if "run" in args:
            assert self.run is None, "run cannot be set via args and config"
            self.run = args["run"]

        if self.run is None:
            self.run = auto_run_name(prefix="local")

        # Set run_dir based on run name if not explicitly set
        if self.run_dir is None:
            self.run_dir = f"{self.system.data_dir}/{self.run}"

        # Set up checkpoint and replay directories
        if not self.checkpointer.checkpoint_dir:
            self.checkpointer.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.run)

        os.makedirs(self.run_dir, exist_ok=True)
        init_logging(run_dir=self.run_dir)
        record_heartbeat()

        # Create distributed helper early to handle distributed setup
        distributed_helper = DistributedHelper(torch.device(self.device))
        distributed_helper.scale_batch_config(self.trainer)

        # stats_client = StatsClient.create(self.stats_server_uri)
        self.training_env.seed += distributed_helper.get_rank()
        env = VectorizedTrainingEnvironment(self.training_env)
        policy = self.policy_architecture.make_policy(env.meta_data)

        trainer = Trainer(
            self.trainer,
            env,
            policy,
            torch.device(self.device),
            run_name=self.run,
        )

        # trainer.register(Evaluator(self.evaluation))
        # trainer.register(PolicyUploader(self.policy_uploader))
        # trainer.register(GradientStatsRecorder(self.gradient_stats))
        # trainer.register(TrainingStatsReporter(StatsConfig(
        # report_to_wandb=bool(wandb_run), report_to_stats_client=bool(stats_client), interval=1)))
        # trainer.register(TrainerCheckpointer(self.checkpointer, distributed_helper))
        # trainer.register(TorchProfiler(self.torch_profiler))
        # trainer.register(SystemMonitor(self.system_monitor))

        # checkpoint_manager = CheckpointManager(
        #     self.run,
        #     self.run_dir,
        # )
        # trainer_state = checkpoint_manager.load_trainer_state()

        if distributed_helper.is_master():
            logger.info(f"Training environment: {env}")

            # Override group if provided via args (for sweep support) xcxc
            # wandb_run = WandbConfig.from_config(self.wandb)

            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                f.write(self.model_dump_json(indent=2))
                logger.info(f"Config saved to {os.path.join(self.run_dir, 'config.json')}")

        if distributed_helper.is_master() and self.wandb.enabled:
            wandb_manager = WandbContext(self.wandb, self)
        else:
            wandb_manager = contextlib.nullcontext(None)

        try:
            with wandb_manager as wandb_run:
                if wandb_run is not None:
                    trainer.register(WandbLoggerComponent(wandb_run))
                trainer.train()
        finally:
            env.close()
            distributed_helper.cleanup()

    def _minimize_config_for_debugging(self) -> None:
        self.trainer.minibatch_size = min(self.trainer.minibatch_size, 1024)
        self.trainer.batch_size = min(self.trainer.batch_size, 1024)
        self.trainer.bptt_horizon = min(self.trainer.bptt_horizon, 8)

        self.training_env.async_factor = 1
        self.training_env.forward_pass_minibatch_target_size = min(
            self.training_env.forward_pass_minibatch_target_size, 4
        )
        self.checkpointer.epoch_interval = min(self.checkpointer.epoch_interval, 10)
        self.policy_uploader.epoch_interval = min(self.policy_uploader.epoch_interval, 10)

        self.evaluator.epoch_interval = min(self.evaluator.epoch_interval, 10)
