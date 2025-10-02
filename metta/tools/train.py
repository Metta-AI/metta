import contextlib
import os
import platform
from datetime import timedelta
from typing import Optional

import torch
from pydantic import Field, model_validator

from metta.agent.components.transformers import get_backbone_spec
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.agent.util.torch_backends import build_sdpa_context
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool import Tool
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.log_config import getRankAwareLogger, init_logging
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.trainer import Trainer
from metta.rl.trainer_config import OptimizerConfig, TorchProfilerConfig, TrainerConfig
from metta.rl.training import (
    Checkpointer,
    CheckpointerConfig,
    ContextCheckpointer,
    ContextCheckpointerConfig,
    DistributedHelper,
    Evaluator,
    EvaluatorConfig,
    GradientReporter,
    GradientReporterConfig,
    Heartbeat,
    Monitor,
    ProgressLogger,
    Scheduler,
    SchedulerConfig,
    StatsReporter,
    StatsReporterConfig,
    TorchProfiler,
    TrainerComponent,
    TrainingEnvironmentConfig,
    Uploader,
    UploaderConfig,
    VectorizedTrainingEnvironment,
    WandbAborter,
    WandbAborterConfig,
    WandbLogger,
)
from metta.tools.utils.auto_config import (
    auto_run_name,
    auto_stats_server_uri,
    auto_wandb_config,
)

logger = getRankAwareLogger(__name__)


class TrainTool(Tool):
    run: Optional[str] = None

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    training_env: TrainingEnvironmentConfig
    policy_architecture: PolicyArchitecture = Field(default_factory=ViTDefaultConfig)
    initial_policy_uri: Optional[str] = None
    uploader: UploaderConfig = Field(default_factory=UploaderConfig)
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    gradient_reporter: GradientReporterConfig = Field(default_factory=GradientReporterConfig)

    stats_server_uri: Optional[str] = auto_stats_server_uri()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    group: Optional[str] = None
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    torch_profiler: TorchProfilerConfig = Field(default_factory=TorchProfilerConfig)

    context_checkpointer: ContextCheckpointerConfig = Field(default_factory=ContextCheckpointerConfig)
    stats_reporter: StatsReporterConfig = Field(default_factory=StatsReporterConfig)
    wandb_aborter: WandbAborterConfig = Field(default_factory=WandbAborterConfig)

    map_preview_uri: str | None = None
    disable_macbook_optimize: bool = False

    @model_validator(mode="after")
    def validate_fields(self) -> "TrainTool":
        if self.evaluator.epoch_interval != 0:
            if self.evaluator.epoch_interval < self.checkpointer.epoch_interval:
                raise ValueError(
                    "evaluator.epoch_interval must be at least as large as checkpointer.epoch_interval "
                    "to ensure policies are saved before evaluation"
                )

        if isinstance(self.policy_architecture, TransformerPolicyConfig):
            spec = get_backbone_spec(self.policy_architecture.variant)
            hint = spec.policy_defaults.get("learning_rate_hint")
            if (
                hint is not None
                and self.trainer.optimizer.learning_rate
                == OptimizerConfig.model_fields["learning_rate"].default
            ):
                self.trainer.optimizer.learning_rate = hint

        return self

    def invoke(self, args: dict[str, str]) -> int | None:
        if "run" in args:
            assert self.run is None, "run cannot be set via args if already provided in TrainTool config"
            self.run = args["run"]

        if self.run is None:
            self.run = auto_run_name(prefix="local")

        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.run)

        if self.group:
            self.wandb.group = self.group

        if platform.system() == "Darwin" and not self.disable_macbook_optimize:
            self._minimize_config_for_debugging()  # this overrides many config settings for local testings

        if self.evaluator and self.evaluator.evaluate_local:
            logger.warning("Local policy evaluation can be inefficient - consider switching to remote evaluation!")
            self.system.nccl_timeout = timedelta(hours=4)

        distributed_helper = DistributedHelper(self.system)
        distributed_helper.scale_batch_config(self.trainer, self.training_env)

        self.training_env.seed += distributed_helper.get_rank()
        env = VectorizedTrainingEnvironment(self.training_env)

        self._configure_torch_backends()

        checkpoint_manager = CheckpointManager(run=self.run or "default", system_cfg=self.system)

        if self.evaluator.evaluate_remote and not checkpoint_manager.remote_checkpoints_enabled:
            raise ValueError("without a remote prefix we cannot use remote evaluation")

        init_logging(run_dir=checkpoint_manager.run_dir)
        record_heartbeat()

        policy_checkpointer, policy = self._load_or_create_policy(checkpoint_manager, distributed_helper, env)
        trainer = self._initialize_trainer(env, policy, distributed_helper)

        self._log_run_configuration(distributed_helper, checkpoint_manager, env)

        stats_client = self._maybe_create_stats_client(distributed_helper)
        wandb_manager = self._build_wandb_manager(distributed_helper)

        try:
            with wandb_manager as wandb_run:
                self._register_components(
                    trainer=trainer,
                    distributed_helper=distributed_helper,
                    checkpoint_manager=checkpoint_manager,
                    stats_client=stats_client,
                    policy_checkpointer=policy_checkpointer,
                    wandb_run=wandb_run,
                )

                trainer.restore()
                trainer.train()
        finally:
            env.close()
            if stats_client and hasattr(stats_client, "close"):
                stats_client.close()
            distributed_helper.cleanup()
            stack = getattr(self, "_sdpa_context_stack", None)
            if stack is not None:
                stack.close()
                self._sdpa_context_stack = None

    def _load_or_create_policy(
        self,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        env: VectorizedTrainingEnvironment,
    ) -> tuple[Checkpointer, Policy]:
        policy_checkpointer = Checkpointer(
            config=self.checkpointer,
            checkpoint_manager=checkpoint_manager,
            distributed_helper=distributed_helper,
        )
        policy = policy_checkpointer.load_or_create_policy(
            env.meta_data,
            self.policy_architecture,
            policy_uri=self.initial_policy_uri,
        )
        return policy_checkpointer, policy

    def _initialize_trainer(
        self,
        env: VectorizedTrainingEnvironment,
        policy: Policy,
        distributed_helper: DistributedHelper,
    ) -> Trainer:
        trainer = Trainer(
            self.trainer,
            policy,
            env,
            distributed_helper,
        )
        return trainer

    def _configure_torch_backends(self) -> None:
        if not torch.cuda.is_available():
            return

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as exc:  # pragma: no cover - backend feature gating
            logger.debug("Skipping CUDA matmul backend configuration: %s", exc)

        context = build_sdpa_context(
            prefer_flash=True,
            prefer_mem_efficient=True,
            prefer_math=True,
            set_priority=True,
        )
        if context is not None:
            stack = getattr(self, "_sdpa_context_stack", None)
            if stack is None:
                stack = contextlib.ExitStack()
                self._sdpa_context_stack = stack
            stack.enter_context(context)
