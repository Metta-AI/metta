import contextlib
import os
import platform
from datetime import timedelta
from typing import Any, Optional

import torch
from pydantic import Field, model_validator

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
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import (
    Checkpointer,
    CheckpointerConfig,
    ContextCheckpointer,
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

    context_checkpointer: dict[str, Any] = Field(default_factory=dict)
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
            # suppress NCCL watchdog timeouts while ranks wait for master to complete evals
            logger.warning("Local policy evaluation can be inefficient - consider switching to remote evaluation!")
            self.system.nccl_timeout = timedelta(hours=4)

        distributed_helper = DistributedHelper(self.system)
        distributed_helper.scale_batch_config(self.trainer, self.training_env)

        self.training_env.seed += distributed_helper.get_rank()
        env = VectorizedTrainingEnvironment(self.training_env)

        self._configure_torch_backends()

        checkpoint_manager = CheckpointManager(run=self.run or "default", system_cfg=self.system)

        # this check is not in the model validator because we setup the remote prefix in `invoke` rather than `init``
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

            # Training completed successfully
            return 0

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            return 130  # Standard exit code for Ctrl+C

        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            return 1

        finally:
            env.close()
            if stats_client and hasattr(stats_client, "close"):
                stats_client.close()
            distributed_helper.cleanup()
            sdpa_stack = getattr(self, "_sdpa_context_stack", None)
            if sdpa_stack is not None:
                sdpa_stack.close()
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
            policy_architecture=self.policy_architecture,
        )
        policy = policy_checkpointer.load_or_create_policy(
            env.game_rules,
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
            env,
            policy,
            torch.device(self.system.device),
            distributed_helper=distributed_helper,
            run_name=self.run,
        )

        if not self.gradient_reporter.epoch_interval and getattr(self.trainer, "grad_mean_variance_interval", 0):
            self.gradient_reporter.epoch_interval = self.stats_reporter.grad_mean_variance_interval

        return trainer

    def _register_components(
        self,
        *,
        trainer: Trainer,
        distributed_helper: DistributedHelper,
        checkpoint_manager: CheckpointManager,
        stats_client: Optional[StatsClient],
        policy_checkpointer: Checkpointer,
        wandb_run,
    ) -> None:
        components: list[TrainerComponent] = []

        heartbeat_cfg = getattr(self.trainer, "heartbeat", None)
        if heartbeat_cfg is not None:
            components.append(Heartbeat(epoch_interval=heartbeat_cfg.epoch_interval))

        # Ensure learning-rate schedules stay in sync across ranks
        hyper_cfg = getattr(self.trainer, "hyperparameter_scheduler", None)
        if hyper_cfg and getattr(hyper_cfg, "enabled", False):
            interval = getattr(hyper_cfg, "epoch_interval", 1) or 1
            hyper_component = Scheduler(SchedulerConfig(interval=max(1, int(interval))))
            components.append(hyper_component)

        stats_component: TrainerComponent | None = None

        if distributed_helper.is_master():
            stats_config = self.stats_reporter.model_copy(update={"report_to_wandb": bool(wandb_run)})
            reporting_enabled = (
                stats_config.report_to_wandb or stats_config.report_to_stats_client or stats_config.report_to_console
            )

            if self.gradient_reporter.epoch_interval:
                components.append(GradientReporter(self.gradient_reporter))

            stats_component = StatsReporter.from_config(
                stats_config,
                stats_client=stats_client,
                wandb_run=wandb_run,
            )

            if stats_component is not None:
                components.append(stats_component)

            components.append(policy_checkpointer)

            self.evaluator = self.evaluator.model_copy(deep=True)
            components.append(
                Evaluator(
                    config=self.evaluator,
                    device=torch.device(self.system.device),
                    system_cfg=self.system,
                    stats_client=stats_client,
                )
            )

            components.append(
                Uploader(
                    config=self.uploader,
                    checkpoint_manager=checkpoint_manager,
                    distributed_helper=distributed_helper,
                    wandb_run=wandb_run,
                )
            )

            components.append(Monitor(enabled=reporting_enabled))
            components.append(ProgressLogger())
        else:
            components.append(policy_checkpointer)

        if self.context_checkpointer:
            logger.debug(
                "Context checkpointer configuration is ignored; checkpointing is policy-driven now: %s",
                self.context_checkpointer,
            )

        trainer_checkpointer = ContextCheckpointer(
            checkpoint_manager=checkpoint_manager,
            distributed_helper=distributed_helper,
        )
        components.append(trainer_checkpointer)

        components.append(WandbAborter(wandb_run=wandb_run, config=self.wandb_aborter))

        if distributed_helper.is_master() and getattr(self.torch_profiler, "interval_epochs", 0):
            components.append(
                TorchProfiler(
                    profiler_config=self.torch_profiler,
                    wandb_run=wandb_run,
                    run_dir=checkpoint_manager.run_dir,
                    is_master=True,
                )
            )

        for component in components:
            if component is None:
                continue
            trainer.register(component)

        if wandb_run is not None and distributed_helper.is_master():
            trainer.register(WandbLogger(wandb_run))

    def _configure_torch_backends(self) -> None:
        if not torch.cuda.is_available():
            return

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.debug("Skipping CUDA matmul backend configuration: %s", exc)

        # Opportunistically enable flash attention when available
        if os.environ.get("FLASH_ATTENTION") is None:
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                pass
            else:
                os.environ["FLASH_ATTENTION"] = "1"

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

    def _log_run_configuration(
        self,
        distributed_helper: DistributedHelper,
        checkpoint_manager: CheckpointManager,
        env: VectorizedTrainingEnvironment,
    ) -> None:
        if not distributed_helper.is_master():
            return

        if not checkpoint_manager.run_dir:
            raise ValueError("cannot _log_run_configuration without a valid run_dir")

        logger.info(f"Training environment: {env}")
        config_path = os.path.join(checkpoint_manager.run_dir, "config.json")
        with open(config_path, "w") as config_file:
            config_file.write(self.model_dump_json(indent=2))
        logger.info(f"Config saved to {config_path}")

    def _maybe_create_stats_client(self, distributed_helper: DistributedHelper) -> Optional[StatsClient]:
        if not (distributed_helper.is_master() and self.stats_server_uri):
            return None
        try:
            return StatsClient.create(stats_server_uri=self.stats_server_uri)

        except Exception as exc:
            logger.warning("Failed to initialize stats client: %s", exc)
            return None

    def _build_wandb_manager(self, distributed_helper: DistributedHelper):
        if distributed_helper.is_master() and self.wandb.enabled:
            return WandbContext(self.wandb, self)
        return contextlib.nullcontext(None)

    def _minimize_config_for_debugging(self) -> None:
        self.trainer.minibatch_size = min(self.trainer.minibatch_size, 1024)
        self.trainer.batch_size = min(self.trainer.batch_size, 1024)
        self.trainer.bptt_horizon = min(self.trainer.bptt_horizon, 8)

        self.training_env.async_factor = 1
        self.training_env.forward_pass_minibatch_target_size = min(
            self.training_env.forward_pass_minibatch_target_size, 4
        )
        self.checkpointer.epoch_interval = min(self.checkpointer.epoch_interval, 10)
        self.uploader.epoch_interval = min(self.uploader.epoch_interval, 10)
        self.evaluator.epoch_interval = min(self.evaluator.epoch_interval, 10)
