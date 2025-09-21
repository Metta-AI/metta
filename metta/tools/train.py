import contextlib
import os
import platform
from pathlib import Path
from typing import Any, ClassVar, Optional

import torch
from pydantic import Field, model_validator

from metta.agent.policies.fast import FastConfig
from metta.agent.policies.transformer import (
    TransformerImprovedConfig,
    TransformerNvidiaConfig,
    TransformerPolicyConfig,
)
from metta.agent.policy import Policy, PolicyArchitecture
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool import Tool
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.log_config import getRankAwareLogger, init_logging
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import guess_device
from metta.rl.trainer import Trainer
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import (
    Checkpointer,
    CheckpointerConfig,
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
    Uploader,
    UploaderConfig,
    WandbAborter,
    WandbAborterConfig,
)
from metta.rl.training.component import TrainerComponent
from metta.rl.training.context_checkpointer import ContextCheckpointer
from metta.rl.training.torch_profiler import TorchProfiler
from metta.rl.training.training_environment import TrainingEnvironmentConfig, VectorizedTrainingEnvironment
from metta.rl.training.wandb_logger import WandbLogger
from metta.tools.utils.auto_config import (
    auto_policy_storage_decision,
    auto_run_name,
    auto_stats_server_uri,
    auto_wandb_config,
)

logger = getRankAwareLogger(__name__)
class TrainTool(Tool):
    POLICY_PRESETS: ClassVar[dict[str, type[PolicyArchitecture]]] = {
        "fast": FastConfig,
        "transformer": TransformerPolicyConfig,
        "transformer_improved": TransformerImprovedConfig,
        "transformer_nvidia": TransformerNvidiaConfig,
    }

    @model_validator(mode="before")
    @classmethod
    def _resolve_policy_preset(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        value = data.get("policy_architecture")
        if isinstance(value, str) and "." not in value:
            preset_cls = cls.POLICY_PRESETS.get(value.lower())
            if preset_cls is None:
                valid = ", ".join(sorted(cls.POLICY_PRESETS))
                raise ValueError(f"Unknown policy preset '{value}'. Valid options: {valid}")
            data["policy_architecture"] = preset_cls()
        return data
    run: Optional[str] = None
    run_dir: Optional[str] = None
    device: str = guess_device()

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    training_env: TrainingEnvironmentConfig
    policy_architecture: PolicyArchitecture = Field(default_factory=FastConfig)
    initial_policy_uri: Optional[str] = None
    uploader: UploaderConfig = Field(default_factory=UploaderConfig)
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    gradient_reporter: GradientReporterConfig = Field(default_factory=GradientReporterConfig)

    stats_server_uri: Optional[str] = auto_stats_server_uri()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    torch_profiler: TorchProfilerConfig = Field(default_factory=TorchProfilerConfig)

    context_checkpointer: ContextCheckpointerConfig = Field(default_factory=ContextCheckpointerConfig)
    stats_reporter: StatsReporterConfig = Field(default_factory=StatsReporterConfig)
    wandb_aborter: WandbAborterConfig = Field(default_factory=WandbAborterConfig)

    map_preview_uri: str | None = None
    disable_macbook_optimize: bool = False

    def invoke(self, args: dict[str, str]) -> int | None:
        init_logging(run_dir=self.run_dir)

        if platform.system() == "Darwin" and not self.disable_macbook_optimize:
            self._minimize_config_for_debugging()

        self._configure_run_metadata(args)
        self._prepare_run_directories()

        distributed_helper = DistributedHelper(torch.device(self.device))
        distributed_helper.scale_batch_config(self.trainer, self.training_env)

        self.training_env.seed += distributed_helper.get_rank()
        env = VectorizedTrainingEnvironment(self.training_env)

        checkpoint_manager = CheckpointManager(
            run=self.run or "default",
            run_dir=self.run_dir or str(Path(self.system.data_dir)),
            remote_prefix=self.trainer.checkpoint.remote_prefix,
        )
        policy_checkpointer, policy = self._load_or_create_policy(checkpoint_manager, distributed_helper, env)
        trainer = self._initialize_trainer(env, policy, distributed_helper)

        self._log_run_configuration(distributed_helper, env)

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

    def _configure_run_metadata(self, args: dict[str, str]) -> Optional[str]:
        if "run" in args:
            assert self.run is None, "run cannot be set via args and config"
            self.run = args["run"]

        if self.run is None:
            self.run = auto_run_name(prefix="local")

        group_override = args.get("group")

        if self.run_dir is None:
            self.run_dir = f"{self.system.data_dir}/{self.run}"

        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.run)

        if group_override:
            self.wandb.group = group_override

        return group_override

    def _prepare_run_directories(self) -> None:
        if not self.context_checkpointer.checkpoint_dir:
            self.context_checkpointer.checkpoint_dir = f"{self.run_dir}/checkpoints/"

        if self.trainer.checkpoint.remote_prefix is None and self.run is not None:
            storage_decision = auto_policy_storage_decision(self.run)
            if storage_decision.remote_prefix:
                self.trainer.checkpoint.remote_prefix = storage_decision.remote_prefix
                if storage_decision.reason == "env_override":
                    logger.info("Using POLICY_REMOTE_PREFIX for policy storage: %s", storage_decision.remote_prefix)
                else:
                    logger.info(
                        "Policies will sync to %s (Softmax AWS profile detected).",
                        storage_decision.remote_prefix,
                    )
            elif storage_decision.reason == "not_connected":
                logger.info(
                    "Softmax AWS SSO not detected; policies will remain local. "
                    "Run 'aws sso login --profile softmax' then 'metta status --components=aws' to enable uploads."
                )
            elif storage_decision.reason == "aws_not_enabled":
                logger.info(
                    "AWS component disabled; policies will remain local. Run 'metta configure aws' to set up S3."
                )
            elif storage_decision.reason == "no_base_prefix":
                logger.info(
                    "Remote policy prefix unset; policies will remain local. Configure POLICY_REMOTE_PREFIX or run "
                    "'metta configure aws'."
                )

        os.makedirs(self.run_dir, exist_ok=True)
        init_logging(run_dir=self.run_dir)
        record_heartbeat()

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
            env,
            policy,
            torch.device(self.device),
            distributed_helper=distributed_helper,
            run_name=self.run,
        )

        if not self.gradient_reporter.epoch_interval and getattr(self.trainer, "grad_mean_variance_interval", 0):
            self.gradient_reporter.epoch_interval = self.trainer.grad_mean_variance_interval

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

            components.append(
                Evaluator(
                    config=self.evaluator,
                    device=torch.device(self.device),
                    system_cfg=self.system,
                    trainer_cfg=self.trainer,
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

        trainer_checkpointer = ContextCheckpointer(
            config=self.context_checkpointer,
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
                    run_dir=self.run_dir,
                    is_master=True,
                )
            )

        for component in components:
            if component is None:
                continue
            trainer.register(component)

        if wandb_run is not None and distributed_helper.is_master():
            trainer.register(WandbLogger(wandb_run))

    def _log_run_configuration(
        self,
        distributed_helper: DistributedHelper,
        env: VectorizedTrainingEnvironment,
    ) -> None:
        if not distributed_helper.is_master():
            return
        logger.info(f"Training environment: {env}")
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as config_file:
            config_file.write(self.model_dump_json(indent=2))
        logger.info(f"Config saved to {config_path}")

    def _maybe_create_stats_client(self, distributed_helper: DistributedHelper) -> Optional[StatsClient]:
        if not (distributed_helper.is_master() and self.stats_server_uri):
            return None
        try:
            return StatsClient.create(self.stats_server_uri)
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
        self.context_checkpointer.epoch_interval = min(self.context_checkpointer.epoch_interval, 10)
        self.checkpointer.epoch_interval = min(self.checkpointer.epoch_interval, 10)
        self.uploader.epoch_interval = min(self.uploader.epoch_interval, 10)

        self.evaluator.epoch_interval = min(self.evaluator.epoch_interval, 10)
