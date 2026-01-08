import contextlib
import logging
import multiprocessing
import os
import platform
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import torch
from pydantic import Field

from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.agent.util.torch_backends import build_sdpa_context
from metta.app_backend.clients.stats_client import StatsClient
from metta.cogworks.curriculum import Curriculum
from metta.common.tool import Tool
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.log_config import getRankAwareLogger, init_logging
from metta.common.wandb.context import WandbConfig, WandbContext, WandbRun
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
    StatsReporter,
    StatsReporterConfig,
    TorchProfiler,
    TrainerComponent,
    TrainingEnvironmentConfig,
    UpdateEpochAutoTuner,
    VectorizedTrainingEnvironment,
    WandbAborter,
    WandbAborterConfig,
    WandbLogger,
)
from metta.rl.training.scheduler import LossScheduler, SchedulerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import (
    PolicyStorageDecision,
    auto_policy_storage_decision,
    auto_run_name,
    auto_stats_server_uri,
    auto_wandb_config,
)
from mettagrid.policy.loader import resolve_policy_class_path
from mettagrid.policy.policy import PolicySpec
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri, resolve_uri

logger = getRankAwareLogger(__name__)


class TrainTool(Tool):
    run: Optional[str] = None

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    training_env: TrainingEnvironmentConfig
    policy_architecture: PolicyArchitecture = Field(default_factory=ViTDefaultConfig)
    initial_policy_uri: Optional[str] = None
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    gradient_reporter: GradientReporterConfig = Field(default_factory=GradientReporterConfig)

    stats_server_uri: Optional[str] = auto_stats_server_uri()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    group: Optional[str] = None
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    torch_profiler: TorchProfilerConfig = Field(default_factory=TorchProfilerConfig)
    scheduler: SchedulerConfig | None = None

    context_checkpointer: dict[str, Any] = Field(default_factory=dict)
    stats_reporter: StatsReporterConfig = Field(default_factory=StatsReporterConfig)
    wandb_aborter: WandbAborterConfig = Field(default_factory=WandbAborterConfig)

    map_preview_uri: str | None = None
    disable_macbook_optimize: bool = False
    sandbox: bool = False

    def output_references(self, job_name: str) -> dict:
        storage = auto_policy_storage_decision(job_name)
        if storage.remote_prefix:
            policy_uri = storage.remote_prefix
        else:
            policy_uri = f"file://{self.system.data_dir / job_name / 'checkpoints'}"
        return {"policy_uri": policy_uri}

    def invoke(self, args: dict[str, str]) -> int | None:
        if "run" in args:
            assert self.run is None, "run cannot be set via args if already provided in TrainTool config"
            self.run = args["run"]

        self._apply_resume_hints()

        if self.run is None:
            self.run = auto_run_name(prefix="local")

        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.run)

        if self.group:
            self.wandb.group = self.group

        if platform.system() == "Darwin" and not self.disable_macbook_optimize:
            self._minimize_config_for_debugging()  # this overrides many config settings for local testings

        if self.sandbox:
            self._apply_sandbox_config()
            logger.info("Running in sandbox mode (fast validation: 1M steps, epoch-1 checkpoints/evals)")

        # Ensure we checkpoint whenever we evaluate by making checkpointer.epoch_interval
        # a divisor of evaluator.epoch_interval
        if self.evaluator.epoch_interval != 0:
            if self.evaluator.epoch_interval % self.checkpointer.epoch_interval != 0:
                logger.warning(
                    "evaluator.epoch_interval (%d) is not a multiple of checkpointer.epoch_interval (%d). "
                    "Adjusting checkpointer.epoch_interval to %d to ensure checkpoints occur during evaluations.",
                    self.evaluator.epoch_interval,
                    self.checkpointer.epoch_interval,
                    self.evaluator.epoch_interval,
                )
                self.checkpointer.epoch_interval = self.evaluator.epoch_interval

        if self.evaluator.evaluate_local:
            # suppress NCCL watchdog timeouts while ranks wait for master to complete evals
            logger.warning("Local policy evaluation can be inefficient - consider switching to remote evaluation!")
            self.system.nccl_timeout = timedelta(hours=4)

        distributed_helper = DistributedHelper(self.system)
        distributed_helper.scale_batch_config(self.trainer, self.training_env)

        self.training_env.seed += distributed_helper.get_rank()

        sup_uri = self.training_env.supervisor_policy_uri
        supervisor_policy_spec: PolicySpec | None = None
        if sup_uri:
            candidate = Path(sup_uri)
            if "://" in sup_uri or candidate.suffix or os.sep in sup_uri or candidate.parent != Path("."):
                supervisor_policy_spec = policy_spec_from_uri(sup_uri)
            else:
                class_path = resolve_policy_class_path(sup_uri)
                supervisor_policy_spec = PolicySpec(class_path=class_path)

        run_name = self.run or "default"
        preflight_executor: ThreadPoolExecutor | None = None
        storage_future: Future[PolicyStorageDecision] | None = None
        stats_future: Future[Optional[StatsClient]] | None = None
        storage_decision: PolicyStorageDecision | None = None
        stats_client: Optional[StatsClient] = None
        needs_preflight = not self.system.local_only or (distributed_helper.is_master() and self.stats_server_uri)
        start_method = multiprocessing.get_start_method(allow_none=True)
        if start_method is None:
            start_method = multiprocessing.get_context().get_start_method()
        can_thread_preflight = needs_preflight and (
            self.training_env.vectorization == "serial" or start_method != "fork"
        )
        if can_thread_preflight:
            preflight_executor = ThreadPoolExecutor(max_workers=2)
            if not self.system.local_only:
                storage_future = preflight_executor.submit(auto_policy_storage_decision, run_name)
            if distributed_helper.is_master() and self.stats_server_uri:
                stats_future = preflight_executor.submit(self._maybe_create_stats_client, distributed_helper)

        env = VectorizedTrainingEnvironment(self.training_env, supervisor_policy_spec=supervisor_policy_spec)

        if needs_preflight and not can_thread_preflight:
            if not self.system.local_only:
                storage_decision = auto_policy_storage_decision(run_name)
            if distributed_helper.is_master() and self.stats_server_uri:
                stats_client = self._maybe_create_stats_client(distributed_helper)

        self._configure_torch_backends()

        if storage_future:
            storage_decision = storage_future.result()

        checkpoint_manager = CheckpointManager(
            run=run_name,
            system_cfg=self.system,
            require_remote_enabled=self.evaluator.evaluate_remote,
            storage_decision=storage_decision,
        )

        init_logging(run_dir=checkpoint_manager.run_dir)
        record_heartbeat()

        checkpointer = Checkpointer(
            config=self.checkpointer,
            checkpoint_manager=checkpoint_manager,
            distributed_helper=distributed_helper,
            policy_architecture=self.policy_architecture,
        )
        policy = checkpointer.load_or_create_policy(
            env.policy_env_info,
            policy_uri=self.initial_policy_uri,
        )

        if distributed_helper.is_master():
            total_params = sum(param.numel() for param in policy.parameters())
            trainable_params = sum(param.numel() for param in policy.parameters() if param.requires_grad)
            logging.info("policy parameters: total=%d trainable=%d", total_params, trainable_params)

        trainer = self._initialize_trainer(env, policy, distributed_helper)

        self._log_run_configuration(distributed_helper, checkpoint_manager, env)

        if stats_future:
            stats_client = stats_future.result()
        elif stats_client is None:
            stats_client = self._maybe_create_stats_client(distributed_helper)

        if preflight_executor is not None:
            preflight_executor.shutdown(wait=False)
        wandb_manager = self._build_wandb_manager(distributed_helper)

        try:
            with wandb_manager as wandb_run:
                self._register_components(
                    trainer=trainer,
                    distributed_helper=distributed_helper,
                    checkpoint_manager=checkpoint_manager,
                    stats_client=stats_client,
                    policy_checkpointer=checkpointer,
                    run_name=self.run,
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
            logger.error(f"Training failed with exception: {e}", exc_info=True)
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

    def _apply_resume_hints(self) -> None:
        run_from_uri: str | None = None
        if self.initial_policy_uri:
            try:
                parsed = resolve_uri(self.initial_policy_uri)
            except Exception:
                parsed = None
            if parsed and parsed.checkpoint_info:
                run_from_uri = parsed.checkpoint_info[0]

        if run_from_uri and self.run is None:
            self.run = run_from_uri

        if not self.run:
            return

        trainer_state_path = self.system.data_dir / self.run / "checkpoints" / "trainer_state.pt"
        if trainer_state_path.exists():
            logger.info("Trainer state found at %s; optimizer/curriculum state will be restored.", trainer_state_path)

    def _register_components(
        self,
        *,
        trainer: Trainer,
        distributed_helper: DistributedHelper,
        checkpoint_manager: CheckpointManager,
        stats_client: Optional[StatsClient],
        policy_checkpointer: Checkpointer,
        run_name: str,
        wandb_run: WandbRun | None,
    ) -> None:
        components: list[TrainerComponent] = []
        trainer_checkpointer = ContextCheckpointer(
            checkpoint_manager=checkpoint_manager,
            distributed_helper=distributed_helper,
            epoch_interval=max(1, self.checkpointer.epoch_interval),
        )

        heartbeat_cfg = getattr(self.trainer, "heartbeat", None)
        if heartbeat_cfg is not None:
            components.append(Heartbeat(epoch_interval=heartbeat_cfg.epoch_interval))

        autotune_cfg = getattr(self.trainer, "update_epochs_autotune", None)
        if autotune_cfg and getattr(autotune_cfg, "enabled", False):
            components.append(UpdateEpochAutoTuner(autotune_cfg))

        if distributed_helper.is_master():
            stats_config = self.stats_reporter.model_copy(update={"report_to_wandb": bool(wandb_run)})
            reporting_enabled = stats_config.report_to_wandb or stats_config.report_to_console

            if self.gradient_reporter.epoch_interval:
                components.append(GradientReporter(self.gradient_reporter))

            stats_component = StatsReporter.from_config(
                stats_config,
                wandb_run=wandb_run,
            )
            components.append(stats_component)

            components.append(trainer_checkpointer)
            components.append(policy_checkpointer)

            self.evaluator = self.evaluator.model_copy(deep=True)
            components.append(
                Evaluator(
                    config=self.evaluator,
                    device=torch.device(self.system.device),
                    seed=self.system.seed,
                    run_name=run_name,
                    stats_client=stats_client,
                    wandb_run=wandb_run,
                )
            )

            components.append(Monitor(enabled=reporting_enabled))
            components.append(ProgressLogger())
        else:
            components.append(trainer_checkpointer)
            components.append(policy_checkpointer)

        if self.context_checkpointer:
            logger.debug(
                "Context checkpointer configuration is ignored; checkpointing is policy-driven now: %s",
                self.context_checkpointer,
            )

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
            trainer.register(component)

        if wandb_run is not None and distributed_helper.is_master():
            trainer.register(WandbLogger(wandb_run))

        if self.scheduler is not None:
            trainer.register(LossScheduler(self.scheduler))

    def _configure_torch_backends(self) -> None:
        if not torch.cuda.is_available():
            return

        # Opportunistically enable flash attention when available
        if os.environ.get("FLASH_ATTENTION") is None:
            try:
                import flash_attn  # noqa: F401 # type: ignore[import-not-found]
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
        self.evaluator.epoch_interval = min(self.evaluator.epoch_interval, 10)

    def _apply_sandbox_config(self) -> None:
        """Apply sandbox mode configuration for fast validation testing."""
        # Reduce total timesteps for very quick testing (1M instead of 50B)
        self.trainer.total_timesteps = 1_000_000

        # Save checkpoint after first epoch
        self.checkpointer.epoch_interval = 1

        # Run evaluation after first epoch with short episodes for fast validation
        self.evaluator.epoch_interval = 1
        self.evaluator.allow_eval_without_stats = True

        # Create a short evaluation environment (100 steps instead of 1000+)
        # This makes evaluations complete in ~10-20 seconds instead of minutes
        curriculum = Curriculum(self.training_env.curriculum)
        eval_env = curriculum.get_task().get_env_cfg().model_copy(deep=True)
        eval_env.game.max_steps = 100

        self.evaluator.training_replay_envs = [
            SimulationConfig(
                suite="training",
                name="sandbox_validation",
                env=eval_env,
            )
        ]
        # Clear any additional simulations - only run the quick training validation
        self.evaluator.simulations = []
