import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Set
from uuid import UUID

import einops
import numpy as np
import torch
import torch.distributed
import wandb
from heavyball import ForeachMuon
from omegaconf import DictConfig, ListConfig

from app_backend.stats_client import StatsClient
from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent
from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.agent.util.debug import assert_shape
from metta.common.memory_monitor import MemoryMonitor
from metta.common.stopwatch import Stopwatch, with_instance_timer
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.common.util.wandb.wandb_context import WandbRun
from metta.eval.eval_stats_db import EvalStatsDB
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import MettaGridEnv, dtype_actions
from metta.mettagrid.util.dict_utils import unroll_nested_dict
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite

try:
    from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None

torch.set_float32_matmul_precision("high")

# Get rank for logger name
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = logging.getLogger(f"trainer-{rank}-{local_rank}")


class MettaTrainer:
    def __init__(
        self,
        cfg: DictConfig | ListConfig,
        wandb_run: WandbRun | None,
        policy_store: PolicyStore,
        sim_suite_config: SimulationSuiteConfig,
        stats_client: StatsClient | None,
        **kwargs: Any,
    ):
        self.cfg = cfg
        self.trainer_cfg = trainer_cfg = cfg.trainer

        # it doesn't make sense to evaluate more often than we checkpoint since we need a saved policy to evaluate
        if trainer_cfg.evaluate_interval != 0 and trainer_cfg.evaluate_interval < trainer_cfg.checkpoint_interval:
            raise ValueError("evaluate_interval must be at least as large as checkpoint_interval")

        if trainer_cfg.evaluate_interval != 0 and trainer_cfg.evaluate_interval < trainer_cfg.wandb_checkpoint_interval:
            raise ValueError("evaluate_interval must be at least as large as wandb_checkpoint_interval")

        # Validate that we save policies locally at least as often as we upload to wandb
        if (
            trainer_cfg.wandb_checkpoint_interval != 0
            and trainer_cfg.checkpoint_interval != 0
            and trainer_cfg.wandb_checkpoint_interval < trainer_cfg.checkpoint_interval
        ):
            raise ValueError(
                "wandb_checkpoint_interval must be at least as large as checkpoint_interval "
                "to ensure policies exist locally before uploading to wandb"
            )

        if trainer_cfg.checkpoint_dir:
            os.makedirs(trainer_cfg.checkpoint_dir, exist_ok=True)

        self.sim_suite_config = sim_suite_config
        self._stats_client = stats_client

        self._master = True
        self._world_size = 1
        self.device: torch.device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        self._batch_size = trainer_cfg.batch_size
        self._minibatch_size = trainer_cfg.minibatch_size
        if torch.distributed.is_initialized():
            self._master = int(os.environ["RANK"]) == 0
            self._world_size = torch.distributed.get_world_size()

            self._batch_size = trainer_cfg.batch_size // self._world_size
            self._minibatch_size = trainer_cfg.minibatch_size // self._world_size

            logger.info(
                f"Rank: {os.environ['RANK']}, Local rank: {os.environ['LOCAL_RANK']}, World size: {self._world_size}"
            )

        self.torch_profiler = TorchProfiler(self._master, cfg.run_dir, trainer_cfg.profiler_interval_epochs, wandb_run)
        self.losses = Losses()
        self.stats = defaultdict(list)
        self.grad_stats = {}
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.evals: dict[str, float] = {}

        self.timer = Stopwatch(logger)
        self.timer.start()

        self._memory_monitor = MemoryMonitor()

        self._system_monitor = SystemMonitor(
            sampling_interval_sec=1.0,  # Sample every second
            history_size=100,  # Keep last 100 samples
            logger=logger,
            auto_start=True,  # Start monitoring immediately
        )

        curriculum_config = trainer_cfg.get("curriculum", trainer_cfg.get("env", {}))
        env_overrides = DictConfig(trainer_cfg.env_overrides)
        self._curriculum = curriculum_from_config_path(curriculum_config, env_overrides)
        self._make_vecenv()

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), (
            f"vecenv.driver_env type {type(metta_grid_env).__name__} is not MettaGridEnv"
        )

        self.agent_step: int = 0
        self.epoch: int = 0

        checkpoint = TrainerCheckpoint.load(cfg.run_dir)
        if checkpoint:
            logger.info(f"Restoring from checkpoint at {checkpoint.agent_step} steps")
            self.agent_step = checkpoint.agent_step
            self.epoch = checkpoint.epoch
            if checkpoint.stopwatch_state is not None:
                logger.info("Restoring timer state from checkpoint")
                self.timer.load_state(checkpoint.stopwatch_state, resume_running=True)

        # Note that these fields are specific to MettaGridEnv, which is why we can't keep
        # self.vecenv.driver_env as just the parent class pufferlib.PufferEnv
        actions_names = metta_grid_env.action_names
        actions_max_params = metta_grid_env.max_action_args

        # Load or create policy with proper distributed coordination
        policy_record = self._load_policy(checkpoint, policy_store)

        if policy_record is not None:
            # Models loaded via torch.package have modified class names (prefixed with <torch_package_N>)
            # which prevents them from being saved again. We work around this by creating a fresh
            # instance of the policy class and copying the state dict, allowing successful re-saving.
            # TODO: Remove this workaround when checkpointing refactor is complete
            loaded_policy = policy_record.policy()
            loaded_policy.activate_actions(actions_names, actions_max_params, self.device)

            fresh_policy_record = policy_store.create(metta_grid_env)
            fresh_policy = fresh_policy_record.policy()
            fresh_policy.activate_actions(actions_names, actions_max_params, self.device)
            fresh_policy.load_state_dict(loaded_policy.state_dict(), strict=False)

            self.initial_policy_record = fresh_policy_record
            self.policy = fresh_policy

        else:
            if self._master:
                policy_record = self._create_policy(policy_store, metta_grid_env)
            else:
                policy_record = self._wait_for_policy(policy_store)
            self.initial_policy_record = policy_record
            self.policy = policy_record.policy()
            self.policy.activate_actions(actions_names, actions_max_params, self.device)

        assert self.policy is not None, "Failed to obtain policy"

        self.latest_saved_policy_record = self.initial_policy_record

        if self._master:
            logger.info(f"MettaTrainer loaded: {self.policy}")

        if trainer_cfg.compile:
            logger.info("Compiling policy")
            self.policy = torch.compile(self.policy, mode=trainer_cfg.compile_mode)

        self.kickstarter = Kickstarter(cfg, policy_store, actions_names, actions_max_params)

        if torch.distributed.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
            self.policy = DistributedMettaAgent(self.policy, self.device)

        self._make_experience_buffer()

        self._stats_epoch_start = self.epoch
        self._stats_epoch_id: UUID | None = None
        self._stats_run_id: UUID | None = None

        # Optimizer
        optimizer_type = getattr(trainer_cfg.optimizer, "type", "adam")
        assert optimizer_type in ("adam", "muon"), f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}"
        opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
        self.optimizer = opt_cls(
            self.policy.parameters(),
            lr=trainer_cfg.optimizer.learning_rate,
            betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
            eps=trainer_cfg.optimizer.eps,
            weight_decay=trainer_cfg.optimizer.weight_decay,
        )

        # validate that policy matches environment
        if isinstance(self.policy, MettaAgent):
            agent = self.policy
        elif isinstance(self.policy, DistributedMettaAgent):
            agent = self.policy.module
        else:
            raise ValueError(f"Policy must be of type MettaAgent or DistributedMettaAgent, got {type(self.policy)}")

        _env_shape = metta_grid_env.single_observation_space.shape
        environment_shape = tuple(_env_shape) if isinstance(_env_shape, list) else _env_shape

        # The rest of the validation logic continues to work with duck typing
        if hasattr(agent, "components"):
            found_match = False
            for component_name, component in agent.components.items():
                if hasattr(component, "_obs_shape"):
                    found_match = True
                    component_shape = (
                        tuple(component._obs_shape) if isinstance(component._obs_shape, list) else component._obs_shape
                    )
                    if component_shape != environment_shape:
                        raise ValueError(
                            f"Observation space mismatch error:\n"
                            f"[policy] component_name: {component_name}\n"
                            f"[policy] component_shape: {component_shape}\n"
                            f"environment_shape: {environment_shape}\n"
                        )

            if not found_match:
                raise ValueError(
                    "No component with observation shape found in policy. "
                    f"Environment observation shape: {environment_shape}"
                )

        self.lr_scheduler = None
        if hasattr(trainer_cfg, "lr_scheduler") and getattr(trainer_cfg.lr_scheduler, "enabled", False):
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=trainer_cfg.total_timesteps // trainer_cfg.batch_size
            )

        if checkpoint and checkpoint.optimizer_state_dict:
            try:
                self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                logger.info("Successfully loaded optimizer state from checkpoint")
            except ValueError as e:
                if "doesn't match the size of optimizer's group" in str(e):
                    # Extract some info about the mismatch
                    old_params = len(checkpoint.optimizer_state_dict.get("param_groups", [{}])[0].get("params", []))
                    new_params = sum(1 for _ in self.policy.parameters())
                    logger.warning(
                        f"Optimizer state dict doesn't match current model architecture. "
                        f"Checkpoint has {old_params} parameter groups, current model has {new_params}. "
                        "This typically happens when layers are added/removed. "
                        "Starting with fresh optimizer state."
                    )
                else:
                    # Re-raise if it's a different ValueError
                    raise

        if wandb_run and self._master:
            # Define metrics (wandb x-axis values)
            metrics = ["agent_step", "epoch", "total_time", "train_time"]
            for metric in metrics:
                wandb_run.define_metric(f"metric/{metric}")

            # set the default x-axis to be step count
            wandb_run.define_metric("*", step_metric="metric/agent_step")

            # set up plots that do not use steps as the x-axis
            metric_overrides = [
                ("overview/reward_vs_total_time", "metric/total_time"),
            ]

            for metric_name, step_metric in metric_overrides:
                wandb_run.define_metric(metric_name, step_metric=step_metric)

        self._memory_monitor.add(self)

        logger.info(f"MettaTrainer initialization complete on device: {self.device}")

    def train(self) -> None:
        logger.info("Starting training")
        trainer_cfg = self.trainer_cfg

        if self._stats_client is not None:
            name = self.wandb_run.name if self.wandb_run is not None and self.wandb_run.name is not None else "unknown"
            url = self.wandb_run.url if self.wandb_run is not None else None
            self._stats_run_id = self._stats_client.create_training_run(name=name, attributes={}, url=url).id

        logger.info(f"Training on {self.device}")
        wandb_policy_name: str | None = None
        while self.agent_step < trainer_cfg.total_timesteps:
            steps_before = self.agent_step

            with self.torch_profiler:
                self._rollout()
                self._train()

            self.torch_profiler.on_epoch_end(self.epoch)

            # Processing stats
            self._process_stats()

            rollout_time = self.timer.get_last_elapsed("_rollout")
            train_time = self.timer.get_last_elapsed("_train")
            stats_time = self.timer.get_last_elapsed("_process_stats")
            steps_calculated = self.agent_step - steps_before

            total_time = train_time + rollout_time + stats_time
            steps_per_sec = steps_calculated / total_time

            train_pct = (train_time / total_time) * 100
            rollout_pct = (rollout_time / total_time) * 100
            stats_pct = (stats_time / total_time) * 100

            logger.info(
                f"Epoch {self.epoch} - "
                f"{steps_per_sec * self._world_size:.0f} steps/sec "
                f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
            )
            record_heartbeat()

            # Interval periodic tasks
            self._maybe_save_policy()
            self._maybe_save_training_state()
            wandb_policy_name = self._maybe_upload_policy_record_to_wandb()
            self._maybe_evaluate_policy(wandb_policy_name)
            self._maybe_generate_replay()
            self._maybe_update_l2_weights()
            self._maybe_compute_grad_stats()

            self._on_train_step()
            # end loop over total_timesteps

        logger.info("Training complete!")
        timing_summary = self.timer.get_all_summaries()

        for name, summary in timing_summary.items():
            logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        # Force final saves
        self._maybe_save_policy(force=True)
        self._maybe_save_training_state(force=True)
        self._maybe_upload_policy_record_to_wandb(force=True)

        self._system_monitor.stop()

    def _on_train_step(self):
        pass

    @with_instance_timer("_rollout")
    def _rollout(self):
        experience = self.experience
        trainer_cfg = self.trainer_cfg
        device = self.device

        policy = self.policy
        infos = defaultdict(list)
        raw_infos = []  # Collect raw info for batch processing later
        experience.reset_for_rollout()

        while not experience.ready_for_training:
            with self.timer("_rollout.env"):
                o, r, d, t, info, env_id, mask = self.vecenv.recv()
                if trainer_cfg.require_contiguous_env_ids:
                    raise ValueError(
                        "We are assuming contiguous eng id is always False. async_factor == num_workers = "
                        f"{trainer_cfg.async_factor} != {trainer_cfg.num_workers}"
                    )

                training_env_id = slice(env_id[0], env_id[-1] + 1)

            # Convert mask to tensor once
            mask = torch.as_tensor(mask)
            num_steps = int(mask.sum().item())
            self.agent_step += num_steps

            # Convert to tensors once
            o = torch.as_tensor(o).to(device, non_blocking=True)
            r = torch.as_tensor(r).to(device, non_blocking=True)
            d = torch.as_tensor(d).to(device, non_blocking=True)
            t = torch.as_tensor(t).to(device, non_blocking=True)

            with torch.no_grad():
                state = PolicyState()

                # Use LSTM state access for performance
                lstm_h, lstm_c = experience.get_lstm_state(training_env_id.start)
                if lstm_h is not None:
                    state.lstm_h = lstm_h
                    state.lstm_c = lstm_c

                # Use pre-moved tensor
                actions, selected_action_log_probs, _, value, _ = policy(o, state)

                if __debug__:
                    assert_shape(selected_action_log_probs, ("BT",), "selected_action_log_probs")
                    assert_shape(actions, ("BT", 2), "actions")

                # Store LSTM state for performance
                lstm_state_to_store = None
                if state.lstm_h is not None:
                    lstm_state_to_store = {"lstm_h": state.lstm_h.detach(), "lstm_c": state.lstm_c.detach()}

                if str(self.device).startswith("cuda"):
                    torch.cuda.synchronize()

            value = value.flatten()
            # mask already converted to tensor above

            # All tensors are already on device, avoid redundant transfers
            experience.store(
                obs=o,
                actions=actions,
                logprobs=selected_action_log_probs,
                rewards=r,
                dones=d,
                truncations=t,
                values=value,
                env_id=training_env_id,
                mask=mask,
                lstm_state=lstm_state_to_store,
            )

            # At this point, infos contains lists of values collected across:
            # 1. Multiple vectorized environments managed by this GPU's vecenv
            # 2. Multiple rollout steps (until experience buffer is full)
            #
            # - Some stats (like "episode/reward") appear only when episodes complete
            # - Other stats might appear every step
            #
            # These will later be averaged in _process_stats() to get mean values
            # across all environments on this GPU. Stats from other GPUs (if using
            # distributed training) are handled separately and not aggregated here.
            if info:
                raw_infos.extend(info)

            with self.timer("_rollout.env"):
                self.vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Batch process info dictionaries after rollout
        for i in raw_infos:
            for k, v in unroll_nested_dict(i):
                infos[k].append(v)

        # Batch process stats more efficiently
        for k, v in infos.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()

            if isinstance(v, list):
                self.stats.setdefault(k, []).extend(v)
            else:
                if k not in self.stats:
                    self.stats[k] = v
                else:
                    try:
                        self.stats[k] += v
                    except TypeError:
                        self.stats[k] = [self.stats[k], v]  # fallback: bundle as list

        # TODO: Better way to enable multiple collects
        return self.stats, infos

    @with_instance_timer("_train")
    def _train(self):
        experience = self.experience
        trainer_cfg = self.trainer_cfg

        self.losses.zero()

        prio_cfg = trainer_cfg.get("prioritized_experience_replay", {})
        vtrace_cfg = trainer_cfg.get("vtrace", {})

        # Reset importance sampling ratios
        experience.reset_importance_sampling_ratios()

        # Prioritized sampling parameters
        b0 = prio_cfg.get("prio_beta0", 0.6)
        a = prio_cfg.get("prio_alpha", 0.0)
        total_epochs = max(1, trainer_cfg.total_timesteps // trainer_cfg.batch_size)
        anneal_beta = b0 + (1 - b0) * a * self.epoch / total_epochs

        # Compute advantages using puff_advantage
        advantages = torch.zeros(experience.values.shape, device=self.device)

        # Initial importance sampling ratio is all ones
        initial_importance_sampling_ratio = torch.ones_like(experience.values)

        advantages = self._compute_advantage(
            experience.values,
            experience.rewards,
            experience.dones,
            initial_importance_sampling_ratio,
            advantages,
            trainer_cfg.gamma,
            trainer_cfg.gae_lambda,
            vtrace_cfg.get("vtrace_rho_clip", 1.0),
            vtrace_cfg.get("vtrace_c_clip", 1.0),
        )

        # Optimizing the policy and value network
        _total_minibatches = experience.num_minibatches * trainer_cfg.update_epochs
        minibatch_idx = 0

        for _epoch in range(trainer_cfg.update_epochs):
            for _ in range(experience.num_minibatches):
                minibatch = experience.sample_minibatch(
                    advantages=advantages,
                    prio_alpha=a,
                    prio_beta=anneal_beta,
                    minibatch_idx=minibatch_idx,
                    total_minibatches=_total_minibatches,
                )

                obs = minibatch["obs"]

                lstm_state = PolicyState()
                _, new_logprobs, entropy, newvalue, full_logprobs = self.policy(
                    obs, lstm_state, action=minibatch["actions"]
                )

                new_logprobs = new_logprobs.reshape(minibatch["logprobs"].shape)
                logratio = new_logprobs - minibatch["logprobs"]
                importance_sampling_ratio = logratio.exp()
                experience.update_ratio(minibatch["indices"], importance_sampling_ratio)

                with torch.no_grad():
                    approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
                    clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.clip_coef).float().mean()

                # Re-compute advantages with new ratios (V-trace)
                adv = self._compute_advantage(
                    minibatch["values"],
                    minibatch["rewards"],
                    minibatch["dones"],
                    importance_sampling_ratio,
                    minibatch["advantages"],
                    trainer_cfg.gamma,
                    trainer_cfg.gae_lambda,
                    vtrace_cfg.get("vtrace_rho_clip", 1.0),
                    vtrace_cfg.get("vtrace_c_clip", 1.0),
                )

                # Normalize advantages with distributed support, then apply prioritized weights
                adv = self._normalize_advantage_distributed(adv)
                adv = minibatch["prio_weights"] * adv

                # Policy loss
                pg_loss1 = -adv * importance_sampling_ratio
                pg_loss2 = -adv * torch.clamp(
                    importance_sampling_ratio, 1 - trainer_cfg.clip_coef, 1 + trainer_cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
                if trainer_cfg.clip_vloss:
                    v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
                    v_clipped = minibatch["values"] + torch.clamp(
                        newvalue_reshaped - minibatch["values"],
                        -trainer_cfg.get("vf_clip_coef", 0.1),
                        trainer_cfg.get("vf_clip_coef", 0.1),
                    )
                    v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

                entropy_loss = entropy.mean()

                ks_action_loss, ks_value_loss = self.kickstarter.loss(
                    self.agent_step, full_logprobs, newvalue, obs, teacher_lstm_state=[]
                )

                l2_reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                if trainer_cfg.l2_reg_loss_coef > 0:
                    l2_reg_loss = trainer_cfg.l2_reg_loss_coef * self.policy.l2_reg_loss().to(self.device)

                l2_init_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                if trainer_cfg.l2_init_loss_coef > 0:
                    l2_init_loss = trainer_cfg.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)

                loss = (
                    pg_loss
                    - trainer_cfg.ent_coef * entropy_loss
                    + v_loss * trainer_cfg.vf_coef
                    + l2_reg_loss
                    + l2_init_loss
                    + ks_action_loss
                    + ks_value_loss
                )

                # Update values in experience buffer
                experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))

                if self.losses is None:
                    raise ValueError("self.losses is None")

                # Update loss tracking for logging
                self.losses.policy_loss_sum += pg_loss.item()
                self.losses.value_loss_sum += v_loss.item()
                self.losses.entropy_sum += entropy_loss.item()
                self.losses.approx_kl_sum += approx_kl.item()
                self.losses.clipfrac_sum += clipfrac.item()
                self.losses.l2_reg_loss_sum += l2_reg_loss.item() if torch.is_tensor(l2_reg_loss) else l2_reg_loss
                self.losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
                self.losses.ks_action_loss_sum += ks_action_loss.item()
                self.losses.ks_value_loss_sum += ks_value_loss.item()
                self.losses.importance_sum += importance_sampling_ratio.mean().item()
                self.losses.minibatches_processed += 1

                self.optimizer.zero_grad()
                loss.backward()
                if (minibatch_idx + 1) % self.experience.accumulate_minibatches == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), trainer_cfg.max_grad_norm)
                    self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                    if str(self.device).startswith("cuda"):
                        torch.cuda.synchronize()

                minibatch_idx += 1
                # end loop over minibatches

            self.epoch += 1

            # check early exit if we have reached target_kl
            if trainer_cfg.target_kl is not None:
                average_approx_kl = self.losses.approx_kl_sum / self.losses.minibatches_processed
                if average_approx_kl > trainer_cfg.target_kl:
                    break
            # end loop over epochs

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Calculate explained variance
        y_pred = experience.values.flatten()
        y_true = advantages.flatten() + experience.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        self.losses.explained_variance = explained_var.item() if torch.is_tensor(explained_var) else float("nan")

    def _should_run(self, interval: int, force: bool = False) -> bool:
        """Check if a periodic task should run based on interval and force flag."""
        if not self._master or not interval:
            return False

        if force:
            return True

        return self.epoch % interval == 0

    def _maybe_save_training_state(self, force=False):
        """Save training state if on checkpoint interval"""
        if not self._should_run(self.trainer_cfg.checkpoint_interval, force):
            return

        extra_args = {}
        if self.kickstarter.enabled and self.kickstarter.teacher_uri is not None:
            extra_args["teacher_pr_uri"] = self.kickstarter.teacher_uri

        checkpoint = TrainerCheckpoint(
            agent_step=self.agent_step,
            epoch=self.epoch,
            total_agent_step=self.agent_step * self._world_size,
            optimizer_state_dict=self.optimizer.state_dict(),
            stopwatch_state=self.timer.save_state(),
            policy_path=self.latest_saved_policy_uri,
            extra_args=extra_args,
        )
        checkpoint.save(self.cfg.run_dir)
        logger.info(f"Saved training state at epoch {self.epoch}")

    def _maybe_save_policy(self, force=False):
        """Save policy locally if on checkpoint interval"""
        if not self._should_run(self.trainer_cfg.checkpoint_interval, force):
            return

        name = self.policy_store.make_model_name(self.epoch)
        path = os.path.join(self.trainer_cfg.checkpoint_dir, name)

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), "vecenv.driver_env must be a MettaGridEnv"

        training_time = self.timer.get_elapsed("_rollout") + self.timer.get_elapsed("_train")

        category_scores_map = {key.split("/")[0]: value for key, value in self.evals.items() if key.endswith("/score")}
        category_score_values = [v for k, v in category_scores_map.items()]
        overall_score = sum(category_score_values) / len(category_score_values) if category_score_values else 0

        metadata = {
            "agent_step": self.agent_step,
            "epoch": self.epoch,
            "run": self.cfg.run,
            "action_names": metta_grid_env.action_names,
            "generation": self.current_policy_generation,
            "initial_uri": self.initial_policy_uri,
            "train_time": training_time,
            "score": overall_score,
            "eval_scores": category_scores_map,
        }

        if isinstance(self.policy, MettaAgent):
            policy_to_save = self.policy
        elif isinstance(self.policy, DistributedMettaAgent):
            policy_to_save = self.policy.module
        else:
            raise ValueError(f"Policy must be of type MettaAgent or DistributedMettaAgent, got {type(self.policy)}")

        # Models loaded via torch.package have modified class names (prefixed with <torch_package_N>)
        # which prevents them from being saved again. We work around this by creating a fresh
        # instance of the policy class and copying the state dict, allowing successful re-saving.
        # TODO: Remove this workaround when checkpointing refactor is complete
        logger.info("Creating a fresh policy instance for torch.package to save")
        fresh_policy = self.policy_store.create(metta_grid_env).policy()
        fresh_policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, self.device)
        fresh_policy.load_state_dict(self.policy.state_dict(), strict=False)
        policy_to_save = fresh_policy

        self.latest_saved_policy_record = self.policy_store.save(name, path, policy_to_save, metadata)
        logger.info(f"Saved policy locally: {name}")
        return self.latest_saved_policy_record

    def _maybe_upload_policy_record_to_wandb(self, force: bool = False) -> str | None:
        """Upload policy to wandb if on wandb interval"""
        if not self._should_run(self.trainer_cfg.wandb_checkpoint_interval, force):
            return

        if not self.wandb_run:
            return

        if not self.latest_saved_policy_record:
            logger.warning("No policy record to upload to wandb")
            return

        if not self.wandb_run.name:
            logger.warning("No wandb run name was provided")
            return

        result = self.policy_store.add_to_wandb_run(self.wandb_run.name, self.latest_saved_policy_record)
        logger.info(f"Uploaded policy to wandb at epoch {self.epoch}")
        return result

    def _maybe_update_l2_weights(self, force=False):
        """Update L2 init weights if on update interval"""
        if self._should_run(self.cfg.agent.l2_init_weight_update_interval, force):
            self.policy.update_l2_init_weight_copy()

    def _maybe_evaluate_policy(self, wandb_policy_name: str | None = None, force: bool = False):
        """Evaluate policy if on evaluation interval"""
        if self._should_run(self.trainer_cfg.evaluate_interval, force):
            self._evaluate_policy(wandb_policy_name)

    @with_instance_timer("_evaluate_policy", log_level=logging.INFO)
    def _evaluate_policy(self, wandb_policy_name: str | None = None):
        if self._stats_run_id is not None and self._stats_client is not None:
            self._stats_epoch_id = self._stats_client.create_epoch(
                run_id=self._stats_run_id,
                start_training_epoch=self._stats_epoch_start,
                end_training_epoch=self.epoch,
                attributes={},
            ).id
            self._stats_epoch_start = self.epoch + 1

        logger.info(f"Simulating policy: {self.latest_saved_policy_uri} with config: {self.sim_suite_config}")
        sim = SimulationSuite(
            config=self.sim_suite_config,
            policy_pr=self.latest_saved_policy_record,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.cfg.vectorization,
            stats_dir="/tmp/stats",
            stats_client=self._stats_client,
            stats_epoch_id=self._stats_epoch_id,
            wandb_policy_name=wandb_policy_name,
        )
        result = sim.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)
        logger.info("Simulation complete")

        # Build evaluation metrics
        self.evals = {}  # used for wandb
        categories: Set[str] = set()
        for sim_name in self.sim_suite_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        for category in categories:
            score = stats_db.get_average_metric_by_filter(
                "reward", self.latest_saved_policy_record, f"sim_name LIKE '%{category}%'"
            )
            logger.info(f"{category} score: {score}")
            record_heartbeat()
            if score is None:
                continue
            self.evals[f"{category}/score"] = score

        # Get detailed per-simulation scores
        all_scores = stats_db.simulation_scores(self.latest_saved_policy_record, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            self.evals[f"{category}/{sim_short_name}"] = score

    def _maybe_generate_replay(self, force=False):
        """Generate replay if on replay interval"""
        if self._should_run(self.trainer_cfg.replay_interval, force):
            self._generate_and_upload_replay()

    @with_instance_timer("_generate_and_upload_replay", log_level=logging.INFO)
    def _generate_and_upload_replay(self):
        replay_sim_config = SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",
            num_episodes=1,
            env_overrides=self._curriculum.get_task().env_cfg(),
        )

        replay_simulator = Simulation(
            name=f"replay_{self.epoch}",
            config=replay_sim_config,
            policy_pr=self.latest_saved_policy_record,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.cfg.vectorization,
            replay_dir=self.trainer_cfg.replay_dir,
        )
        results = replay_simulator.simulate()

        if self.wandb_run is not None:
            replay_urls = results.stats_db.get_replay_urls(
                policy_key=self.latest_saved_policy_record.key(),
                policy_version=self.latest_saved_policy_record.version(),
            )
            if len(replay_urls) > 0:
                replay_url = replay_urls[0]
                player_url = "https://metta-ai.github.io/metta/?replayUrl=" + replay_url
                link_summary = {
                    "replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {self.epoch})</a>')
                }
                self.wandb_run.log(link_summary)

    @with_instance_timer("_process_stats")
    def _process_stats(self):
        if not self._master or not self.wandb_run:
            self.stats.clear()
            return

        # convert lists of values (collected across all environments and rollout steps on this GPU)
        # into single mean values.
        mean_stats = {}
        for k, v in self.stats.items():
            try:
                mean_stats[k] = np.mean(v)
            except (TypeError, ValueError) as e:
                raise RuntimeError(
                    f"Cannot compute mean for stat '{k}' with value {v!r} (type: {type(v)}). "
                    f"All collected stats must be numeric values or lists of numeric values. "
                    f"Error: {e}"
                ) from e
        self.stats = mean_stats

        weight_stats = {}
        if self.cfg.agent.analyze_weights_interval != 0 and self.epoch % self.cfg.agent.analyze_weights_interval == 0:
            for metrics in self.policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key != "name":
                        weight_stats[f"weights/{key}/{name}"] = value

        elapsed_times = self.timer.get_all_elapsed()
        wall_time = self.timer.get_elapsed()
        train_time = elapsed_times.get("_rollout", 0) + elapsed_times.get("_train", 0)

        lap_times = self.timer.lap_all(self.agent_step, exclude_global=False)
        wall_time_for_lap = lap_times.pop("global", 0)

        # X-axis values for wandb
        metric_stats = {
            "metric/agent_step": self.agent_step * self._world_size,
            "metric/epoch": self.epoch,
            "metric/total_time": wall_time,
            "metric/train_time": train_time,
        }

        epoch_steps = self.timer.get_lap_steps()
        if epoch_steps is None:
            epoch_steps = self.agent_step
        epoch_steps_per_second = epoch_steps / wall_time_for_lap if wall_time_for_lap > 0 else 0
        steps_per_second = self.timer.get_rate(self.agent_step) if wall_time > 0 else 0

        epoch_steps_per_second *= self._world_size
        steps_per_second *= self._world_size

        timing_stats = {
            **{
                f"timing_per_epoch/frac/{op}": lap_elapsed / wall_time_for_lap if wall_time_for_lap > 0 else 0
                for op, lap_elapsed in lap_times.items()
            },
            "timing_per_epoch/sps": epoch_steps_per_second,
            **{
                f"timing_cumulative/frac/{op}": elapsed / wall_time if wall_time > 0 else 0
                for op, elapsed in elapsed_times.items()
            },
            "timing_cumulative/sps": steps_per_second,
        }

        environment_stats = {f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in self.stats.items()}

        overview = {
            "sps": epoch_steps_per_second,
        }

        # Calculate average reward from all env_task_reward entries
        task_reward_values = [v for k, v in environment_stats.items() if k.startswith("env_task_reward")]
        if task_reward_values:
            mean_reward = sum(task_reward_values) / len(task_reward_values)
            overview["reward"] = mean_reward
            overview["reward_vs_total_time"] = mean_reward

        # include custom stats from trainer config
        if hasattr(self.trainer_cfg, "stats") and hasattr(self.trainer_cfg.stats, "overview"):
            for k, v in self.trainer_cfg.stats.overview.items():
                if k in self.stats:
                    overview[v] = self.stats[k]

        category_scores_map = {key.split("/")[0]: value for key, value in self.evals.items() if key.endswith("/score")}

        for category, score in category_scores_map.items():
            overview[f"{category}_score"] = score

        losses = self.losses.stats()

        # don't plot losses that are unused
        if self.trainer_cfg.l2_reg_loss_coef == 0:
            losses.pop("l2_reg_loss")
        if self.trainer_cfg.l2_init_loss_coef == 0:
            losses.pop("l2_init_loss")
        if not self.kickstarter.enabled:
            losses.pop("ks_action_loss")
            losses.pop("ks_value_loss")

        parameters = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_steps": epoch_steps,
            "num_minibatches": self.experience.num_minibatches,
        }

        self.wandb_run.log(
            {
                **{f"overview/{k}": v for k, v in overview.items()},
                **{f"losses/{k}": v for k, v in losses.items()},
                **{f"experience/{k}": v for k, v in self.experience.stats().items()},
                **{f"parameters/{k}": v for k, v in parameters.items()},
                **{f"eval_{k}": v for k, v in self.evals.items()},
                **{f"monitor/{k}": v for k, v in self._system_monitor.stats().items()},
                **{f"trainer_memory/{k}": v for k, v in self._memory_monitor.stats().items()},
                **environment_stats,
                **weight_stats,
                **timing_stats,
                **metric_stats,
                **self.grad_stats,
            }
        )

        self.stats.clear()
        self.grad_stats.clear()

    def _compute_advantage(
        self,
        values,
        rewards,
        dones,
        importance_sampling_ratio,
        advantages,
        gamma,
        gae_lambda,
        vtrace_rho_clip,
        vtrace_c_clip,
    ):
        """CUDA kernel for puffer advantage with automatic CPU fallback."""

        # Get correct device for this process
        device = torch.device(self.device) if isinstance(self.device, str) else self.device

        # Move tensors to device and compute advantage
        tensors = [values, rewards, dones, importance_sampling_ratio, advantages]
        tensors = [t.to(device) for t in tensors]
        values, rewards, dones, importance_sampling_ratio, advantages = tensors

        # Create context manager that only applies CUDA device context if needed
        device_context = torch.cuda.device(device) if str(device).startswith("cuda") else nullcontext()
        with device_context:
            torch.ops.pufferlib.compute_puff_advantage(
                values,
                rewards,
                dones,
                importance_sampling_ratio,
                advantages,
                gamma,
                gae_lambda,
                vtrace_rho_clip,
                vtrace_c_clip,
            )

        return advantages

    def _normalize_advantage_distributed(self, adv: torch.Tensor) -> torch.Tensor:
        """Normalize advantages with distributed training support while preserving shape."""
        if not self.trainer_cfg.get("norm_adv", True):
            return adv

        if torch.distributed.is_initialized():
            # Compute local statistics
            adv_flat = adv.view(-1)
            local_sum = einops.rearrange(adv_flat.sum(), "-> 1")
            local_sq_sum = einops.rearrange((adv_flat * adv_flat).sum(), "-> 1")
            local_count = torch.tensor([adv_flat.numel()], dtype=adv.dtype, device=adv.device)

            # Combine statistics for single all_reduce
            stats = einops.rearrange([local_sum, local_sq_sum, local_count], "one float -> (float one)")
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

            # Extract global statistics
            global_sum, global_sq_sum, global_count = stats[0], stats[1], stats[2]
            global_mean = global_sum / global_count
            global_var = (global_sq_sum / global_count) - (global_mean * global_mean)
            global_std = torch.sqrt(global_var.clamp(min=1e-8))

            # Normalize and reshape back
            adv = (adv - global_mean) / (global_std + 1e-8)
        else:
            # Local normalization
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return adv

    def close(self):
        self.vecenv.close()
        self._memory_monitor.clear()

    @property
    def latest_saved_policy_uri(self) -> str | None:
        """Get the URI of the latest saved policy, if any."""
        if self.latest_saved_policy_record is None:
            return None
        return self.latest_saved_policy_record.uri

    @property
    def initial_policy_uri(self) -> str | None:
        """Get the URI of the initial policy, if any."""
        if self.initial_policy_record is None:
            return None
        return self.initial_policy_record.uri

    @property
    def current_policy_generation(self) -> int:
        """Get the generation number for new policies saved in this training run.
        This is the initial policy's generation + 1, representing that we're
        training the next generation from that starting point.
        """
        if self.initial_policy_record is None:
            return 0
        return self.initial_policy_record.metadata.get("generation", 0) + 1

    def _make_experience_buffer(self):
        """Create experience buffer with tensor-based storage for prioritized sampling."""
        vecenv = self.vecenv
        trainer_cfg = self.trainer_cfg

        # Get environment info
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents

        # Calculate minibatch parameters
        minibatch_size = trainer_cfg.minibatch_size
        max_minibatch_size = trainer_cfg.get("max_minibatch_size", minibatch_size)

        # Get LSTM parameters
        hidden_size = getattr(self.policy, "hidden_size", 256)
        num_lstm_layers = 2  # Default value

        # Try to get actual number of LSTM layers from policy
        if hasattr(self.policy, "components") and "_core_" in self.policy.components:
            lstm_module = self.policy.components["_core_"]
            if hasattr(lstm_module, "_net") and hasattr(lstm_module._net, "num_layers"):
                num_lstm_layers = lstm_module._net.num_layers

        # Create experience buffer
        self.experience = Experience(
            total_agents=total_agents,
            batch_size=self._batch_size,
            bptt_horizon=trainer_cfg.bptt_horizon,
            minibatch_size=self._minibatch_size,
            max_minibatch_size=max_minibatch_size,
            obs_space=obs_space,
            atn_space=atn_space,
            device=self.device,
            hidden_size=hidden_size,
            cpu_offload=trainer_cfg.cpu_offload,
            num_lstm_layers=num_lstm_layers,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )

    def _make_vecenv(self):
        """Create a vectorized environment."""
        trainer_cfg = self.trainer_cfg

        num_agents = self._curriculum.get_task().env_cfg().game.num_agents

        self.target_batch_size = trainer_cfg.forward_pass_minibatch_target_size // num_agents
        if self.target_batch_size < max(2, trainer_cfg.num_workers):  # pufferlib bug requires batch size >= 2
            self.target_batch_size = trainer_cfg.num_workers

        logger.info(
            f"target_batch_size: {self.target_batch_size} = "
            f"min ({trainer_cfg.forward_pass_minibatch_target_size} // {num_agents} , {trainer_cfg.num_workers})"
        )

        self.batch_size = (self.target_batch_size // trainer_cfg.num_workers) * trainer_cfg.num_workers
        logger.info(
            f"forward_pass_batch_size: {self.batch_size} = "
            f"({self.target_batch_size} // {trainer_cfg.num_workers}) * {trainer_cfg.num_workers}"
        )

        num_envs = self.batch_size * trainer_cfg.async_factor
        logger.info(f"num_envs: {num_envs}")

        if num_envs < 1:
            logger.error(
                f"num_envs = batch_size ({self.batch_size}) * async_factor ({trainer_cfg.async_factor}) "
                f"is {num_envs}, which is less than 1! (Increase trainer.forward_pass_minibatch_target_size)"
            )

        self.vecenv = make_vecenv(
            self._curriculum,
            self.cfg.vectorization,
            num_envs=num_envs,
            batch_size=self.batch_size,
            num_workers=trainer_cfg.num_workers,
            zero_copy=trainer_cfg.zero_copy,
            is_training=True,
        )

        self._memory_monitor.add(self.vecenv)

        if self.cfg.seed is None:
            self.cfg.seed = np.random.randint(0, 1000000)

        # Use rank-specific seed for environment reset to ensure different
        # processes generate uncorrelated environments in distributed training
        rank = int(os.environ.get("RANK", 0))
        self.vecenv.async_reset(self.cfg.seed + rank)

    def _load_policy(self, checkpoint: TrainerCheckpoint | None, policy_store) -> PolicyRecord | None:
        """Try to load policy from checkpoint or config. Returns None if not found."""
        trainer_cfg = self.trainer_cfg

        # Try checkpoint first
        if checkpoint and checkpoint.policy_path:
            logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            return policy_store.policy(checkpoint.policy_path)

        # Try initial_policy from config
        if hasattr(trainer_cfg, "initial_policy") and getattr(trainer_cfg.initial_policy, "uri", None) is not None:
            initial_uri = trainer_cfg.initial_policy.uri
            logger.info(f"Loading initial policy URI: {initial_uri}")
            return policy_store.policy(initial_uri)

        # Try default checkpoint path
        policy_path = os.path.join(trainer_cfg.checkpoint_dir, policy_store.make_model_name(0))
        if os.path.exists(policy_path):
            logger.info(f"Loading policy from checkpoint: {policy_path}")
            return policy_store.policy(policy_path)

        return None

    def _create_policy(self, policy_store, metta_grid_env) -> PolicyRecord:
        """Create a new policy. Only master should call this."""
        if not self._master:
            raise RuntimeError("Only master process should create policies")

        logger.info("Creating new policy")
        return policy_store.create(metta_grid_env)

    def _wait_for_policy(self, policy_store, timeout_attempts: int = 10) -> PolicyRecord:
        """Non-master processes wait for master to create policy."""
        if self._master:
            raise RuntimeError("Master process should not wait for policy")

        policy_path = os.path.join(self.trainer_cfg.checkpoint_dir, policy_store.make_model_name(0))

        for attempt in range(timeout_attempts):
            if os.path.exists(policy_path):
                logger.info(f"Found policy created by master: {policy_path}")
                return policy_store.policy(policy_path)
            logger.info(f"Waiting for master to create policy... attempt {attempt + 1}/{timeout_attempts}")
            time.sleep(5)

        raise RuntimeError(f"Timeout: Master failed to create policy at {policy_path}")

    def _maybe_compute_grad_stats(self, force=False):
        """Compute and store gradient statistics if on interval."""
        interval = self.trainer_cfg.get("grad_mean_variance_interval", 0)
        if not self._should_run(interval, force):
            return

        with self.timer("grad_stats"):
            all_gradients = []
            for param in self.policy.parameters():
                if param.grad is not None:
                    all_gradients.append(param.grad.view(-1))

            if not all_gradients:
                logger.warning("No gradients found to compute stats.")
                self.grad_stats = {}
                return

            all_gradients_tensor = torch.cat(all_gradients).to(torch.float32)

            grad_mean = all_gradients_tensor.mean()
            grad_variance = all_gradients_tensor.var()
            grad_norm = all_gradients_tensor.norm(2)

            self.grad_stats = {
                "grad/mean": grad_mean.item(),
                "grad/variance": grad_variance.item(),
                "grad/norm": grad_norm.item(),
            }
            logger.info(
                f"Computed gradient stats at epoch {self.epoch}: "
                f"mean={self.grad_stats['grad/mean']:.2e}, "
                f"var={self.grad_stats['grad/variance']:.2e}, "
                f"norm={self.grad_stats['grad/norm']:.2e}"
            )


class AbortingTrainer(MettaTrainer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _on_train_step(self):
        if self.wandb_run is None:
            return

        if "abort" not in wandb.Api().run(self.wandb_run.path).tags:
            return

        logger.info("Abort tag detected. Stopping the run.")
        self.trainer_cfg.total_timesteps = int(self.agent_step)
        self.wandb_run.config.update(
            {"trainer.total_timesteps": self.trainer_cfg.total_timesteps}, allow_val_change=True
        )
