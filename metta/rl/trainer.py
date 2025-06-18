import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Set
from uuid import UUID

import einops
import numpy as np
import torch
import torch.distributed
import wandb
from heavyball import ForeachMuon
from omegaconf import DictConfig, ListConfig
from pufferlib import unroll_nested_dict

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent
from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.agent.util.debug import assert_shape
from metta.app.stats_client import StatsClient
from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.policy import PytorchAgent
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.util.wandb.wandb_context import WandbRun
from mettagrid.curriculum import curriculum_from_config_path
from mettagrid.mettagrid_env import MettaGridEnv, dtype_actions
from mettagrid.util.stopwatch import Stopwatch, with_instance_timer

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

        self.sim_suite_config = sim_suite_config
        self._stats_client = stats_client

        self._master = True
        self._world_size = 1
        self.device: torch.device = cfg.device
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
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.mean_reward = 0.0
        self.filtered_mean_reward = 0.0  # IIR filtered value used by self.trainer_cfg.average_reward

        self._current_eval_score: float | None = None
        self._eval_grouped_scores: Dict[str, float] = {}
        self._eval_suite_avgs: Dict[str, float] = {}
        self._eval_categories: Set[str] = set()

        self.timer = Stopwatch(logger)
        self.timer.start()

        curriculum_config = trainer_cfg.get("curriculum", trainer_cfg.get("env", {}))
        env_overrides = DictConfig({"env_overrides": trainer_cfg.env_overrides})
        self._curriculum = curriculum_from_config_path(curriculum_config, env_overrides)

        self._make_vecenv()

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), (
            f"vecenv.driver_env type {type(metta_grid_env).__name__} is not MettaGridEnv"
        )

        logger.info("Loading checkpoint")
        os.makedirs(trainer_cfg.checkpoint_dir, exist_ok=True)

        checkpoint = TrainerCheckpoint.load(cfg.run_dir)
        policy_record = self._load_policy(checkpoint, policy_store, metta_grid_env)

        assert policy_record is not None, "No policy found"

        if self._master:
            logger.info(f"MettaTrainer loaded: {policy_record.policy()}")

        self._initial_pr = policy_record
        self.last_pr = policy_record
        self.policy = policy_record.policy().to(self.device)
        self.policy_record = policy_record
        self.uncompiled_policy = self.policy

        # Note that these fields are specific to MettaGridEnv, which is why we can't keep
        # self.vecenv.driver_env as just the parent class pufferlib.PufferEnv
        actions_names = metta_grid_env.action_names
        actions_max_params = metta_grid_env.max_action_args

        self.policy.activate_actions(actions_names, actions_max_params, self.device)

        if trainer_cfg.compile:
            logger.info("Compiling policy")
            self.policy = torch.compile(self.policy, mode=trainer_cfg.compile_mode)

        self.kickstarter = Kickstarter(cfg, policy_store, actions_names, actions_max_params)

        if torch.distributed.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
            self._original_policy = self.policy
            self.policy = DistributedMettaAgent(self.policy, self.device)

        self._make_experience_buffer()

        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch

        self._stats_epoch_start = self.epoch
        self._stats_epoch_id: UUID | None = None
        self._stats_run_id: UUID | None = None

        # Optimizer
        assert trainer_cfg.optimizer.type in ("adam", "muon"), (
            f"Optimizer type must be 'adam' or 'muon', got {trainer_cfg.optimizer.type}"
        )
        opt_cls = torch.optim.Adam if trainer_cfg.optimizer.type == "adam" else ForeachMuon
        self.optimizer = opt_cls(
            self.policy.parameters(),
            lr=trainer_cfg.optimizer.learning_rate,
            betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
            eps=trainer_cfg.optimizer.eps,
            weight_decay=trainer_cfg.optimizer.weight_decay,
        )

        # validate that policy matches environment
        self.metta_agent: MettaAgent | DistributedMettaAgent = self.policy  # type: ignore
        assert isinstance(self.metta_agent, (MettaAgent, DistributedMettaAgent, PytorchAgent)), self.metta_agent
        _env_shape = metta_grid_env.single_observation_space.shape
        environment_shape = tuple(_env_shape) if isinstance(_env_shape, list) else _env_shape

        if isinstance(self.metta_agent, (MettaAgent, DistributedMettaAgent)):
            found_match = False
            for component_name, component in self.metta_agent.components.items():
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
        if trainer_cfg.lr_scheduler.enabled:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=trainer_cfg.total_timesteps // trainer_cfg.batch_size
            )

        if checkpoint.agent_step > 0:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        if wandb_run and self._master:
            # Define metrics (wandb x-axis values)
            metrics = ["step", "epoch", "total_time", "train_time"]
            for metric in metrics:
                wandb_run.define_metric(f"metric/{metric}")

            # set the default x-axis to be step count
            for k in ["overview", "env", "losses", "performance"]:
                wandb_run.define_metric(f"{k}/*", step_metric="metric/step")

            # set up plots that do not use steps as the x-axis
            metric_overrides = [
                ("overview/reward_vs_total_time", "metric/total_time"),
                ("overview/reward_vs_train_time", "metric/train_time"),
                ("overview/reward_vs_epoch", "metric/epoch"),
            ]

            for metric_name, step_metric in metric_overrides:
                wandb_run.define_metric(metric_name, step_metric=step_metric)

        logger.info(f"MettaTrainer initialization complete on device: {self.device}")

    def train(self) -> None:
        logger.info("Starting training")
        trainer_cfg = self.trainer_cfg

        # it doesn't make sense to evaluate more often than checkpointing since we need a saved policy to evaluate
        if trainer_cfg.evaluate_interval != 0 and trainer_cfg.evaluate_interval < trainer_cfg.checkpoint_interval:
            raise ValueError("evaluate_interval must be at least as large as checkpoint_interval")

        if self._stats_client is not None:
            name = self.wandb_run.name if self.wandb_run is not None and self.wandb_run.name is not None else "unknown"
            url = self.wandb_run.url if self.wandb_run is not None else None
            self._stats_run_id = self._stats_client.create_training_run(name=name, attributes={}, url=url).id

        logger.info(f"Training on {self.device}")
        while self.agent_step < trainer_cfg.total_timesteps:
            steps_before = self.agent_step

            with self.torch_profiler:
                self._rollout()
                self._train()

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
                f"{steps_per_sec:.0f} steps/sec "
                f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
            )

            # Checkpointing trainer
            if self.epoch % trainer_cfg.checkpoint_interval == 0:
                self._checkpoint_trainer()

            if trainer_cfg.evaluate_interval != 0 and self.epoch % trainer_cfg.evaluate_interval == 0:
                self._evaluate_policy()

            self.torch_profiler.on_epoch_end(self.epoch)

            if self.epoch % trainer_cfg.wandb_checkpoint_interval == 0:
                self._save_policy_to_wandb()

            if (
                self.cfg.agent.l2_init_weight_update_interval != 0
                and self.epoch % self.cfg.agent.l2_init_weight_update_interval == 0
            ):
                self.policy.update_l2_init_weight_copy()

            if trainer_cfg.replay_interval != 0 and self.epoch % trainer_cfg.replay_interval == 0:
                self._generate_and_upload_replay()

            self._on_train_step()

        timing_summary = self.timer.get_all_summaries()
        logger.info("Training complete!")
        for name, summary in timing_summary.items():
            logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        self._checkpoint_trainer()
        self._save_policy_to_wandb()

    @with_instance_timer("_evaluate_policy", log_level=logging.INFO)
    def _evaluate_policy(self):
        if not self._master:
            return

        if self._stats_run_id is not None and self._stats_client is not None:
            self._stats_epoch_id = self._stats_client.create_epoch(
                run_id=self._stats_run_id,
                start_training_epoch=self._stats_epoch_start,
                end_training_epoch=self.epoch,
                attributes={},
            ).id
            self._stats_epoch_start = self.epoch + 1

        logger.info(f"Simulating policy: {self.last_pr.uri} with config: {self.sim_suite_config}")
        sim = SimulationSuite(
            config=self.sim_suite_config,
            policy_pr=self.last_pr,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.cfg.vectorization,
            stats_dir="/tmp/stats",
            stats_client=self._stats_client,
            stats_epoch_id=self._stats_epoch_id,
        )
        result = sim.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)

        logger.info("Simulation complete")

        self._eval_categories: Set[str] = set()
        for sim_name in self.sim_suite_config.simulations.keys():
            self._eval_categories.add(sim_name.split("/")[0])
        self._eval_suite_avgs = {}

        # Compute scores for each evaluation category
        for category in self._eval_categories:
            score = stats_db.get_average_metric_by_filter("reward", self.last_pr, f"sim_name LIKE '%{category}%'")
            logger.info(f"{category} score: {score}")
            # Only add the score if we got a non-None result
            if score is not None:
                self._eval_suite_avgs[f"{category}_score"] = score
            else:
                self._eval_suite_avgs[f"{category}_score"] = 0.0

        # Get overall score (average of all rewards)
        overall_score = stats_db.get_average_metric_by_filter("reward", self.last_pr)
        self._current_eval_score = overall_score if overall_score is not None else 0.0
        all_scores = stats_db.simulation_scores(self.last_pr, "reward")

        # Categorize scores by environment type
        self._eval_grouped_scores = {}
        # Process each score and assign to the right category
        for (_, sim_name, _), score in all_scores.items():
            for category in self._eval_categories:
                if category in sim_name.lower():
                    self._eval_grouped_scores[f"{category}/{sim_name.split('/')[-1]}"] = score

    def _on_train_step(self):
        pass

    @with_instance_timer("_rollout")
    def _rollout(self):
        experience = self.experience
        trainer_cfg = self.trainer_cfg

        policy = self.policy
        infos = defaultdict(list)
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

            num_steps = sum(mask)
            self.agent_step += num_steps * self._world_size

            o = torch.as_tensor(o)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)
            t = torch.as_tensor(t)

            with torch.no_grad():
                state = PolicyState()

                lstm_state = experience.get_lstm_state(training_env_id.start)
                if lstm_state is not None:
                    state.lstm_h = lstm_state["lstm_h"]
                    state.lstm_c = lstm_state["lstm_c"]

                o_device = o.to(self.device, non_blocking=True)
                actions, selected_action_log_probs, _, value, _ = policy(o_device, state)

                if __debug__:
                    assert_shape(selected_action_log_probs, ("BT",), "selected_action_log_probs")
                    assert_shape(actions, ("BT", 2), "actions")

                lstm_state_to_store = None
                if trainer_cfg.get("use_rnn", True) and state.lstm_h is not None:
                    lstm_state_to_store = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c}

                if self.device == "cuda":
                    torch.cuda.synchronize()

            value = value.flatten()
            mask = torch.as_tensor(mask)  # * policy.mask)

            experience.store(
                obs=o if trainer_cfg.cpu_offload else o_device,
                actions=actions,
                logprobs=selected_action_log_probs,
                rewards=r.to(self.device, non_blocking=True),
                dones=d.to(self.device, non_blocking=True),
                truncations=t.to(self.device, non_blocking=True),
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

            for i in info:
                for k, v in unroll_nested_dict(i):
                    infos[k].append(v)

            with self.timer("_rollout.env"):
                actions_np = actions.cpu().numpy().astype(dtype_actions)
                self.vecenv.send(actions_np)

        for k, v in infos.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()

            if isinstance(v, list):
                if k not in self.stats:
                    self.stats[k] = []
                self.stats[k].extend(v)
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

    def _get_experience_buffer_mean_reward(self) -> float:
        # Use rewards from experience buffer
        if hasattr(self, "experience") and self.experience.rewards_np is not None:
            return float(np.mean(self.experience.rewards_np))

        return 0.0

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

        self.mean_reward = self._get_experience_buffer_mean_reward()

        if self.trainer_cfg.average_reward:
            # Average reward formulation: A_t = GAE(r_t - ρ, γ=1.0)
            # where ρ is the average reward estimate

            # Apply IIR filter (exponential moving average)
            alpha = trainer_cfg.average_reward_alpha
            self.filtered_mean_reward = (1 - alpha) * self.filtered_mean_reward + alpha * self.mean_reward

            # Use filtered estimate for advantage computation
            rewards_np_adjusted = (rewards_np - self.filtered_mean_reward).astype(np.float32)
            effective_gamma = 1.0
            advantages_np = compute_gae(
                dones_np, values_np, rewards_np_adjusted, effective_gamma, trainer_cfg.gae_lambda
            )
        else:
            # Standard discounted formulation: A_t = GAE(r_t, γ<1.0)
            effective_gamma = trainer_cfg.gamma
            advantages_np = compute_gae(dones_np, values_np, rewards_np, effective_gamma, trainer_cfg.gae_lambda)

        experience.returns_np = advantages_np + values_np
        experience.flatten_batch(advantages_np)

        # Optimizing the policy and value network
        for _epoch in range(trainer_cfg.update_epochs):
            lstm_state = PolicyState()
            teacher_lstm_state = []
            for mb in range(experience.num_minibatches):
                obs = experience.b_obs[mb]
                obs = obs.to(self.device, non_blocking=True)
                atn = experience.b_actions[mb]
                old_action_log_probs = experience.b_logprobs[mb]
                val = experience.b_values[mb]
                adv = experience.b_advantages[mb]
                ret = experience.b_returns[mb]

                # Forward pass returns: (action, new_action_log_probs, entropy, value, full_log_probs_distribution)
                _, new_action_log_probs, entropy, newvalue, full_log_probs_distribution = self.policy(
                    obs, lstm_state, action=atn
                )
                if self.device == "cuda":
                    torch.cuda.synchronize()

                if __debug__:
                    assert_shape(new_action_log_probs, ("BT",), "new_action_log_probs")
                    assert_shape(old_action_log_probs, ("B", "T"), "old_action_log_probs")

                logratio = new_action_log_probs - old_action_log_probs.reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > trainer_cfg.clip_coef).float().mean()

                adv = self._compute_advantage(adv)

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
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                entropy_loss = entropy.mean()

                ks_action_loss, ks_value_loss = self.kickstarter.loss(
                    self.agent_step, full_log_probs_distribution, newvalue, obs, teacher_lstm_state
                )

                l2_reg_loss = torch.tensor(0.0, device=self.device)
                if trainer_cfg.l2_reg_loss_coef > 0:
                    l2_reg_loss = trainer_cfg.l2_reg_loss_coef * self.policy.l2_reg_loss().to(self.device)

                l2_init_loss = torch.tensor(0.0, device=self.device)
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

                experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))

                if self.losses is None:
                    raise ValueError("self.losses is None")

                # Update loss tracking for logging
                self.losses.policy_loss += pg_loss.item()
                self.losses.value_loss += v_loss.item()
                self.losses.entropy += entropy_loss.item()
                self.losses.old_approx_kl += old_approx_kl.item()
                self.losses.approx_kl += approx_kl.item()
                self.losses.clipfrac += clipfrac.item()
                self.losses.l2_reg_loss += l2_reg_loss.item() if torch.is_tensor(l2_reg_loss) else l2_reg_loss
                self.losses.l2_init_loss += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
                self.losses.ks_action_loss += ks_action_loss.item()
                self.losses.ks_value_loss += ks_value_loss.item()
                self.losses.importance += importance_sampling_ratio.mean().item()
                self.losses.minibatches_processed += 1

                self.optimizer.zero_grad()
                loss.backward()
                if (mb + 1) % self.experience.accumulate_minibatches == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), trainer_cfg.max_grad_norm)
                    self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), trainer_cfg.max_grad_norm)
                self.optimizer.step()

                if self.cfg.agent.clip_range > 0:
                    self.policy.clip_weights()

                # end loop over minibatches

            # Calculate explained variance
            y_pred = experience.values.flatten()
            y_true = advantages.flatten() + experience.values.flatten()
            var_y = y_true.var()
            explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
            self.losses.explained_variance = explained_var.item() if torch.is_tensor(explained_var) else float("nan")

            # check early exit if we have reached target_kl
            if trainer_cfg.target_kl is not None:
                average_approx_kl = self.losses.approx_kl_sum / self.losses.minibatches_processed
                if average_approx_kl > trainer_cfg.target_kl:
                    break

            self.epoch += 1
            # end loop over epochs

    def _checkpoint_trainer(self):
        if not self._master:
            return

        self._checkpoint_policy()

        extra_args = {}
        if self.trainer_cfg.average_reward:
            extra_args["filtered_mean_reward"] = self.filtered_mean_reward

        checkpoint = TrainerCheckpoint(
            agent_step=self.agent_step,
            epoch=self.epoch,
            total_agent_step=self.agent_step * torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else self.agent_step,
            optimizer_state_dict=self.optimizer.state_dict(),
            extra_args=extra_args,
        )
        checkpoint.save(self.cfg.run_dir)

    def _checkpoint_policy(self) -> PolicyRecord | None:
        if not self._master:
            return

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), "vecenv.driver_env must be a MettaGridEnv for checkpointing"

        name = self.policy_store.make_model_name(self.epoch)

        generation = 0
        if self._initial_pr:
            generation = self._initial_pr.metadata.get("generation", 0) + 1

        training_time = self.timer.get_elapsed("_rollout") + self.timer.get_elapsed("_train")

        self.last_pr = self.policy_store.save(
            name,
            os.path.join(self.trainer_cfg.checkpoint_dir, name),
            self.uncompiled_policy,
            metadata={
                "agent_step": self.agent_step,
                "epoch": self.epoch,
                "run": self.cfg.run,
                "action_names": metta_grid_env.action_names,
                "generation": generation,
                "initial_uri": self._initial_pr.uri,
                "train_time": training_time,
                "score": self._current_eval_score,
                "eval_scores": self._eval_suite_avgs,
            },
        )
        # this is hacky, but otherwise the initial_pr points
        # at the same policy as the last_pr
        return self.last_pr

    def _save_policy_to_wandb(self):
        if not self._master:
            return

        if self.wandb_run is None:
            return

        pr = self._checkpoint_policy()
        self.policy_store.add_to_wandb_run(self.wandb_run.name, pr)

    @with_instance_timer("_generate_and_upload_replay", log_level=logging.INFO)
    def _generate_and_upload_replay(self):
        if not self._master:
            return

        replay_sim_config = SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",
            num_episodes=1,
            env_overrides=self._curriculum.get_task().env_cfg(),
        )

        replay_simulator = Simulation(
            name=f"replay_{self.epoch}",
            config=replay_sim_config,
            policy_pr=self.last_pr,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.cfg.vectorization,
            replay_dir=self.cfg.trainer.replay_dir,
        )
        results = replay_simulator.simulate()

        if self.wandb_run is not None:
            replay_urls = results.stats_db.get_replay_urls(
                policy_key=self.last_pr.key(), policy_version=self.last_pr.version()
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
        if not self.wandb_run or not self._master:
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

        # X-axis values for wandb
        metric_stats = {
            "metric/step": self.agent_step,
            "metric/epoch": self.epoch,
            "metric/total_time": wall_time,
            "metric/train_time": train_time,
        }
        lap_times = self.timer.lap_all(self.agent_step)
        wall_time_for_lap = lap_times.pop("global", 0)

        timing_stats = {
            **{
                f"timing_per_epoch/fraction/{op}": lap_elapsed / wall_time_for_lap if wall_time_for_lap > 0 else 0
                for op, lap_elapsed in lap_times.items()
            },
            **{
                f"timing_cumulative/fraction/{op}": elapsed / wall_time if wall_time > 0 else 0
                for op, elapsed in elapsed_times.items()
            },
        }

        delta_steps = self.timer.get_lap_steps()
        if delta_steps is None:
            delta_steps = self.agent_step
        lap_steps_per_second = delta_steps / wall_time_for_lap if wall_time_for_lap > 0 else 0
        steps_per_second = self.timer.get_rate(self.agent_step) if wall_time > 0 else 0

        overview = {
            "sps": steps_per_second,
            "lap_sps": lap_steps_per_second,
            "reward": self.mean_reward,
            "reward_vs_total_time": self.mean_reward,
            "reward_vs_train_time": self.mean_reward,
            "reward_vs_epoch": self.mean_reward,
        }

        for k, v in self.trainer_cfg.stats.overview.items():
            if k in self.stats:
                overview[v] = self.stats[k]

        for category in self._eval_categories:
            score = self._eval_suite_avgs.get(f"{category}_score", None)
            if score is not None:
                overview[f"{category}_evals"] = score

        # Add filtered average reward if applicable
        if self.trainer_cfg.average_reward:
            overview["filtered_mean_reward"] = self.filtered_mean_reward

        losses = self.losses.to_dict()

        # don't plot losses that are unused
        if self.trainer_cfg.l2_reg_loss_coef == 0:
            losses.pop("l2_reg_loss")
        if self.trainer_cfg.l2_init_loss_coef == 0:
            losses.pop("l2_init_loss")
        if not self.kickstarter.enabled:
            losses.pop("ks_action_loss")
            losses.pop("ks_value_loss")

        environment_stats = {f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in self.stats.items()}

        parameter_stats = {
            "parameter/learning_rate": self.optimizer.param_groups[0]["lr"],
            "parameter/delta_steps": delta_steps,
            "parameter/num_minibatches": self.experience.num_minibatches,
        }

        # Log everything to wandb
        self.wandb_run.log(
            {
                **{f"overview/{k}": v for k, v in overview.items()},
                **{f"losses/{k}": v for k, v in losses.items()},
                **environment_stats,
                **weight_stats,
                **self._eval_grouped_scores,
                **parameter_stats,
                **timing_stats,
                **metric_stats,
            }
        )

        self._eval_grouped_scores = {}
        self.stats.clear()

    @with_instance_timer("_compute_advantage")
    def _compute_advantage(self, adv: torch.Tensor) -> torch.Tensor:
        """Compute normalized advantages, handling distributed training synchronization."""
        adv = adv.reshape(-1)
        if self.trainer_cfg.norm_adv:
            if torch.distributed.is_initialized():
                local_sum = einops.rearrange(adv.sum(), "-> 1")
                local_sq_sum = einops.rearrange((adv * adv).sum(), "-> 1")
                local_count = torch.tensor([adv.numel()], dtype=adv.dtype, device=adv.device)

        # Get correct device for this process
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        if torch.distributed.is_initialized() and str(device).startswith("cuda"):
            device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

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

    def initial_pr_uri(self) -> str:
        return self._initial_pr.uri

    def last_pr_uri(self) -> str:
        return self.last_pr.uri

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

        # Get LSTM parameters if using RNN
        use_rnn = trainer_cfg.get("use_rnn", True)
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
            use_rnn=use_rnn,
            hidden_size=hidden_size,
            cpu_offload=trainer_cfg.cpu_offload,
            num_lstm_layers=num_lstm_layers,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )

    @with_instance_timer("_make_vecenv")
    def _make_vecenv(self):
        """Create a vectorized environment."""
        trainer_cfg = self.trainer_cfg

        num_agents = self._curriculum.get_task().env_cfg().game.num_agents

        self.target_batch_size = trainer_cfg.forward_pass_minibatch_target_size // num_agents
        if self.target_batch_size < 2:  # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2

        self.batch_size = (self.target_batch_size // trainer_cfg.num_workers) * trainer_cfg.num_workers
        logger.info(f"forward_pass_batch_size: {self.batch_size}")

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
        )

        if self.cfg.seed is None:
            self.cfg.seed = np.random.randint(0, 1000000)

        # Use rank-specific seed for environment reset to ensure different
        # processes generate uncorrelated environments in distributed training
        rank = int(os.environ.get("RANK", 0))
        self.vecenv.async_reset(self.cfg.seed + rank)

    def _load_policy(self, checkpoint, policy_store, metta_grid_env):
        """Load policy from checkpoint, initial_policy.uri, or create new."""
        trainer_cfg = self.trainer_cfg

        for _attempt in range(10):
            if checkpoint.policy_path:
                logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
                return policy_store.policy(checkpoint.policy_path)
            elif trainer_cfg.initial_policy.uri is not None:
                logger.info(f"Loading initial policy URI: {trainer_cfg.initial_policy.uri}")
                return policy_store.policy(trainer_cfg.initial_policy)
            else:
                policy_path = os.path.join(trainer_cfg.checkpoint_dir, policy_store.make_model_name(0))
                if os.path.exists(policy_path):
                    logger.info(f"Loading policy from checkpoint: {policy_path}")
                    return policy_store.policy(policy_path)
                elif self._master:
                    logger.info(f"Failed to load policy from default checkpoint: {policy_path}. Creating a new policy!")
                    return policy_store.create(metta_grid_env)
            time.sleep(5)

        raise RuntimeError("Failed to load policy after 10 attempts")


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
