import logging
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed
import wandb
from heavyball import ForeachMuon
from omegaconf import DictConfig, ListConfig
from pufferlib import unroll_nested_dict

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent
from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyStore
from metta.agent.util.debug import assert_shape
from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.pufferlib.experience import Experience
from metta.rl.pufferlib.kickstarter import Kickstarter
from metta.rl.pufferlib.policy import PufferAgent
from metta.rl.pufferlib.profile import Profile, profile_section
from metta.rl.pufferlib.torch_profiler import TorchProfiler
from metta.rl.pufferlib.trainer_checkpoint import TrainerCheckpoint
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sim.vecenv import make_vecenv
from metta.util.timing import Stopwatch
from mettagrid.curriculum import curriculum_from_config_path
from mettagrid.mettagrid_env import MettaGridEnv, dtype_actions

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


class PufferTrainer:
    def __init__(
        self,
        cfg: DictConfig | ListConfig,
        wandb_run,
        policy_store: PolicyStore,
        sim_suite_config: SimulationSuiteConfig,
        **kwargs,
    ):
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer

        self.sim_suite_config = sim_suite_config

        self._master = True
        self._world_size = 1
        self.device: torch.device = cfg.device
        self._batch_size = self.trainer_cfg.batch_size
        self._minibatch_size = self.trainer_cfg.minibatch_size
        if torch.distributed.is_initialized():
            self._master = int(os.environ["RANK"]) == 0
            self._world_size = torch.distributed.get_world_size()

            self._batch_size = self.trainer_cfg.batch_size // self._world_size
            self._minibatch_size = self.trainer_cfg.minibatch_size // self._world_size

            logger.info(
                f"Rank: {os.environ['RANK']}, Local rank: {os.environ['LOCAL_RANK']}, World size: {self._world_size}"
            )

        self.profile = Profile()
        self.torch_profiler = TorchProfiler(self._master, cfg.run_dir, cfg.trainer.profiler_interval_epochs, wandb_run)
        self.losses = self._make_losses()
        self.stats = defaultdict(list)
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.average_reward = 0.0
        self._current_eval_score = None
        self._eval_grouped_scores = {}
        self._eval_suite_avgs = {}
        self._eval_categories = set()

        curriculum_config = self.trainer_cfg.get("curriculum", self.trainer_cfg.get("env", {}))
        env_overrides = DictConfig({"env_overrides": self.trainer_cfg.env_overrides})
        self._curriculum = curriculum_from_config_path(curriculum_config, env_overrides)
        self._make_vecenv()

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), (
            f"vecenv.driver_env type {type(metta_grid_env).__name__} is not MettaGridEnv"
        )

        logger.info("Loading checkpoint")
        os.makedirs(cfg.trainer.checkpoint_dir, exist_ok=True)

        checkpoint = TrainerCheckpoint.load(cfg.run_dir)
        policy_record = self._load_policy(checkpoint, policy_store, metta_grid_env)

        assert policy_record is not None, "No policy found"
        if "average_reward" in checkpoint.extra_args:
            self.average_reward = checkpoint.extra_args["average_reward"]

        if self._master:
            logger.info(f"PufferTrainer loaded: {policy_record.policy()}")

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

        if self.trainer_cfg.compile:
            logger.info("Compiling policy")
            self.policy = torch.compile(self.policy, mode=self.trainer_cfg.compile_mode)

        self.kickstarter = Kickstarter(cfg, policy_store, actions_names, actions_max_params)

        if torch.distributed.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
            self._original_policy = self.policy
            self.policy = DistributedMettaAgent(self.policy, self.device)

        self._make_experience_buffer()

        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch
        self.profile.start_agent_steps = self.agent_step
        self._last_agent_step = self.agent_step
        self._total_minibatches = 0

        # Optimizer
        assert self.trainer_cfg.optimizer.type in ("adam", "muon"), (
            f"Optimizer type must be 'adam' or 'muon', got {self.trainer_cfg.optimizer.type}"
        )
        opt_cls = torch.optim.Adam if self.trainer_cfg.optimizer.type == "adam" else ForeachMuon
        self.optimizer = opt_cls(
            self.policy.parameters(),
            lr=self.trainer_cfg.optimizer.learning_rate,
            betas=(self.trainer_cfg.optimizer.beta1, self.trainer_cfg.optimizer.beta2),
            eps=self.trainer_cfg.optimizer.eps,
            weight_decay=self.trainer_cfg.optimizer.weight_decay,
        )

        # validate that policy matches environment
        self.metta_agent: MettaAgent | DistributedMettaAgent = self.policy  # type: ignore
        assert isinstance(self.metta_agent, (MettaAgent, DistributedMettaAgent, PufferAgent)), self.metta_agent
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
        if self.trainer_cfg.lr_scheduler.enabled:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.trainer_cfg.total_timesteps // self.trainer_cfg.batch_size
            )

        if checkpoint.agent_step > 0:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        if wandb_run and self._master:
            wandb_run.define_metric("train/agent_step")
            wandb_run.define_metric("train/avg_agent_steps_per_update", step_metric="train/agent_step")
            for k in ["0verview", "env", "losses", "performance", "train"]:
                wandb_run.define_metric(f"{k}/*", step_metric="train/agent_step")

        self.replay_sim_config = SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",
            num_episodes=1,
            env_overrides=self._curriculum.get_task().env_cfg(),
        )

        self.timer = Stopwatch(logger)
        self.timer.start()

        logger.info(f"PufferTrainer initialization complete on device: {self.device}")

    def train(self):
        logger.info("Starting training")

        # it doesn't make sense to evaluate more often than checkpointing since we need a saved policy to evaluate
        if (
            self.trainer_cfg.evaluate_interval != 0
            and self.trainer_cfg.evaluate_interval < self.trainer_cfg.checkpoint_interval
        ):
            raise ValueError("evaluate_interval must be at least as large as checkpoint_interval")

        logger.info(f"Training on {self.device}")
        while self.agent_step < self.trainer_cfg.total_timesteps:
            steps_before = self.agent_step

            with self.torch_profiler:
                with self.timer("_rollout"):
                    self._rollout()

                with self.timer("_train"):
                    self._train()

            # Processing stats
            with self.timer("_process_stats"):
                self._process_stats()

            rollout_time = self.timer.get_last_elapsed("_rollout")
            train_time = self.timer.get_last_elapsed("_train")
            stats_time = self.timer.get_last_elapsed("_process_stats")
            steps_calculated = self.agent_step - steps_before
            steps_per_sec = steps_calculated / (train_time + rollout_time)

            logger.info(
                f"Epoch {self.epoch} - "
                f"rollout: {rollout_time:.3f}s, "
                f"train: {train_time:.3f}s, "
                f"stats: {stats_time:.3f}s, "
                f"[{steps_per_sec:.0f} steps/sec]"
            )

            # Checkpointing trainer
            if self.epoch % self.trainer_cfg.checkpoint_interval == 0:
                with self.timer("_checkpoint_trainer", log=logging.INFO):
                    self._checkpoint_trainer()

            if self.trainer_cfg.evaluate_interval != 0 and self.epoch % self.trainer_cfg.evaluate_interval == 0:
                with self.timer("_evaluate_policy", log=logging.INFO):
                    self._evaluate_policy()

            self.torch_profiler.on_epoch_end(self.epoch)

            if self.epoch % self.trainer_cfg.wandb_checkpoint_interval == 0:
                with self.timer("_save_policy_to_wandb"):
                    self._save_policy_to_wandb()

            if (
                self.cfg.agent.l2_init_weight_update_interval != 0
                and self.epoch % self.cfg.agent.l2_init_weight_update_interval == 0
            ):
                self.policy.update_l2_init_weight_copy()

            if self.trainer_cfg.replay_interval != 0 and self.epoch % self.trainer_cfg.replay_interval == 0:
                with self.timer("_generate_and_upload_replay", log=logging.INFO):
                    self._generate_and_upload_replay()

            self._on_train_step()

        timing_summary = self.timer.get_all_summaries()
        logger.info("Training complete!")
        for name, summary in timing_summary.items():
            logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        self._checkpoint_trainer()
        self._save_policy_to_wandb()

    def _evaluate_policy(self):
        if not self._master:
            return

        logger.info(f"Simulating policy: {self.last_pr.uri} with config: {self.sim_suite_config}")
        sim = SimulationSuite(
            config=self.sim_suite_config,
            policy_pr=self.last_pr,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.cfg.vectorization,
            stats_dir="/tmp/stats",
        )
        result = sim.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)

        logger.info("Simulation complete")

        self._eval_categories = set()
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

    @profile_section("eval")
    def _rollout(self):
        experience, profile = self.experience, self.profile

        with profile.eval_misc:
            policy = self.policy
            infos = defaultdict(list)

            experience.reset_for_rollout()

        while not experience.ready_for_training:
            with profile.env:
                o, r, d, t, info, env_id, mask = self.vecenv.recv()
                if self.trainer_cfg.require_contiguous_env_ids:
                    raise ValueError(
                        "We are assuming contiguous eng id is always False. async_factor == num_workers = "
                        f"{self.trainer_cfg.async_factor} != {self.trainer_cfg.num_workers}"
                    )

                training_env_id = slice(env_id[0], env_id[-1] + 1)

            with profile.eval_misc:
                num_steps = sum(mask)
                self.agent_step += num_steps * self._world_size

                o = torch.as_tensor(o)
                r = torch.as_tensor(r)
                d = torch.as_tensor(d)
                t = torch.as_tensor(t)

            with profile.eval_forward, torch.no_grad():
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
                if self.trainer_cfg.get("use_rnn", True) and state.lstm_h is not None:
                    lstm_state_to_store = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c}

                if self.device == "cuda":
                    torch.cuda.synchronize()

            with profile.eval_misc:
                value = value.flatten()
                mask = torch.as_tensor(mask)  # * policy.mask)

                experience.store(
                    obs=o if self.trainer_cfg.cpu_offload else o_device,
                    actions=actions,
                    logprobs=selected_action_log_probs,
                    rewards=r.to(self.device, non_blocking=True),
                    terminals=d.to(self.device, non_blocking=True),
                    truncations=t.to(self.device, non_blocking=True),
                    values=value,
                    env_id=training_env_id,
                    mask=mask,
                    lstm_state=lstm_state_to_store,
                )

                for i in info:
                    for k, v in unroll_nested_dict(i):
                        infos[k].append(v)

            with profile.env:
                actions_np = actions.cpu().numpy().astype(dtype_actions)
                self.vecenv.send(actions_np)

        with profile.eval_misc:
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

    @profile_section("train")
    def _train(self):
        experience, profile = self.experience, self.profile
        self.losses = self._make_losses()
        self._total_minibatches = experience.num_minibatches * self.trainer_cfg.update_epochs
        steps_since_last = self.agent_step - self._last_agent_step
        self._agent_steps_per_update = steps_since_last / max(self._total_minibatches, 1)

        with profile.train_misc:
            config = self.trainer_cfg

            # Reset importance sampling ratios
            experience.reset_ratio()

            # Prioritized sampling parameters
            b0 = config.get("prio_beta0", 0.6)
            a = config.get("prio_alpha", 0.0)
            total_epochs = max(1, config.total_timesteps // config.batch_size)
            anneal_beta = b0 + (1 - b0) * a * self.epoch / total_epochs

            # Update average reward estimate if enabled
            if config.average_reward:
                alpha = config.average_reward_alpha
                current_batch_mean = experience.get_mean_reward()
                self.average_reward = (1 - alpha) * self.average_reward + alpha * current_batch_mean

            # Compute advantages using puff_advantage
            advantages = torch.zeros(experience.values.shape, device=self.device)

            # Adjust rewards for average reward if enabled
            rewards_for_advantage = experience.rewards
            if config.average_reward:
                rewards_for_advantage = experience.rewards - self.average_reward

            # Initial ratio is all ones
            initial_ratio = torch.ones_like(experience.values)

            advantages = self._compute_advantage(
                experience.values,
                rewards_for_advantage,
                experience.dones,
                initial_ratio,
                advantages,
                config.gamma if not config.average_reward else 1.0,
                config.gae_lambda,
                config.get("vtrace_rho_clip", 1.0),
                config.get("vtrace_c_clip", 1.0),
            )

        # Optimizing the policy and value network
        for mb in range(self._total_minibatches):
            with profile.train_misc:
                minibatch = experience.sample_minibatch(
                    advantages=advantages,
                    prio_alpha=a,
                    prio_beta=anneal_beta,
                    minibatch_idx=mb,
                    total_minibatches=self._total_minibatches,
                )

            with profile.train_forward:
                obs = minibatch["obs"]
                if not config.get("use_rnn", True):
                    obs = obs.reshape(-1, *self.vecenv.single_observation_space.shape)

                lstm_state = PolicyState()
                _, new_logprobs, entropy, newvalue, full_logprobs = self.policy(
                    obs, lstm_state, action=minibatch["actions"]
                )

            with profile.train_misc:
                new_logprobs = new_logprobs.reshape(minibatch["logprobs"].shape)
                logratio = new_logprobs - minibatch["logprobs"]
                ratio = logratio.exp()
                experience.update_ratio(minibatch["indices"], ratio)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

                # Re-compute advantages with new ratios (V-trace)
                rewards_adjusted = minibatch["rewards"] - (self.average_reward if config.average_reward else 0)
                adv = self._compute_advantage(
                    minibatch["values"],
                    rewards_adjusted,
                    minibatch["terminals"],
                    ratio,
                    minibatch["advantages"],
                    1.0 if config.average_reward else config.gamma,
                    config.gae_lambda,
                    config.get("vtrace_rho_clip", 1.0),
                    config.get("vtrace_c_clip", 1.0),
                )

                # Normalize advantages with prioritized weights
                if config.get("norm_adv", True):
                    adv = minibatch["prio_weights"] * (adv - adv.mean()) / (adv.std() + 1e-8)
                else:
                    adv = minibatch["prio_weights"] * adv

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
                    v_clipped = minibatch["values"] + torch.clamp(
                        newvalue_reshaped - minibatch["values"],
                        -config.get("vf_clip_coef", 0.1),
                        config.get("vf_clip_coef", 0.1),
                    )
                    v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

                entropy_loss = entropy.mean()

                ks_action_loss, ks_value_loss = self.kickstarter.loss(
                    self.agent_step, full_logprobs, newvalue, obs, teacher_lstm_state=[]
                )

                l2_reg_loss = torch.tensor(0.0, device=self.device)
                if self.trainer_cfg.l2_reg_loss_coef > 0:
                    l2_reg_loss = self.trainer_cfg.l2_reg_loss_coef * self.policy.l2_reg_loss().to(self.device)

                l2_init_loss = torch.tensor(0.0, device=self.device)
                if self.trainer_cfg.l2_init_loss_coef > 0:
                    l2_init_loss = self.trainer_cfg.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)

                loss = (
                    pg_loss
                    - config.ent_coef * entropy_loss
                    + v_loss * config.vf_coef
                    + l2_reg_loss
                    + l2_init_loss
                    + ks_action_loss
                    + ks_value_loss
                )

                experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))

                # Update loss tracking for logging
                self.losses.policy_loss += pg_loss.item() / self._total_minibatches
                self.losses.value_loss += v_loss.item() / self._total_minibatches
                self.losses.entropy += entropy_loss.item() / self._total_minibatches
                self.losses.old_approx_kl += old_approx_kl.item() / self._total_minibatches
                self.losses.approx_kl += approx_kl.item() / self._total_minibatches
                self.losses.clipfrac += clipfrac.item() / self._total_minibatches
                self.losses.l2_reg_loss += (
                    l2_reg_loss.item() if torch.is_tensor(l2_reg_loss) else l2_reg_loss
                ) / self._total_minibatches
                self.losses.l2_init_loss += (
                    l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
                ) / self._total_minibatches
                self.losses.ks_action_loss += ks_action_loss.item() / self._total_minibatches
                self.losses.ks_value_loss += ks_value_loss.item() / self._total_minibatches
                self.losses.importance += ratio.mean().item() / self._total_minibatches

            with profile.learn:
                self.optimizer.zero_grad()
                loss.backward()
                if (mb + 1) % self.experience.accumulate_minibatches == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                    self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                    if self.device == "cuda":
                        torch.cuda.synchronize()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        with profile.train_misc:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Calculate explained variance
            y_pred = experience.values.flatten()
            y_true = advantages.flatten() + experience.values.flatten()
            var_y = y_true.var()
            explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
            self.losses.explained_variance = explained_var.item() if torch.is_tensor(explained_var) else float("nan")

            self.epoch += 1
            profile.update_stats(
                self.agent_step,
                self.trainer_cfg.total_timesteps,
            )

    def _checkpoint_trainer(self):
        if not self._master:
            return

        pr = self._checkpoint_policy()

        # Save filtered average reward estimate for restart continuity
        extra_args = {}
        if self.trainer_cfg.average_reward:
            extra_args["average_reward"] = self.average_reward

        self.checkpoint = TrainerCheckpoint(
            self.agent_step, self.epoch, self.optimizer.state_dict(), pr.local_path(), **extra_args
        ).save(self.cfg.run_dir)

    def _checkpoint_policy(self):
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
            os.path.join(self.cfg.trainer.checkpoint_dir, name),
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

    def _generate_and_upload_replay(self):
        if self._master:
            logger.info("Generating and saving a replay to wandb and S3.")

            replay_simulator = Simulation(
                name=f"replay_{self.epoch}",
                config=self.replay_sim_config,
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

    def _process_stats(self):
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

        weight_metrics = {}
        if self.cfg.agent.analyze_weights_interval != 0 and self.epoch % self.cfg.agent.analyze_weights_interval == 0:
            for metrics in self.policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key != "name":
                        weight_metrics[f"weights/{key}/{name}"] = value

        # Calculate derived stats from local roll-outs (master process will handle logging)
        sps = self.profile.SPS
        agent_steps = self.agent_step
        avg_steps_per_update = 0.0
        if self._total_minibatches:
            avg_steps_per_update = (agent_steps - self._last_agent_step) / self._total_minibatches
            self._last_agent_step = agent_steps
        epoch = self.epoch
        learning_rate = self.optimizer.param_groups[0]["lr"]
        losses = {k: v for k, v in vars(self.losses).items() if not k.startswith("_")}
        performance = {k: v for k, v in self.profile}

        overview = {"SPS": sps}
        for k, v in self.trainer_cfg.stats.overview.items():
            if k in self.stats:
                overview[v] = self.stats[k]

        for category in self._eval_categories:
            score = self._eval_suite_avgs.get(f"{category}_score", None)
            if score is not None:
                overview[f"{category}_evals"] = score

        environment = {f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in self.stats.items()}

        # Add timing metrics to wandb
        if self.wandb_run and self._master:
            timer_data = {}
            wall_time = self.timer.get_elapsed()  # global timer
            timer_data = self.timer.get_all_elapsed()

            training_time = timer_data.get("_rollout", 0) + timer_data.get("_train", 0)
            overhead_time = wall_time - training_time
            steps_per_sec = (self.agent_step - self._last_agent_step) / training_time if training_time > 0 else 0

            timing_logs = {
                # Key performance indicators
                "timing/steps_per_second": steps_per_sec,
                "timing/training_efficiency": training_time / wall_time if wall_time > 0 else 0,
                "timing/overhead_ratio": overhead_time / wall_time if wall_time > 0 else 0,
                # Breakdown by operation (as a single structured metric)
                "timing/breakdown": {
                    op: {"seconds": elapsed, "fraction": elapsed / wall_time if wall_time > 0 else 0}
                    for op, elapsed in timer_data.items()
                },
                # Total time for reference
                "timing/total_seconds": wall_time,
            }

            # Log everything to wandb
            self.wandb_run.log(
                {
                    **{f"overview/{k}": v for k, v in overview.items()},
                    **{f"losses/{k}": v for k, v in losses.items()},
                    **{f"performance/{k}": v for k, v in performance.items()},
                    **environment,
                    **weight_metrics,
                    **self._eval_grouped_scores,
                    "train/agent_step": agent_steps,
                    "train/avg_agent_steps_per_update": avg_steps_per_update,
                    "train/epoch": epoch,
                    "train/learning_rate": learning_rate,
                    "train/average_reward": self.average_reward if self.trainer_cfg.average_reward else None,
                    **timing_logs,
                }
            )

        self._eval_grouped_scores = {}
        self.stats.clear()

    def _compute_advantage(
        self, values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
    ):
        """CUDA kernel for puffer advantage with automatic CPU fallback."""
        try:
            torch.ops.pufferlib.compute_puff_advantage(
                values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
            )
        except (RuntimeError, AssertionError):
            # Fallback to CPU if CUDA kernel fails or not available
            device = values.device
            values_cpu = values.cpu()
            rewards_cpu = rewards.cpu()
            terminals_cpu = terminals.cpu()
            ratio_cpu = ratio.cpu()
            advantages_cpu = advantages.cpu()

            torch.ops.pufferlib.compute_puff_advantage(
                values_cpu,
                rewards_cpu,
                terminals_cpu,
                ratio_cpu,
                advantages_cpu,
                gamma,
                gae_lambda,
                vtrace_rho_clip,
                vtrace_c_clip,
            )

            advantages.copy_(advantages_cpu.to(device))

        return advantages

    def close(self):
        self.vecenv.close()

    def initial_pr_uri(self):
        return self._initial_pr.uri

    def last_pr_uri(self):
        return self.last_pr.uri

    def _make_experience_buffer(self):
        """Create experience buffer with tensor-based storage for prioritized sampling."""
        vecenv = self.vecenv

        # Get environment info
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents

        # Calculate minibatch parameters
        batch_size = self.trainer_cfg.batch_size
        minibatch_size = self.trainer_cfg.minibatch_size
        max_minibatch_size = self.trainer_cfg.get("max_minibatch_size", minibatch_size)

        # Get LSTM parameters if using RNN
        use_rnn = self.trainer_cfg.get("use_rnn", True)
        hidden_size = getattr(self.policy, "hidden_size", 256)
        num_lstm_layers = 2  # Default value

        # Try to get actual number of LSTM layers from policy
        lstm = None
        if hasattr(self.policy, "components") and "_core_" in self.policy.components:
            lstm_module = self.policy.components["_core_"]
            if hasattr(lstm_module, "_net") and hasattr(lstm_module._net, "num_layers"):
                num_lstm_layers = lstm_module._net.num_layers
                lstm = lstm_module._net

        # Create experience buffer
        self.experience = Experience(
            total_agents=total_agents,
            batch_size=self._batch_size,
            bptt_horizon=self.trainer_cfg.bptt_horizon,
            minibatch_size=self._minibatch_size,
            max_minibatch_size=max_minibatch_size,
            obs_space=obs_space,
            atn_space=atn_space,
            device=self.device,
            cpu_offload=self.trainer_cfg.cpu_offload,
            use_rnn=use_rnn,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )

    def _make_losses(self):
        return SimpleNamespace(
            policy_loss=0,
            value_loss=0,
            entropy=0,
            old_approx_kl=0,
            approx_kl=0,
            clipfrac=0,
            explained_variance=0,
            l2_reg_loss=0,
            l2_init_loss=0,
            ks_action_loss=0,
            ks_value_loss=0,
            importance=0,
        )

    def _make_vecenv(self):
        """Create a vectorized environment."""
        num_agents = self._curriculum.get_task().env_cfg().game.num_agents

        self.target_batch_size = self.trainer_cfg.forward_pass_minibatch_target_size // num_agents
        if self.target_batch_size < 2:  # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2

        self.batch_size = (self.target_batch_size // self.trainer_cfg.num_workers) * self.trainer_cfg.num_workers
        logger.info(f"forward_pass_batch_size: {self.batch_size}")

        num_envs = self.batch_size * self.trainer_cfg.async_factor
        logger.info(f"num_envs: {num_envs}")

        if num_envs < 1:
            logger.error(
                f"num_envs = batch_size ({self.batch_size}) * async_factor ({self.trainer_cfg.async_factor}) "
                f"is {num_envs}, which is less than 1! (Increase trainer.forward_pass_minibatch_target_size)"
            )

        self.vecenv = make_vecenv(
            self._curriculum,
            self.cfg.vectorization,
            num_envs=num_envs,
            batch_size=self.batch_size,
            num_workers=self.trainer_cfg.num_workers,
            zero_copy=self.trainer_cfg.zero_copy,
        )

        if self.cfg.seed is None:
            self.cfg.seed = np.random.randint(0, 1000000)

        # Use rank-specific seed for environment reset to ensure different
        # processes generate uncorrelated environments in distributed training
        rank = int(os.environ.get("RANK", 0))
        self.vecenv.async_reset(self.cfg.seed + rank)

    def _load_policy(self, checkpoint, policy_store, metta_grid_env):
        """Load policy from checkpoint, initial_policy.uri, or create new."""
        for attempt in range(10):
            if checkpoint.policy_path:
                logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
                return policy_store.policy(checkpoint.policy_path)
            elif self.cfg.trainer.initial_policy.uri is not None:
                logger.info(f"Loading initial policy URI: {self.cfg.trainer.initial_policy.uri}")
                return policy_store.policy(self.cfg.trainer.initial_policy)
            else:
                policy_path = os.path.join(self.cfg.trainer.checkpoint_dir, policy_store.make_model_name(0))
                if os.path.exists(policy_path):
                    logger.info(f"Loading policy from checkpoint: {policy_path}")
                    return policy_store.policy(policy_path)
                elif self._master:
                    logger.info(f"Failed to load policy from default checkpoint: {policy_path}. Creating a new policy!")
                    return policy_store.create(metta_grid_env)
            time.sleep(5)

        raise RuntimeError("Failed to load policy after 10 attempts")


class AbortingTrainer(PufferTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_train_step(self):
        if self.wandb_run is None:
            return

        if "abort" not in wandb.Api().run(self.wandb_run.path).tags:
            return

        logger.info("Abort tag detected. Stopping the run.")
        self.cfg.trainer.total_timesteps = int(self.agent_step)
        self.wandb_run.config.update(
            {"trainer.total_timesteps": self.cfg.trainer.total_timesteps}, allow_val_change=True
        )
