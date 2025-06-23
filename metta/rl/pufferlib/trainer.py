import logging
import os
import time
from collections import defaultdict

import hydra
import numpy as np
import pufferlib
import torch
import wandb
from heavyball import ForeachMuon
from omegaconf import DictConfig, ListConfig

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent
from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyStore
from metta.agent.util.debug import assert_shape
from metta.agent.util.weights_analysis import WeightsMetricsHelper
from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.fast_gae import compute_gae
from metta.rl.pufferlib.experience import Experience
from metta.rl.pufferlib.kickstarter import Kickstarter
from metta.rl.pufferlib.policy import PufferAgent
from metta.rl.pufferlib.profile import Profile
from metta.rl.pufferlib.torch_profiler import TorchProfiler
from metta.rl.pufferlib.trainer_checkpoint import TrainerCheckpoint
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.pufferlib_utils import profile, unroll_nested_dict
from mettagrid.curriculum import SamplingCurriculum
from mettagrid.mettagrid_env import MettaGridEnv

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
        if torch.distributed.is_initialized():
            self._master = int(os.environ["RANK"]) == 0
            self._world_size = torch.distributed.get_world_size()
            logger.info(
                f"Rank: {os.environ['RANK']}, Local rank: {os.environ['LOCAL_RANK']}, World size: {self._world_size}"
            )
            self.device = f"cuda:{os.environ['LOCAL_RANK']}"
            logger.info(f"Setting up distributed training on device {self.device}")

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
        self._weights_helper = WeightsMetricsHelper(cfg)
        env_overrides = DictConfig({"env_overrides": self.trainer_cfg.env_overrides})

        if "curriculum" in self.trainer_cfg:
            curriculum_cfg = config_from_path(self.trainer_cfg.curriculum, env_overrides)
            self._curriculum = hydra.utils.instantiate(curriculum_cfg)
        else:
            self._curriculum = SamplingCurriculum(self.trainer_cfg.env, env_overrides)
        self._make_vecenv()

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), (
            f"vecenv.driver_env type {type(metta_grid_env).__name__} is not MettaGridEnv"
        )

        logger.info("Loading checkpoint")
        os.makedirs(cfg.trainer.checkpoint_dir, exist_ok=True)
        checkpoint = TrainerCheckpoint.load(cfg.run_dir)

        policy_record = None
        load_policy_attempts = 10
        while policy_record is None and load_policy_attempts > 0:
            if checkpoint.policy_path:
                logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
                policy_record = policy_store.policy(checkpoint.policy_path)
                if "average_reward" in checkpoint.extra_args:
                    self.average_reward = checkpoint.extra_args["average_reward"]
            elif cfg.trainer.initial_policy.uri is not None:
                logger.info(f"Loading initial policy URI: {cfg.trainer.initial_policy.uri}")
                policy_record = policy_store.policy(cfg.trainer.initial_policy)
            else:
                policy_path = os.path.join(cfg.trainer.checkpoint_dir, policy_store.make_model_name(0))

                if os.path.exists(policy_path):
                    logger.info(f"Loading policy from checkpoint: {policy_path}")
                    policy_record = policy_store.policy(policy_path)
                elif self._master:
                    logger.info(f"Failed to load policy from default checkpoint: {policy_path}. Creating a new policy!")
                    policy_record = policy_store.create(metta_grid_env)
            if policy_record is not None:
                break
            load_policy_attempts -= 1
            time.sleep(5)

        assert policy_record is not None, "No policy found"

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

        self.kickstarter = Kickstarter(self.cfg, self.policy_store, actions_names, actions_max_params)

        if torch.distributed.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
            # Store the original policy for cleanup purposes
            self._original_policy = self.policy
            self.policy = DistributedMettaAgent(self.policy, self.device)

        self._make_experience_buffer()

        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch

        assert self.trainer_cfg.optimizer.type in (
            "adam",
            "muon",
        ), f"Optimizer type must be 'adam' or 'muon', got {self.trainer_cfg.optimizer.type}"
        opt_cls = torch.optim.Adam if self.trainer_cfg.optimizer.type == "adam" else ForeachMuon
        self.optimizer = opt_cls(
            self.policy.parameters(),
            lr=self.trainer_cfg.optimizer.learning_rate,
            betas=(self.trainer_cfg.optimizer.beta1, self.trainer_cfg.optimizer.beta2),
            eps=self.trainer_cfg.optimizer.eps,
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
                            f"component_name: {component_name}\n"
                            f"component_shape: {component_shape}\n"
                            f"environment_shape: {environment_shape}\n"
                        )
                    # delete below after evaluate is tested with tokenized obs
                    if len(environment_shape) == 2:
                        assert self.trainer_cfg.evaluate_interval == 0, (
                            "Tokenized obs agents aren't set up for evaluate yet (5-30-25)."
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
            for k in ["0verview", "env", "losses", "performance", "train"]:
                wandb_run.define_metric(f"{k}/*", step_metric="train/agent_step")

        self.replay_sim_config = SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",
            num_episodes=1,
            env_overrides=self._curriculum.get_task().env_cfg(),
        )

        logger.info(f"PufferTrainer initialization complete on device: {self.device}")

    def train(self):
        self.train_start = time.time()
        self.steps_start = self.agent_step

        logger.info("Starting training")

        # it doesn't make sense to evaluate more often than checkpointing since we need a saved policy to evaluate
        if (
            self.trainer_cfg.evaluate_interval != 0
            and self.trainer_cfg.evaluate_interval < self.trainer_cfg.checkpoint_interval
        ):
            raise ValueError("evaluate_interval must be at least as large as checkpoint_interval")

        logger.info(f"Training on {self.device}")
        while self.agent_step < self.trainer_cfg.total_timesteps:
            with self.torch_profiler:
                self._rollout()

                # Training on collected experience
                self._train()

            # Processing stats
            self._process_stats()

            # log progress
            steps_per_second = (self.agent_step - self.steps_start) / (time.time() - self.train_start)
            remaining_steps = self.trainer_cfg.total_timesteps - self.agent_step
            remaining_time_sec = remaining_steps / steps_per_second

            # Format remaining time in appropriate units
            if remaining_time_sec < 60:
                time_str = f"{remaining_time_sec:.0f} sec"
            elif remaining_time_sec < 3600:
                time_str = f"{remaining_time_sec / 60:.1f} min"
            elif remaining_time_sec < 86400:  # Less than a day
                time_str = f"{remaining_time_sec / 3600:.1f} hours"
            else:
                time_str = f"{remaining_time_sec / 86400:.1f} days"

            logger.info(
                f"Epoch {self.epoch} - {self.agent_step} [{steps_per_second:.0f}/sec]"
                f" ({100.00 * self.agent_step / self.trainer_cfg.total_timesteps:.2f}%)"
                f" - {time_str} remaining"
            )

            # Checkpointing trainer
            if self.epoch % self.trainer_cfg.checkpoint_interval == 0:
                self._checkpoint_trainer()
            if self.trainer_cfg.evaluate_interval != 0 and self.epoch % self.trainer_cfg.evaluate_interval == 0:
                self._evaluate_policy()
            self._weights_helper.on_epoch_end(self.epoch, self.policy)
            self.torch_profiler.on_epoch_end(self.epoch)
            if self.epoch % self.trainer_cfg.wandb_checkpoint_interval == 0:
                self._save_policy_to_wandb()
            if (
                self.cfg.agent.l2_init_weight_update_interval != 0
                and self.epoch % self.cfg.agent.l2_init_weight_update_interval == 0
            ):
                self._update_l2_init_weight_copy()
            if self.trainer_cfg.replay_interval != 0 and self.epoch % self.trainer_cfg.replay_interval == 0:
                self._generate_and_upload_replay()

            self._on_train_step()

        self.train_time = time.time() - self.train_start
        self._checkpoint_trainer()
        self._save_policy_to_wandb()
        logger.info(f"Training complete. Total time: {self.train_time:.2f} seconds")

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

    def _update_l2_init_weight_copy(self):
        self.policy.update_l2_init_weight_copy()

    def _on_train_step(self):
        pass

    @profile
    def _rollout(self):
        experience, profile = self.experience, self.profile

        with profile.eval_misc:
            policy = self.policy
            infos = defaultdict(list)
            lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

        while not experience.full:
            with profile.env:
                o, r, d, t, info, env_id, mask = self.vecenv.recv()

                if self.trainer_cfg.require_contiguous_env_ids:
                    raise ValueError(
                        "We are assuming contiguous eng id is always False. async_factor == num_workers = "
                        f"{self.trainer_cfg.async_factor} != {self.trainer_cfg.num_workers}"
                    )

                training_env_id = torch.as_tensor(env_id).to(self.device, non_blocking=True)

            with profile.eval_misc:
                num_steps = sum(mask)
                self.agent_step += num_steps * self._world_size

                o = torch.as_tensor(o)
                r = torch.as_tensor(r)
                d = torch.as_tensor(d)

            with profile.eval_forward, torch.no_grad():
                assert training_env_id.dtype in [torch.int32, torch.int64], "training_env_id must be integer type"
                assert training_env_id.device == lstm_h.device, "training_env_id must be on the same device as lstm_h"
                assert training_env_id.dim() == 1, "training_env_id should be 1D (list of env indices)"
                assert training_env_id.max() < lstm_h.shape[1], "Index out of bounds for lstm_h"
                assert training_env_id.min() >= 0, "Negative index in training_env_id"

                state = PolicyState(lstm_h=lstm_h[:, training_env_id], lstm_c=lstm_c[:, training_env_id])

                o_device = o.to(self.device, non_blocking=True)
                actions, selected_action_log_probs, _, value, _ = policy(o_device, state)

                if __debug__:
                    assert_shape(selected_action_log_probs, ("BT",), "collected_log_probs")

                lstm_h[:, training_env_id] = (
                    state.lstm_h if state.lstm_h is not None else torch.zeros_like(lstm_h[:, training_env_id])
                )
                lstm_c[:, training_env_id] = (
                    state.lstm_c if state.lstm_c is not None else torch.zeros_like(lstm_c[:, training_env_id])
                )

                if self.device == "cuda":
                    torch.cuda.synchronize()

            with profile.eval_misc:
                value = value.flatten()
                mask = torch.as_tensor(mask)  # * policy.mask)
                o = o if self.trainer_cfg.cpu_offload else o_device
                self.experience.store(o, value, actions, selected_action_log_probs, r, d, training_env_id, mask)

                for i in info:
                    for k, v in unroll_nested_dict(i):
                        infos[k].append(v)

            with profile.env:
                actions = actions.cpu().numpy()
                self.vecenv.send(actions)

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
        experience.ptr = 0
        experience.step = 0
        return self.stats, infos

    @profile
    def _train(self):
        experience, profile = self.experience, self.profile
        self.losses = self._make_losses()

        with profile.train_misc:
            idxs = experience.sort_training_data()
            dones_np = experience.dones_np[idxs]
            values_np = experience.values_np[idxs]
            rewards_np = experience.rewards_np[idxs]

            # Update average reward estimate
            if self.trainer_cfg.average_reward:
                # Update average reward estimate using EMA with configured alpha
                alpha = self.trainer_cfg.average_reward_alpha
                self.average_reward = (1 - alpha) * self.average_reward + alpha * np.mean(rewards_np)
                # Adjust rewards by subtracting average reward for advantage computation
                rewards_np_adjusted = rewards_np - self.average_reward
                # Set gamma to 1.0 for average reward case
                effective_gamma = 1.0
                # Compute advantages using adjusted rewards
                advantages_np = compute_gae(
                    dones_np, values_np, rewards_np_adjusted, effective_gamma, self.trainer_cfg.gae_lambda
                )
                # For average reward case, returns are computed differently:
                # R(s) = Σ(r_t - ρ) represents the bias function
                experience.returns_np = advantages_np + values_np
            else:
                effective_gamma = self.trainer_cfg.gamma
                # Standard GAE computation for discounted case
                advantages_np = compute_gae(
                    dones_np, values_np, rewards_np, effective_gamma, self.trainer_cfg.gae_lambda
                )
                experience.returns_np = advantages_np + values_np

            experience.flatten_batch(advantages_np)

        # Optimizing the policy and value network
        total_minibatches = experience.num_minibatches * self.trainer_cfg.update_epochs
        for _epoch in range(self.trainer_cfg.update_epochs):
            lstm_state = PolicyState()
            teacher_lstm_state = []
            for mb in range(experience.num_minibatches):
                with profile.train_misc:
                    obs = experience.b_obs[mb]
                    obs = obs.to(self.device, non_blocking=True)
                    atn = experience.b_actions[mb]
                    old_action_log_probs = experience.b_logprobs[mb]
                    val = experience.b_values[mb]
                    adv = experience.b_advantages[mb]
                    ret = experience.b_returns[mb]

                with profile.train_forward:
                    # Forward pass returns: (action, new_action_log_probs, entropy, value, full_log_probs_distribution)
                    _, new_action_log_probs, entropy, newvalue, full_log_probs_distribution = self.policy(
                        obs, lstm_state, action=atn
                    )
                    if self.device == "cuda":
                        torch.cuda.synchronize()

                with profile.train_misc:
                    if __debug__:
                        assert_shape(new_action_log_probs, ("BT",), "new_action_log_probs")
                        assert_shape(old_action_log_probs, ("B", "T"), "old_action_log_probs")

                    logratio = new_action_log_probs - old_action_log_probs.reshape(-1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > self.trainer_cfg.clip_coef).float().mean()

                    adv = adv.reshape(-1)
                    if self.trainer_cfg.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1 - self.trainer_cfg.clip_coef, 1 + self.trainer_cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.trainer_cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -self.trainer_cfg.vf_clip_coef,
                            self.trainer_cfg.vf_clip_coef,
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
                    if self.trainer_cfg.l2_reg_loss_coef > 0:
                        l2_reg_loss = self.trainer_cfg.l2_reg_loss_coef * self.policy.l2_reg_loss().to(self.device)

                    l2_init_loss = torch.tensor(0.0, device=self.device)
                    if self.trainer_cfg.l2_init_loss_coef > 0:
                        l2_init_loss = self.trainer_cfg.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)

                    loss = (
                        pg_loss
                        - self.trainer_cfg.ent_coef * entropy_loss
                        + v_loss * self.trainer_cfg.vf_coef
                        + l2_reg_loss
                        + l2_init_loss
                        + ks_action_loss
                        + ks_value_loss
                    )

                with profile.learn:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.trainer_cfg.max_grad_norm)
                    self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                    if self.device == "cuda":
                        torch.cuda.synchronize()

                with profile.train_misc:
                    if self.losses is None:
                        raise ValueError("self.losses is None")

                    self.losses.policy_loss += pg_loss.item() / total_minibatches
                    self.losses.value_loss += v_loss.item() / total_minibatches
                    self.losses.entropy += entropy_loss.item() / total_minibatches
                    self.losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                    self.losses.approx_kl += approx_kl.item() / total_minibatches
                    self.losses.clipfrac += clipfrac.item() / total_minibatches
                    self.losses.l2_reg_loss += l2_reg_loss.item() / total_minibatches
                    self.losses.l2_init_loss += l2_init_loss.item() / total_minibatches
                    self.losses.ks_action_loss += ks_action_loss.item() / total_minibatches
                    self.losses.ks_value_loss += ks_value_loss.item() / total_minibatches

            if self.trainer_cfg.target_kl is not None:
                if approx_kl > self.trainer_cfg.target_kl:
                    break

        with profile.train_misc:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            y_pred = experience.values_np
            y_true = experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.losses.explained_variance = explained_var
            self.epoch += 1
            profile.update(self.agent_step, self.trainer_cfg.total_timesteps, self._timers)

    def _checkpoint_trainer(self):
        if not self._master:
            return

        pr = self._checkpoint_policy()
        self.checkpoint = TrainerCheckpoint(
            self.agent_step,
            self.epoch,
            self.optimizer.state_dict(),
            pr.local_path(),
            average_reward=self.average_reward,  # Save average reward state
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
                "train_time": time.time() - self.train_start,
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
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
                self.stats[k] = v
            except (TypeError, ValueError):
                del self.stats[k]

        # Now synchronize and aggregate stats across processes
        sps = self.profile.SPS
        agent_steps = self.agent_step
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

        if self.wandb_run and self._master:
            self.wandb_run.log(
                {
                    **{f"overview/{k}": v for k, v in overview.items()},
                    **{f"losses/{k}": v for k, v in losses.items()},
                    **{f"performance/{k}": v for k, v in performance.items()},
                    **environment,
                    **self._weights_helper.stats(),
                    **self._eval_grouped_scores,
                    "train/agent_step": agent_steps,
                    "train/epoch": epoch,
                    "train/learning_rate": learning_rate,
                    "train/average_reward": self.average_reward if self.trainer_cfg.average_reward else None,
                }
            )

        self._eval_grouped_scores = {}
        self._weights_helper.reset()
        self.stats.clear()

    def close(self):
        self.vecenv.close()

    def initial_pr_uri(self):
        return self._initial_pr.uri

    def last_pr_uri(self):
        return self.last_pr.uri

    def _make_experience_buffer(self):
        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), (
            "vecenv.driver_env must be a MettaGridEnv for experience buffer"
        )

        # Extract environment specifications
        obs_shape = metta_grid_env.single_observation_space.shape
        obs_dtype = metta_grid_env.single_observation_space.dtype
        atn_shape = metta_grid_env.single_action_space.shape
        atn_dtype = metta_grid_env.single_action_space.dtype

        # Use num_agents for the total number of environments/states to track
        lstm_total_agents = getattr(self.vecenv, "num_agents", 0)
        assert lstm_total_agents > 0, "self.vecenv.num_agents not found!"
        logging.info(f"Creating experience buffer with lstm_total_agents={lstm_total_agents} (from vecenv.num_agents)")

        # Handle policy fields with assertions
        assert hasattr(self.policy, "hidden_size"), "Policy must have hidden_size attribute"
        hidden_size = int(getattr(self.policy, "hidden_size", -1))
        assert hidden_size > 0, f"Policy hidden_size cannot be converted to int: {type(hidden_size)}"

        assert hasattr(self.policy, "lstm"), "Policy must have lstm attribute"
        lstm = getattr(self.policy, "lstm", {})
        assert isinstance(lstm, torch.nn.modules.rnn.LSTM), (
            f"Policy lstm must be a valid LSTM instance, got: {type(lstm)}"
        )

        # Create the Experience buffer with appropriate parameters
        self.experience = Experience(
            batch_size=self.trainer_cfg.batch_size,  # Total number of environment steps to collect before updating
            bptt_horizon=self.trainer_cfg.bptt_horizon,  # Sequence length for BPTT (backpropagation through time)
            minibatch_size=self.trainer_cfg.minibatch_size,  # Size of minibatches for training
            hidden_size=hidden_size,  # Dimension of the policy's hidden state
            obs_shape=obs_shape,  # Shape of a single observation
            obs_dtype=obs_dtype,  # Data type of observations
            atn_shape=atn_shape,  # Shape of a single action
            atn_dtype=atn_dtype,  # Data type of actions
            cpu_offload=self.trainer_cfg.cpu_offload,  # Whether to store data on CPU and transfer to GPU as needed
            device=self.device,  # Device to store tensors on ("cuda" or "cpu")
            lstm=lstm,  # LSTM module from the policy (needed for dimensions) # type: ignore - Pylance is wrong
            lstm_total_agents=lstm_total_agents,  # Total number of LSTM states to maintain
        )

    def _make_losses(self):
        return pufferlib.namespace(
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
        )

    def _make_vecenv(self):
        """Create a vectorized environment."""

        num_agents = self._curriculum.get_task().env_cfg().game.num_agents

        self.target_batch_size = self.trainer_cfg.forward_pass_minibatch_target_size // num_agents
        if self.target_batch_size < 2:  # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2

        self.batch_size = (self.target_batch_size // self.trainer_cfg.num_workers) * self.trainer_cfg.num_workers
        num_envs = self.batch_size * self.trainer_cfg.async_factor

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

        self.vecenv.async_reset(self.cfg.seed)


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
