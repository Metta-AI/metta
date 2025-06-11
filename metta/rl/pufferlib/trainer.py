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
from metta.rl.pufferlib.kickstarter import Kickstarter
from metta.rl.pufferlib.policy import PufferAgent
from metta.rl.pufferlib.profile import Profile
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

        # Backend optimization
        torch.backends.cudnn.deterministic = self.trainer_cfg.get("torch_deterministic", True)
        torch.backends.cudnn.benchmark = True

        # Distributed setup
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

        # Core components
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.losses = self._make_losses()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.average_reward = 0.0
        self._current_eval_score = None
        self._eval_grouped_scores = {}
        self._eval_suite_avgs = {}
        self._eval_categories = set()

        # Curriculum setup
        curriculum_config = self.trainer_cfg.get("curriculum", self.trainer_cfg.get("env", {}))
        env_overrides = DictConfig({"env_overrides": self.trainer_cfg.env_overrides})
        self._curriculum = curriculum_from_config_path(curriculum_config, env_overrides)

        # Create vecenv
        self._make_vecenv()

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv), (
            f"vecenv.driver_env type {type(metta_grid_env).__name__} is not MettaGridEnv"
        )

        # Load policy
        logger.info("Loading checkpoint")
        os.makedirs(cfg.trainer.checkpoint_dir, exist_ok=True)

        checkpoint = TrainerCheckpoint.load(cfg.run_dir)
        policy_record = self._load_policy(checkpoint)

        assert policy_record is not None, "No policy found"

        if self._master:
            logger.info(f"PufferTrainer loaded: {policy_record.policy()}")

        self._initial_pr = policy_record
        self.last_pr = policy_record
        self.policy_record = policy_record
        self.uncompiled_policy = policy_record.policy().to(self.device)
        self.policy = self.uncompiled_policy

        # Action setup
        actions_names = metta_grid_env.action_names
        actions_max_params = metta_grid_env.max_action_args
        self.policy.activate_actions(actions_names, actions_max_params, self.device)

        # Kickstarter
        self.kickstarter = Kickstarter(self.cfg, self.policy_store, actions_names, actions_max_params)

        # Compile policy if requested
        if self.trainer_cfg.compile:
            logger.info("Compiling policy")
            self.policy = torch.compile(self.policy, mode=self.trainer_cfg.compile_mode)

        # Distributed setup
        if torch.distributed.is_initialized():
            logger.info(f"Initializing DistributedDataParallel on device {self.device}")
            self._original_policy = self.policy
            self.policy = DistributedMettaAgent(self.policy, self.device)

        self._make_experience_buffer()

        # Training state
        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch
        self.global_step = self.agent_step
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()

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

        if checkpoint.agent_step > 0:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        # Learning rate scheduler
        self.lr_scheduler = None
        if self.trainer_cfg.lr_scheduler.enabled:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.trainer_cfg.total_timesteps // self.trainer_cfg.batch_size
            )

        # Mixed precision
        precision = self.trainer_cfg.get("precision", "float32")
        self.amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))

        # Monitoring
        self.profile = Profile(frequency=1)
        self.torch_profiler = None
        if cfg.trainer.get("profiler_interval_epochs", 0) > 0 and wandb_run is not None:
            self.torch_profiler = TorchProfiler(
                self._master, cfg.run_dir, cfg.trainer.profiler_interval_epochs, wandb_run
            )
        self.timer = Stopwatch(logger)
        self.timer.start()

        # Policy validation
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

        self.model_size = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        self.experience = None  # For compatibility

        # Replay config
        self.replay_sim_config = SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",
            num_episodes=1,
            env_overrides=self._curriculum.get_task().env_cfg(),
        )

        # Define wandb metrics
        if wandb_run and self._master:
            wandb_run.define_metric("train/agent_step")
            for k in ["overview", "env", "losses", "performance", "train"]:
                wandb_run.define_metric(f"{k}/*", step_metric="train/agent_step")

        logger.info(f"PufferTrainer initialization complete on device: {self.device}")

    def _load_policy(self, checkpoint):
        """Load policy from checkpoint or create new one."""
        load_policy_attempts = 10
        while load_policy_attempts > 0:
            if checkpoint.policy_path:
                logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
                policy_record = self.policy_store.policy(checkpoint.policy_path)
                if "average_reward" in checkpoint.extra_args:
                    self.average_reward = checkpoint.extra_args["average_reward"]
                return policy_record
            elif self.cfg.trainer.initial_policy.uri is not None:
                logger.info(f"Loading initial policy URI: {self.cfg.trainer.initial_policy.uri}")
                return self.policy_store.policy(self.cfg.trainer.initial_policy)
            else:
                policy_path = os.path.join(self.cfg.trainer.checkpoint_dir, self.policy_store.make_model_name(0))
                if os.path.exists(policy_path):
                    logger.info(f"Loading policy from checkpoint: {policy_path}")
                    return self.policy_store.policy(policy_path)
                elif self._master:
                    logger.info(f"Failed to load policy from default checkpoint: {policy_path}. Creating a new policy!")
                    metta_grid_env: MettaGridEnv = self.vecenv.driver_env
                    return self.policy_store.create(metta_grid_env)
            load_policy_attempts -= 1
            time.sleep(5)
        return None

    def _make_experience_buffer(self):
        """Create experience buffer with tensor-based storage for prioritized sampling."""
        vecenv = self.vecenv
        device = self.device

        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        batch_size = self.trainer_cfg.batch_size
        horizon = self.trainer_cfg.bptt_horizon
        segments = batch_size // horizon
        self.segments = segments

        # Create tensor storage
        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=torch.float32 if obs_space.dtype == np.float32 else torch.uint8,
            pin_memory=device == "cuda" and self.trainer_cfg.cpu_offload,
            device="cpu" if self.trainer_cfg.cpu_offload else device,
        )
        self.actions = torch.zeros(
            segments,
            horizon,
            *atn_space.shape,
            device=device,
            dtype=torch.int32 if atn_space.dtype in (np.int32, np.int64) else torch.float32,
        )

        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32) % segments
        self.free_idx = total_agents % segments

        # LSTM states
        if self.trainer_cfg.get("use_rnn", True):
            n = vecenv.agents_per_batch
            h = getattr(self.policy, "hidden_size", 256)
            num_layers = 2
            if hasattr(self.policy, "components") and "_core_" in self.policy.components:
                lstm_module = self.policy.components["_core_"]
                if hasattr(lstm_module, "_net") and hasattr(lstm_module._net, "num_layers"):
                    num_layers = lstm_module._net.num_layers
            self.lstm_h = {i * n: torch.zeros(num_layers, n, h, device=device) for i in range(total_agents // n)}
            self.lstm_c = {i * n: torch.zeros(num_layers, n, h, device=device) for i in range(total_agents // n)}

        # Minibatch setup
        minibatch_size = self.trainer_cfg.minibatch_size
        max_minibatch_size = self.trainer_cfg.get("max_minibatch_size", minibatch_size)
        self.minibatch_size = min(minibatch_size, max_minibatch_size)

        if batch_size < minibatch_size:
            raise ValueError(f"batch_size {batch_size} must be >= minibatch_size {minibatch_size}")

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(self.trainer_cfg.update_epochs * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon

        if self.minibatch_segments * horizon != self.minibatch_size:
            raise ValueError(f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}")

        self.full_rows = 0

    def train(self):
        logger.info("Starting training")

        if (
            self.trainer_cfg.evaluate_interval != 0
            and self.trainer_cfg.evaluate_interval < self.trainer_cfg.checkpoint_interval
        ):
            raise ValueError("evaluate_interval must be at least as large as checkpoint_interval")

        logger.info(f"Training on {self.device}")
        while self.agent_step < self.trainer_cfg.total_timesteps:
            steps_before = self.agent_step

            if self.torch_profiler:
                with self.torch_profiler:
                    with self.timer("_rollout"):
                        self._rollout()

                    with self.timer("_train"):
                        self._train()

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

            # Checkpointing
            if self.epoch % self.trainer_cfg.checkpoint_interval == 0:
                with self.timer("_checkpoint_trainer", log=logging.INFO):
                    self._checkpoint_trainer()

            # Evaluation
            if self.trainer_cfg.evaluate_interval != 0 and self.epoch % self.trainer_cfg.evaluate_interval == 0:
                with self.timer("_evaluate_policy", log=logging.INFO):
                    self._evaluate_policy()

            if self.torch_profiler:
                self.torch_profiler.on_epoch_end(self.epoch)

            if self.epoch % self.trainer_cfg.wandb_checkpoint_interval == 0:
                with self.timer("_save_policy_to_wandb"):
                    self._save_policy_to_wandb()

            if self.trainer_cfg.replay_interval != 0 and self.epoch % self.trainer_cfg.replay_interval == 0:
                with self.timer("_generate_and_upload_replay", log=logging.INFO):
                    self._generate_and_upload_replay()

            self._on_train_step()

        # Training complete
        timing_summary = self.timer.get_all_summaries()
        logger.info("Training complete!")
        for name, summary in timing_summary.items():
            logger.info(f"  {name}: {self.timer.format_time(summary['total_elapsed'])}")

        self._checkpoint_trainer()
        self._save_policy_to_wandb()

    def _rollout(self):
        """Rollout phase - collect experience from environments."""
        profile = self.profile
        profile.start_epoch(self.epoch, "eval")

        with profile.eval_misc:
            config = self.trainer_cfg
            device = self.device
            infos = defaultdict(list)

            self.full_rows = 0
            while self.full_rows < self.segments:
                with profile.env:
                    o, r, d, t, info, env_id, mask = self.vecenv.recv()

                    if self.trainer_cfg.get("require_contiguous_env_ids", False):
                        raise ValueError(
                            "We are assuming contiguous env id is always False. async_factor == num_workers = "
                            f"{self.trainer_cfg.async_factor} != {self.trainer_cfg.num_workers}"
                        )

                with profile.eval_misc:
                    env_id = slice(env_id[0], env_id[-1] + 1)
                    num_steps = sum(mask)
                    self.agent_step += num_steps * self._world_size
                    self.global_step = self.agent_step

                    o = torch.as_tensor(o)
                    o_device = o.to(device, non_blocking=True)
                    r = torch.as_tensor(r).to(device, non_blocking=True)
                    d = torch.as_tensor(d).to(device, non_blocking=True)
                    t = torch.as_tensor(t).to(device, non_blocking=True)

                with profile.eval_forward, torch.no_grad(), self.amp_context:
                    state = PolicyState()

                    if config.get("use_rnn", True) and hasattr(self, "lstm_h"):
                        batch_key = env_id.start
                        if batch_key in self.lstm_h:
                            state.lstm_h = self.lstm_h[batch_key]
                            state.lstm_c = self.lstm_c[batch_key]

                    actions, selected_action_log_probs, _, value, _ = self.policy(o_device, state)
                    logprob = selected_action_log_probs

                    if __debug__:
                        assert_shape(selected_action_log_probs, ("BT",), "selected_action_log_probs")
                        assert_shape(actions, ("BT", 2), "actions")

                with profile.eval_misc:
                    if config.get("use_rnn", True) and hasattr(self, "lstm_h"):
                        if state.lstm_h is not None and state.lstm_c is not None:
                            batch_key = env_id.start
                            if batch_key in self.lstm_h:
                                self.lstm_h[batch_key] = state.lstm_h
                                self.lstm_c[batch_key] = state.lstm_c

                    if self.device == "cuda":
                        torch.cuda.synchronize()

                with profile.eval_misc:
                    episode_length = self.ep_lengths[env_id.start].item()
                    indices = self.ep_indices[env_id]

                    if self.trainer_cfg.cpu_offload:
                        self.observations[indices, episode_length] = o
                    else:
                        self.observations[indices, episode_length] = o_device

                    if self.actions.dtype == torch.int32:
                        self.actions[indices, episode_length] = actions.int()
                    elif self.actions.dtype == torch.int64:
                        self.actions[indices, episode_length] = actions.long()
                    else:
                        self.actions[indices, episode_length] = actions

                    self.logprobs[indices, episode_length] = logprob
                    self.rewards[indices, episode_length] = r
                    self.terminals[indices, episode_length] = d.float()
                    self.truncations[indices, episode_length] = t.float()
                    self.values[indices, episode_length] = value.flatten()

                    self.ep_lengths[env_id] += 1
                    if episode_length + 1 >= self.trainer_cfg.bptt_horizon:
                        num_full = env_id.stop - env_id.start
                        self.ep_indices[env_id] = (
                            self.free_idx + torch.arange(num_full, device=device).int()
                        ) % self.segments
                        self.ep_lengths[env_id] = 0
                        self.free_idx = (self.free_idx + num_full) % self.segments
                        self.full_rows += num_full

                with profile.eval_misc:
                    for i in info:
                        for k, v in unroll_nested_dict(i):
                            infos[k].append(v)

                with profile.env:
                    actions_np = actions.cpu().numpy().astype(dtype_actions)
                    self.vecenv.send(actions_np)

            with profile.eval_misc:
                for k, v in infos.items():
                    if isinstance(v, np.ndarray):
                        processed_v = v.tolist()
                    else:
                        processed_v = v

                    if isinstance(processed_v, list):
                        if k not in self.stats:
                            self.stats[k] = []
                        self.stats[k].extend(processed_v)
                    else:
                        if k not in self.stats:
                            self.stats[k] = processed_v
                        else:
                            try:
                                self.stats[k] += processed_v
                            except TypeError:
                                self.stats[k] = [self.stats[k], processed_v]

                self.free_idx = self.total_agents % self.segments
                self.ep_indices = torch.arange(self.total_agents, device=self.device, dtype=torch.int32) % self.segments
                self.ep_lengths.zero_()

            return self.stats, infos

    def _get_experience_buffer_mean_reward(self) -> float:
        # Use rewards from tensor buffer
        if hasattr(self, "rewards") and self.rewards is not None:
            return float(self.rewards.mean().item())
        return 0.0

    def _train(self):
        """Train phase with prioritized experience replay and new advantage computation."""
        profile = self.profile
        profile.start_epoch(self.epoch, "train")
        self.losses = self._make_losses()

        with profile.train_misc:
            losses = defaultdict(float)
            config = self.trainer_cfg
            device = self.device

            # Prioritized sampling parameters
            b0 = config.get("prio_beta0", 0.6)
            a = config.get("prio_alpha", 0.0)  # Default to 0 for uniform sampling
            clip_coef = config.clip_coef
            vf_clip = config.vf_clip_coef if hasattr(config, "vf_clip_coef") else config.get("vf_clip_coef", 0.1)
            total_epochs = max(1, self.trainer_cfg.total_timesteps // self.trainer_cfg.batch_size)
            anneal_beta = b0 + (1 - b0) * a * self.epoch / total_epochs
            self.ratio[:] = 1

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)

            # Average reward adjustment
            if config.average_reward:
                # Average reward formulation: A_t = GAE(r_t - ρ, γ=1.0)
                # where ρ is the average reward estimate
                current_batch_mean = self._get_experience_buffer_mean_reward()
                alpha = self.trainer_cfg.average_reward_alpha

                # Update average reward estimate using EMA
                self.average_reward = (1 - alpha) * self.average_reward + alpha * current_batch_mean

                # Adjust rewards by subtracting average reward for advantage computation
                rewards_adjusted = self.rewards - self.average_reward
                # Set gamma to 1.0 for average reward case
                effective_gamma = 1.0
            else:
                rewards_adjusted = self.rewards
                effective_gamma = config.gamma

            # Compute advantages using pufferlib kernel
            torch.ops.pufferlib.compute_puff_advantage(
                self.values,
                rewards_adjusted,
                self.terminals,
                self.ratio,
                advantages,
                effective_gamma,
                config.gae_lambda,
                config.get("vtrace_rho_clip", 1.0),
                config.get("vtrace_c_clip", 1.0),
            )

        for mb in range(self.total_minibatches):
            with profile.train_misc:
                # Prioritized sampling
                adv = advantages.abs().sum(axis=1)
                prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
                prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
                idx = torch.multinomial(prio_probs, self.minibatch_segments)
                mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta

            # Minibatch data
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_terminals = self.terminals[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values

            with profile.train_forward:
                if not config.get("use_rnn", True):
                    mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)
                    # Don't use LSTM state if not using RNN
                    lstm_state = PolicyState()
                else:
                    # For RNN training, let the policy handle its own LSTM states internally
                    # The policy will process sequences and manage state across timesteps
                    lstm_state = PolicyState()

                # Use MettaAgent forward method with proper LSTM state
                _, newlogprob, entropy, newvalue, full_log_probs = self.policy(
                    mb_obs.to(device), lstm_state, action=mb_actions.to(device)
                )

            with profile.train_misc:
                if hasattr(newlogprob, "reshape"):
                    newlogprob = newlogprob.reshape(mb_logprobs.shape)

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                self.ratio[idx] = ratio

                # KL divergence tracking
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

                # Re-compute advantages with new ratios (V-trace)
                adv = advantages[idx]
                torch.ops.pufferlib.compute_puff_advantage(
                    mb_values,
                    rewards_adjusted[idx],
                    mb_terminals,
                    ratio,
                    adv,
                    effective_gamma,
                    config.gae_lambda,
                    config.get("vtrace_rho_clip", 1.0),
                    config.get("vtrace_c_clip", 1.0),
                )

                # Normalize advantages with prioritized weights
                if config.get("norm_adv", True):
                    adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
                else:
                    adv = mb_prio * adv

                # Losses
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(mb_returns.shape)
                if config.clip_vloss:
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Kickstarter losses
                if hasattr(self, "kickstarter"):
                    teacher_lstm_state = []
                    ks_action_loss, ks_value_loss = self.kickstarter.loss(
                        self.agent_step, full_log_probs, newvalue, mb_obs, teacher_lstm_state
                    )
                else:
                    ks_action_loss = torch.tensor(0.0, device=device)
                    ks_value_loss = torch.tensor(0.0, device=device)

                # L2 regularization losses
                l2_reg_loss = torch.tensor(0.0, device=device)
                if config.l2_reg_loss_coef > 0:
                    l2_reg_loss = config.l2_reg_loss_coef * self.policy.l2_reg_loss().to(device)

                l2_init_loss = torch.tensor(0.0, device=device)
                if config.l2_init_loss_coef > 0:
                    l2_init_loss = config.l2_init_loss_coef * self.policy.l2_init_loss().to(device)

                # Total loss
                loss = (
                    pg_loss
                    - config.ent_coef * entropy_loss
                    + config.vf_coef * v_loss
                    + l2_reg_loss
                    + l2_init_loss
                    + ks_action_loss
                    + ks_value_loss
                )

                # Update value estimates
                self.values[idx] = newvalue.detach().float()

                # Logging
                losses["policy_loss"] += pg_loss.item() / self.total_minibatches
                losses["value_loss"] += v_loss.item() / self.total_minibatches
                losses["entropy"] += entropy_loss.item() / self.total_minibatches
                losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
                losses["approx_kl"] += approx_kl.item() / self.total_minibatches
                losses["clipfrac"] += clipfrac.item() / self.total_minibatches
                losses["importance"] += ratio.mean().item() / self.total_minibatches
                losses["l2_reg_loss"] += l2_reg_loss.item() / self.total_minibatches
                losses["l2_init_loss"] += l2_init_loss.item() / self.total_minibatches
                losses["ks_action_loss"] += ks_action_loss.item() / self.total_minibatches
                losses["ks_value_loss"] += ks_value_loss.item() / self.total_minibatches

            # Learn on accumulated minibatches
            with profile.learn:
                loss.backward()
                if (mb + 1) % self.accumulate_minibatches == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

            # Early stopping based on KL
            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        # Update learning rate
        with profile.train_misc:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Calculate explained variance
            y_pred = self.values.flatten()
            y_true = advantages.flatten() + self.values.flatten()
            var_y = y_true.var()
            explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
            losses["explained_variance"] = explained_var.item() if torch.is_tensor(explained_var) else float("nan")

            # Update the SimpleNamespace object instead of replacing it
            for k, v in losses.items():
                setattr(self.losses, k, v)
            self.epoch += 1

            profile.update_stats(self.agent_step, self.trainer_cfg.total_timesteps)

    def _checkpoint_trainer(self):
        if not self._master:
            return

        pr = self._checkpoint_policy()
        if pr is None:
            logger.warning("Failed to checkpoint policy")
            return

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

        metta_grid_env: MettaGridEnv = self.vecenv.driver_env
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
        return self.last_pr

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
            self._eval_suite_avgs[f"{category}_score"] = score if score is not None else 0.0

        # Get overall score
        overall_score = stats_db.get_average_metric_by_filter("reward", self.last_pr)
        self._current_eval_score = overall_score if overall_score is not None else 0.0
        all_scores = stats_db.simulation_scores(self.last_pr, "reward")

        # Categorize scores
        self._eval_grouped_scores = {}
        for (_, sim_name, _), score in all_scores.items():
            for category in self._eval_categories:
                if category in sim_name.lower():
                    self._eval_grouped_scores[f"{category}/{sim_name.split('/')[-1]}"] = score

    def _save_policy_to_wandb(self):
        if not self._master:
            return

        if self.wandb_run is None:
            return

        pr = self._checkpoint_policy()
        if pr is not None:
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
        epoch = self.epoch
        learning_rate = self.optimizer.param_groups[0]["lr"]
        losses = {k: v for k, v in self.losses.__dict__.items() if not k.startswith("_")}
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
            steps_per_sec = self.agent_step / training_time if training_time > 0 else 0

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
                    "train/epoch": epoch,
                    "train/learning_rate": learning_rate,
                    "train/average_reward": self.average_reward if self.trainer_cfg.average_reward else None,
                    **timing_logs,
                }
            )

        self._eval_grouped_scores = {}
        self.stats.clear()

    def close(self):
        self.vecenv.close()

    def initial_pr_uri(self):
        return self._initial_pr.uri

    def last_pr_uri(self):
        return self.last_pr.uri

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
        if self.target_batch_size < 2:
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

        rank = int(os.environ.get("RANK", 0))
        rank_specific_env_seed = self.cfg.seed + rank if self.cfg.seed is not None else rank
        self.vecenv.async_reset(rank_specific_env_seed)

    def _on_train_step(self):
        pass


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
