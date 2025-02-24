from distutils import dist
import logging
import os
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import pufferlib
import pufferlib.utils
import torch
import wandb
from agent.policy_store import PolicyStore
from fast_gae import fast_gae
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from rl.eval.eval_stats_db import EvalStatsDB
from rl.eval.eval_stats_logger import EvalStatsLogger
from rl.pufferlib.experience import Experience
from rl.pufferlib.profile import Profile
from rl.pufferlib.trace import save_trace_image
from rl.pufferlib.trainer_checkpoint import TrainerCheckpoint
from rl.pufferlib.vecenv import make_vecenv

torch.set_float32_matmul_precision('high')

logger = logging.getLogger("trainer")

class PufferTrainer:
    def __init__(self,
                 cfg: OmegaConf,
                 wandb_run,
                 policy_store: PolicyStore,
                 **kwargs):

        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self.device = cfg.device

        # Configure NCCL for distributed training
        if self.trainer_cfg.dist.num_gpus > 1:
            self._setup_distributed_training()

        self.profile = Profile()
        self.losses = self._make_losses()
        self.stats = defaultdict(list)
        self.recent_stats = defaultdict(list)
        self.wandb_run = wandb_run
        self.policy_store = policy_store
        self.use_e3b = self.trainer_cfg.use_e3b
        self.eval_stats_logger = EvalStatsLogger(cfg, wandb_run)
        self.average_reward = 0.0  # Initialize average reward estimate
        self.policy_fitness = []

        self._make_vecenv()

        os.makedirs(cfg.trainer.checkpoint_dir, exist_ok=True)
        checkpoint = TrainerCheckpoint.load(cfg.run_dir)

        if checkpoint.policy_path:
            policy_record = policy_store.policy(checkpoint.policy_path)
            # Load average reward state if it exists
            if hasattr(checkpoint, 'average_reward'):
                self.average_reward = checkpoint.average_reward
        elif cfg.trainer.initial_policy.uri is not None:
            policy_record = policy_store.policy(cfg.trainer.initial_policy)
        else:
            logger.info("No initial policy found, creating new")
            policy_record = policy_store.create(self.vecenv.driver_env)

        if policy_record.metadata["action_names"] != self.vecenv.driver_env.action_names():
            raise ValueError(
                "Action names do not match between policy and environment: "
                f"{policy_record.metadata['action_names']} != {self.vecenv.driver_env.action_names()}")

        self._initial_pr = policy_record
        self.last_pr = policy_record
        self.policy = policy_record.policy()
        self.policy_record = policy_record
        self.uncompiled_policy = self.policy

        if self.trainer_cfg.compile:
            self.policy = torch.compile(self.policy, mode=self.trainer_cfg.compile_mode)

        if self.trainer_cfg.dist.num_gpus > 1:
            orig_policy = self.policy
            self.policy = DistributedDataParallel(self.policy, device_ids=[self.device])
            self.policy.lstm = orig_policy.lstm
            self.policy.hidden_size = orig_policy.hidden_size
            self.policy.compute_effective_rank = orig_policy.compute_effective_rank
            self.policy.update_l2_init_weight_copy = orig_policy.update_l2_init_weight_copy
            self.policy.l2_reg_loss = orig_policy.l2_reg_loss
            self.policy.l2_init_loss = orig_policy.l2_init_loss

        self._make_experience_buffer()

        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
            lr=self.trainer_cfg.learning_rate, eps=1e-5)

        if checkpoint.agent_step > 0:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        if self.cfg.wandb.track and wandb_run:
            wandb_run.define_metric("train/agent_step")
            for k in ["0verview", "env", "losses", "performance", "train"]:
                wandb_run.define_metric(f"{k}/*", step_metric="train/agent_step")

        print("Training on device:", self.device)

    def _setup_distributed_training(self):
        """Configure NCCL settings for distributed training."""
        nccl_cfg = self.trainer_cfg.dist.nccl

        # Basic settings
        os.environ['NCCL_TIMEOUT'] = str(nccl_cfg.timeout)
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = str(nccl_cfg.blocking_wait)
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = str(nccl_cfg.async_error_handling)

        # Debug settings
        os.environ['NCCL_DEBUG'] = str(nccl_cfg.debug)
        os.environ['NCCL_DEBUG_SUBSYS'] = str(nccl_cfg.debug_subsys)
        os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = str(nccl_cfg.trace_buffer_size)
        debug_file = os.path.join(self.cfg.run_dir, nccl_cfg.debug_file)
        os.environ['NCCL_DEBUG_FILE'] = debug_file
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)

        # Log key settings
        logger.info("NCCL Configuration:")
        logger.info(f"  Timeout: {nccl_cfg.timeout}s")
        logger.info(f"  Debug Level: {nccl_cfg.debug}")
        logger.info(f"  Debug File: {debug_file}")
        logger.info(f"  Blocking Wait: {'enabled' if nccl_cfg.blocking_wait else 'disabled'}")
        logger.info(f"  Async Error Handling: {'enabled' if nccl_cfg.async_error_handling else 'disabled'}")

    def train(self):
        self.train_start = time.time()
        logger.info("Starting training")

        # it doesn't make sense to evaluate more often than checkpointing since we need a saved policy to evaluate
        if self.trainer_cfg.evaluate_interval != 0 and self.trainer_cfg.evaluate_interval < self.trainer_cfg.checkpoint_interval:
            self.trainer_cfg.evaluate_interval = self.trainer_cfg.checkpoint_interval

        logger.info(f"Training on {self.device} GPUs")
        while self.agent_step < self.trainer_cfg.total_timesteps:
            # Collecting experience
            logger.info(f"{self.device} Evaluating policy")
            self._evaluate()

            # Training on collected experience
            logger.info(f"{self.device} Training on collected experience")
            self._train()

            # Processing stats
            logger.info(f"{self.device} Processing stats")
            self._process_stats()

            # Checkpointing trainer
            logger.info(f"{self.device} Checkpointing trainer")
            if self.epoch % self.trainer_cfg.checkpoint_interval == 0:
                self._checkpoint_trainer()
            if self.trainer_cfg.evaluate_interval != 0 and self.epoch % self.trainer_cfg.evaluate_interval == 0:
                self._evaluate_policy()
            if self.cfg.agent.effective_rank_interval != 0 and self.epoch % self.cfg.agent.effective_rank_interval == 0:
                self._compute_effective_rank()
            if self.epoch % self.trainer_cfg.wandb_checkpoint_interval == 0:
                self._save_policy_to_wandb()
            if self.cfg.agent.l2_init_weight_update_interval != 0 and self.epoch % self.cfg.agent.l2_init_weight_update_interval == 0:
                self._update_l2_init_weight_copy()
            if (self.trainer_cfg.trace_interval != 0 and
                self.epoch % self.trainer_cfg.trace_interval == 0):
                self._save_trace_to_wandb()


            self._on_train_step()

        self.train_time = time.time() - self.train_start
        self._checkpoint_trainer()
        self._save_policy_to_wandb()
        logger.info(f"Training complete. Total time: {self.train_time:.2f} seconds")

    def _evaluate_policy(self):

        self.cfg.eval.policy_uri = self.last_pr.uri
        self.cfg.analyzer.policy_uri = self.last_pr.uri

        eval = hydra.utils.instantiate(self.cfg.eval, self.policy_store, self.last_pr, self.cfg.env, _recursive_ = False)
        stats = eval.evaluate()
        self.eval_stats_logger.log(stats)

        eval_stats_db = EvalStatsDB.from_uri(self.cfg.eval.eval_db_uri, self.cfg.run_dir, self.wandb_run)
        analyzer = hydra.utils.instantiate(self.cfg.analyzer, eval_stats_db)
        _, policy_fitness_records = analyzer.analyze()
        self.policy_fitness = policy_fitness_records

    def _compute_effective_rank(self):
        effective_rank = self.policy.compute_effective_rank()
        for rank in effective_rank:
            self.wandb_run.log({
                f"train/effective_rank/{rank['name']}": rank['effective_rank'],
                "train/agent_step": self.agent_step,
                "train/epoch": self.epoch,
            })

    def _update_l2_init_weight_copy(self):
        self.policy.update_l2_init_weight_copy()

    def _on_train_step(self):
        pass

    @pufferlib.utils.profile
    def _evaluate(self):
        experience, profile = self.experience, self.profile

        with profile.eval_misc:
            policy = self.policy
            infos = defaultdict(list)
            lstm_h, lstm_c = experience.lstm_h, experience.lstm_c
            e3b_inv = experience.e3b_inv

        while not experience.full:
            with profile.env:
                o, r, d, t, info, env_id, mask = self.vecenv.recv()
                env_id = env_id.tolist()

            with profile.eval_misc:
                num_steps = sum(mask)
                self.agent_step += num_steps

                o = torch.as_tensor(o)
                o_device = o.to(self.device)
                r = torch.as_tensor(r)
                d = torch.as_tensor(d)

            with profile.eval_forward, torch.no_grad():
                # TODO: In place-update should be faster. Leaking 7% speed max
                # Also should be using a cuda tensor to index
                e3b = e3b_inv[env_id] if self.use_e3b else None

                if lstm_h is not None:
                    h = lstm_h[:, env_id]
                    c = lstm_c[:, env_id]
                    actions, logprob, _, value, (h, c), next_e3b, intrinsic_reward = policy(o_device, (h, c), e3b=e3b)
                    lstm_h[:, env_id] = h
                    lstm_c[:, env_id] = c

                if self.use_e3b:
                    e3b_inv[env_id] = next_e3b
                    r += intrinsic_reward.cpu()

                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()

            with profile.eval_misc:
                value = value.flatten()
                actions = actions.cpu().numpy()
                mask = torch.as_tensor(mask)# * policy.mask)
                o = o if self.trainer_cfg.cpu_offload else o_device
                self.experience.store(o, value, actions, logprob, r, d, env_id, mask)

                for i in info:
                    for k, v in pufferlib.utils.unroll_nested_dict(i):
                        infos[k].append(v)

            with profile.env:
                self.vecenv.send(actions)

        with profile.eval_misc:
            for k, v in infos.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                try:
                    iter(v)
                except TypeError:
                    self.stats[k].append(v)
                else:
                    self.stats[k] += v

        # TODO: Better way to enable multiple collects
        experience.ptr = 0
        experience.step = 0
        return self.stats, infos

    @pufferlib.utils.profile
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
                advantages_np = fast_gae.compute_gae(dones_np, values_np,
                    rewards_np_adjusted, effective_gamma, self.trainer_cfg.gae_lambda)
                # For average reward case, returns are computed differently:
                # R(s) = Σ(r_t - ρ) represents the bias function
                experience.returns_np = advantages_np + values_np
            else:
                effective_gamma = self.trainer_cfg.gamma
                # Standard GAE computation for discounted case
                advantages_np = fast_gae.compute_gae(dones_np, values_np,
                    rewards_np, effective_gamma, self.trainer_cfg.gae_lambda)
                experience.returns_np = advantages_np + values_np

            experience.flatten_batch(advantages_np)

        # Optimizing the policy and value network
        total_minibatches = experience.num_minibatches * self.trainer_cfg.update_epochs
        for epoch in range(self.trainer_cfg.update_epochs):
            lstm_state = None
            for mb in range(experience.num_minibatches):
                with profile.train_misc:
                    obs = experience.b_obs[mb]
                    obs = obs.to(self.device)
                    atn = experience.b_actions[mb]
                    log_probs = experience.b_logprobs[mb]
                    val = experience.b_values[mb]
                    adv = experience.b_advantages[mb]
                    ret = experience.b_returns[mb]

                with profile.train_forward:
                    if experience.lstm_h is not None:
                        _, newlogprob, entropy, newvalue, lstm_state, _, _ = self.policy(
                            obs, state=lstm_state, action=atn)
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())

                    # the below can be deleted if no LSTM
                    else:
                        _, newlogprob, entropy, newvalue, _, _ = self.policy(
                            obs.reshape(-1, *self.vecenv.single_observation_space.shape),
                            action=atn,
                        )

                    if self.device.startswith('cuda'):
                        torch.cuda.synchronize()

                with profile.train_misc:
                    logratio = newlogprob - log_probs.reshape(-1)
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
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - self.trainer_cfg.clip_coef, 1 + self.trainer_cfg.clip_coef
                    )
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

                    l2_reg_loss = torch.tensor(0.0, device=self.device)
                    if self.trainer_cfg.l2_reg_loss_coef > 0:
                        l2_reg_loss = self.trainer_cfg.l2_reg_loss_coef * self.policy.l2_reg_loss().to(self.device)

                    l2_init_loss = torch.tensor(0.0, device=self.device)
                    if self.trainer_cfg.l2_init_loss_coef > 0:
                        l2_init_loss = self.trainer_cfg.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)

                    loss = pg_loss - self.trainer_cfg.ent_coef * entropy_loss + v_loss * self.trainer_cfg.vf_coef + l2_reg_loss + l2_init_loss

                with profile.learn:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.trainer_cfg.max_grad_norm)
                    # if self.trainer_cfg.sync_gpus and self.trainer_cfg.dist.num_gpus > 1:
                    #     torch.distributed.barrier()
                    self.optimizer.step()

                    if self.cfg.agent.clip_range > 0:
                        self.policy.clip_weights()

                    if self.device.startswith('cuda'):
                        torch.cuda.synchronize()

                with profile.train_misc:
                    self.losses.policy_loss += pg_loss.item() / total_minibatches
                    self.losses.value_loss += v_loss.item() / total_minibatches
                    self.losses.entropy += entropy_loss.item() / total_minibatches
                    self.losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                    self.losses.approx_kl += approx_kl.item() / total_minibatches
                    self.losses.clipfrac += clipfrac.item() / total_minibatches
                    self.losses.l2_reg_loss += l2_reg_loss.item() / total_minibatches
                    self.losses.l2_init_loss += l2_init_loss.item() / total_minibatches

            if self.trainer_cfg.target_kl is not None:
                if approx_kl > self.trainer_cfg.target_kl:
                    break

        with profile.train_misc:
            if self.trainer_cfg.anneal_lr:
                frac = 1.0 - self.agent_step / self.trainer_cfg.total_timesteps
                lrnow = frac * self.trainer_cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            y_pred = experience.values_np
            y_true = experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.losses.explained_variance = explained_var
            self.epoch += 1
            profile.update(
                self.agent_step,
                self.trainer_cfg.total_timesteps,
                self._timers
            )

    def _checkpoint_trainer(self):
        pr = self._checkpoint_policy()
        self.checkpoint = TrainerCheckpoint(
            self.agent_step,
            self.epoch,
            self.optimizer.state_dict(),
            pr.local_path(),
            average_reward=self.average_reward  # Save average reward state
        ).save(self.cfg.run_dir)

    def _checkpoint_policy(self):
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
                "action_names": self.vecenv.driver_env.action_names(),
                "generation": generation,
                "initial_uri": self._initial_pr.uri
            }
        )
        # this is hacky, but otherwise the initial_pr points
        # at the same policy as the last_pr
        return self.last_pr

    def _save_policy_to_wandb(self):
        if self.wandb_run and self.cfg.wandb.track:
            pr = self._checkpoint_policy()
            self.policy_store.add_to_wandb_run(self.wandb_run.name, pr)

    def _save_trace_to_wandb(self):
        image_path = f"{self.cfg.run_dir}/traces/trace.{self.epoch}.png"
        save_trace_image(self.cfg, self.last_pr, image_path)
        wandb.log({"traces/actions": wandb.Image(image_path)})

    def _dist_sum(self, value):
        if not dist.is_initialized():
            return value

        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def _dist_mean(self, value):
        if not dist.is_initialized():
            return value

        return self._dist_sum(value) / dist.get_world_size()

    def _process_stats(self):
        # Process raw stats first
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
                self.stats[k] = v
            except:
                del self.stats[k]

        try:
            # Synchronize and aggregate stats across processes
            sps = self._dist_sum(self.profile.SPS)
            agent_steps = int(self._dist_sum(self.agent_step))
            epoch = int(self._dist_sum(self.epoch))
            learning_rate = self.optimizer.param_groups[0]["lr"]
            environment = {k: self._dist_mean(v) for k, v in self.stats.items()}
            losses = {k: self._dist_mean(v) for k, v in vars(self.losses).items() if not k.startswith('_')}
            performance = {k: self._dist_mean(v) for k, v in self.profile}
        except Exception as e:
            logger.error(f"Failed to synchronize stats: {str(e)}. Using local values.")
            sps = self.profile.SPS
            agent_steps = self.agent_step
            epoch = self.epoch
            learning_rate = self.optimizer.param_groups[0]["lr"]
            environment = self.stats
            losses = {k: v for k, v in vars(self.losses).items() if not k.startswith('_')}
            performance = self.profile

        # Only log from rank 0 when using distributed training
        should_log = self.wandb_run and self.cfg.wandb.track
        if self.trainer_cfg.dist.num_gpus > 1:
            should_log = should_log and torch.distributed.get_rank() == 0

        if should_log:
            overview = {'SPS': sps}
            for k, v in self.trainer_cfg.stats.overview.items():
                if k in environment:
                    overview[v] = environment[k]

            policy_fitness_metrics = {
                f'pfs/{r["eval"]}:{r["metric"]}': r["fitness"]
                for r in self.policy_fitness
            }
            self.policy_fitness = []

            self.wandb_run.log({
                **{f'0verview/{k}': v for k, v in overview.items()},
                **{f'env/{k}': v for k, v in environment.items()},
                **{f'losses/{k}': v for k, v in losses.items()},
                **{f'performance/{k}': v for k, v in performance.items()},
                **policy_fitness_metrics,
                'train/agent_step': agent_steps,
                'train/epoch': epoch,
                'train/learning_rate': learning_rate,
                'train/average_reward': self.average_reward if self.trainer_cfg.average_reward else None,
            })

        if len(self.stats) > 0:
            self.recent_stats = deepcopy(self.stats)
        self.stats.clear()

    def close(self):
        self.vecenv.close()

    def initial_pr_uri(self):
        return self._initial_pr.uri

    def last_pr_uri(self):
        return self.last_pr.uri

    def _make_experience_buffer(self):
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        atn_dtype = self.vecenv.single_action_space.dtype
        total_agents = self.vecenv.num_agents

        lstm = self.policy.lstm if hasattr(self.policy, 'lstm') else None
        self.experience = Experience(self.trainer_cfg.batch_size, self.trainer_cfg.bptt_horizon,
            self.trainer_cfg.minibatch_size, self.policy.hidden_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
            self.trainer_cfg.cpu_offload, self.device, lstm, total_agents)

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
        )

    def _make_vecenv(self):
        self.target_batch_size = self.trainer_cfg.forward_pass_minibatch_target_size // self.cfg.env.game.num_agents
        if self.target_batch_size < 2: # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2
        self.batch_size = (self.target_batch_size // self.trainer_cfg.num_workers) * self.trainer_cfg.num_workers

        self.vecenv = make_vecenv(
            self.cfg.env,
            self.cfg.vectorization,
            num_envs = self.batch_size * self.trainer_cfg.async_factor,
            batch_size = self.batch_size,
            num_workers=self.trainer_cfg.num_workers,
            zero_copy=self.trainer_cfg.zero_copy)

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
        self.wandb_run.config.update({
            "trainer.total_timesteps": self.cfg.trainer.total_timesteps
        }, allow_val_change=True)
