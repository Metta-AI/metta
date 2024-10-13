import os
import time
import warnings
import logging
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from fast_gae import fast_gae
from omegaconf import OmegaConf
import wandb

import pufferlib
import pufferlib.utils

from rl.pufferlib.experience import Experience
from rl.pufferlib.policy import (
    count_params,
    load_policy_from_uri,
    upload_policy_to_wandb,
)
from rl.pufferlib.profile import Profile
from rl.pufferlib.vecenv import make_vecenv

from . import puffer_agent_wrapper

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)

class PolicyCheckpoint:
    def __init__(self):
        self.agent_steps = 0
        self.epoch = 0
        self.model_name = None
        self.model_path = None
        self.wandb_model_artifact = None
        self.num_params = 0

    def update(self, agent_steps: int, epoch: int, model_name: str, model_path: str):
        self.agent_steps = agent_steps
        self.epoch = epoch
        self.model_name = model_name
        self.model_path = model_path

class PufferTrainer:
    def __init__(self, cfg: OmegaConf, wandb_run, **kwargs):
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self.device = cfg.device
        self.profile = Profile()
        self.losses = self._make_losses()
        self.stats = defaultdict(list)
        self.recent_stats = defaultdict(list)
        self.policy_checkpoint = PolicyCheckpoint()
        self.wandb_run = wandb_run
        self.checkpoints = []

        self._make_vecenv()

        self.vecenv.async_reset(self.cfg.seed)
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        atn_dtype = self.vecenv.single_action_space.dtype
        total_agents = self.vecenv.num_agents

        self.uncompiled_policy = None
        if self.cfg.agent.policy_selector.uri is not None:
            logger.info(f"Loading policy from {self.cfg.agent.policy_selector.uri}")
            self.uncompiled_policy = load_policy_from_uri(self.cfg.agent.policy_selector.uri, self.cfg, self.wandb_run)
        if self.uncompiled_policy is None:
            self.uncompiled_policy = puffer_agent_wrapper.make_policy(self.vecenv.driver_env, self.cfg)
            torch.save(self.uncompiled_policy, os.path.join(self.cfg.run_dir, "initial_policy.pt"))
            self.uncompiled_policy.uri = os.path.join(self.cfg.run_dir, "initial_policy.pt")
            self.uncompiled_policy.name = "initial_policy.pt"
            logger.info("No initial policy found, creating new")

        self.policy = self.uncompiled_policy
        self.policy_checkpoint.num_params = count_params(self.uncompiled_policy)
        if self.trainer_cfg.compile:
            self.policy = torch.compile(self.policy, mode=self.trainer_cfg.compile_mode)

        lstm = self.policy.lstm if hasattr(self.policy, 'lstm') else None
        self.experience = Experience(self.trainer_cfg.batch_size, self.trainer_cfg.bptt_horizon,
            self.trainer_cfg.minibatch_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
            self.trainer_cfg.cpu_offload, self.device, lstm, total_agents)

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
            lr=self.trainer_cfg.learning_rate, eps=1e-5)
        self.global_step = 0
        self.epoch = 0
        if len(self.stats) > 0:
            self.recent_stats = deepcopy(self.stats)
        self.stats.clear()

        if self.trainer_cfg.resume:
            self._try_load_checkpoint()

        if self.uncompiled_policy._action_names != self.vecenv.driver_env.action_names():
            raise ValueError(
                "Action names do not match between policy and environment: "
                f"{self.uncompiled_policy._action_names} != {self.vecenv.driver_env.action_names()}")

        if self.cfg.wandb.track and wandb_run:
            wandb_run.define_metric("train/agent_steps")
            for k in ["0verview", "env", "losses", "performance", "train"]:
                wandb_run.define_metric(f"{k}/*", step_metric="train/agent_steps")

        assert hasattr(self.uncompiled_policy, "uri")
        assert hasattr(self.uncompiled_policy, "name")

    def train(self):
        self.train_start = time.time()
        logger.info("Starting training")

        while self.global_step < self.trainer_cfg.total_timesteps:
            self._evaluate()
            self._train()
            self._process_stats()
            if self.epoch % self.trainer_cfg.checkpoint_interval == 0:
                self._save_checkpoint()
            if self.epoch % self.trainer_cfg.wandb_checkpoint_interval == 0:
                self._upload_model_to_wandb()
            self._on_train_step()

        self.train_time = time.time() - self.train_start
        self._save_checkpoint()
        self._upload_model_to_wandb()
        logger.info(f"Training complete. Total time: {self.train_time:.2f} seconds")

    def _on_train_step(self):
        pass

    @pufferlib.utils.profile
    def _evaluate(self):
        experience, profile = self.experience, self.profile

        with profile.eval_misc:
            policy = self.policy
            infos = defaultdict(list)
            lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

        while not experience.full:
            with profile.env:
                o, r, d, t, info, env_id, mask = self.vecenv.recv()
                env_id = env_id.tolist()

            with profile.eval_misc:
                self.global_step += sum(mask)

                o = torch.as_tensor(o)
                o_device = o.to(self.device)
                r = torch.as_tensor(r)
                d = torch.as_tensor(d)

            with profile.eval_forward, torch.no_grad():
                # TODO: In place-update should be faster. Leaking 7% speed max
                # Also should be using a cuda tensor to index
                if lstm_h is not None:
                    h = lstm_h[:, env_id]
                    c = lstm_c[:, env_id]
                    actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                    lstm_h[:, env_id] = h
                    lstm_c[:, env_id] = c
                else:
                    actions, logprob, _, value = policy(o_device)

                if self.device == 'cuda':
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
            # TODO: bootstrap between segment bounds
            advantages_np = fast_gae.compute_gae(dones_np, values_np,
                rewards_np, self.trainer_cfg.gamma, self.trainer_cfg.gae_lambda)
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
                        _, newlogprob, entropy, newvalue, lstm_state = self.policy(
                            obs, state=lstm_state, action=atn)
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        _, newlogprob, entropy, newvalue = self.policy(
                            obs.reshape(-1, *self.vecenv.single_observation_space.shape),
                            action=atn,
                        )

                    if self.device == 'cuda':
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
                    loss = pg_loss - self.trainer_cfg.ent_coef * entropy_loss + v_loss * self.trainer_cfg.vf_coef

                with profile.learn:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.trainer_cfg.max_grad_norm)
                    self.optimizer.step()
                    if self.device == 'cuda':
                        torch.cuda.synchronize()

                with profile.train_misc:
                    self.losses.policy_loss += pg_loss.item() / total_minibatches
                    self.losses.value_loss += v_loss.item() / total_minibatches
                    self.losses.entropy += entropy_loss.item() / total_minibatches
                    self.losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                    self.losses.approx_kl += approx_kl.item() / total_minibatches
                    self.losses.clipfrac += clipfrac.item() / total_minibatches

            if self.trainer_cfg.target_kl is not None:
                if approx_kl > self.trainer_cfg.target_kl:
                    break

        with profile.train_misc:
            if self.trainer_cfg.anneal_lr:
                frac = 1.0 - self.global_step / self.trainer_cfg.total_timesteps
                lrnow = frac * self.trainer_cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            y_pred = experience.values_np
            y_true = experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.losses.explained_variance = explained_var
            self.epoch += 1
            profile.update(
                self.global_step,
                self.trainer_cfg.total_timesteps,
                self._timers
            )

    def _save_checkpoint(self):
        model_name = f'model_{self.epoch:06d}.pt'
        model_path = os.path.join(self.cfg.run_dir, model_name)
        self.checkpoints.append(model_path)
        self.uncompiled_policy.uri = model_path
        self.uncompiled_policy.name = model_name
        torch.save(self.uncompiled_policy, model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'epoch': self.epoch,
            'model_name': model_name,
            'run': self.cfg.run,
        }
        state_path = os.path.join(self.cfg.run_dir, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.rename(state_path + '.tmp', state_path)
        self.policy_checkpoint.update(self.global_step, self.epoch, model_name, model_path)
        logger.info(f"Saved model to {model_path}")

    def _upload_model_to_wandb(self):
        if self.policy_checkpoint is None:
            return
        if not self.wandb_run:
            return

        artifact_name = upload_policy_to_wandb(
            self.wandb_run,
            self.policy_checkpoint.model_path,
            f"{self.cfg.run}",
            metadata={
                "model_name": self.policy_checkpoint.model_name,
                "agent_step": self.policy_checkpoint.agent_steps,
                "epoch": self.policy_checkpoint.epoch,
                "run": self.cfg.run,
            }
        )

        return artifact_name

    def _try_load_checkpoint(self):
        logger.info("Trying to load training checkpoint")

        trainer_path = os.path.join(self.cfg.run_dir, 'trainer_state.pt')
        if not os.path.exists(trainer_path):
            logger.info('No trainer state found. Assuming new run')
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            resume_state = torch.load(trainer_path)
        model_path = os.path.join(self.cfg.run_dir, resume_state['model_name'])

        logger.info(f"Resuming from {model_path}")
        logger.info(f"Epoch: {resume_state['epoch']}")
        logger.info(f"Global step: {resume_state['global_step']}")

        self.global_step = resume_state['global_step']
        self.epoch = resume_state['epoch']

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.uncompiled_policy = load_policy_from_uri(model_path, self.cfg, self.wandb_run)

        self.policy = self.uncompiled_policy
        if self.trainer_cfg.compile:
            self.policy = torch.compile(self.policy, mode=self.trainer_cfg.compile_mode)
        self.optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        self.policy_checkpoint.update(self.global_step, self.epoch, resume_state['model_name'], model_path)

    def _process_stats(self):
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                self.stats[k] = np.mean(v)
            except:
                del self.stats[k]

        if self.wandb_run and self.cfg.wandb.track:
            overview = {
                'SPS': self.profile.SPS,
            }
            for k, v in self.trainer_cfg.stats.overview.items():
                if k in self.stats:
                    overview[v] = self.stats[k]

            self.wandb_run.log({
                **{f'0verview/{k}': v for k, v in overview.items()},
                **{f'env/{k}': v for k, v in self.stats.items()},
                **{f'losses/{k}': v for k, v in self.losses.items()},
                **{f'performance/{k}': v for k, v in self.profile},
                'train/agent_steps': self.global_step,
                'train/epoch': self.epoch,
                'train/learning_rate': self.optimizer.param_groups[0]["lr"],
            })

        if len(self.stats) > 0:
            self.recent_stats = deepcopy(self.stats)
        self.stats.clear()

    def close(self):
        self.vecenv.close()


    def _make_losses(self):
        return pufferlib.namespace(
            policy_loss=0,
            value_loss=0,
            entropy=0,
            old_approx_kl=0,
            approx_kl=0,
            clipfrac=0,
            explained_variance=0,
        )

    def _make_vecenv(self):
        self.target_batch_size = self.trainer_cfg.forward_pass_minibatch_target_size // self.cfg.env.game.num_agents
        if self.target_batch_size < 2: # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2
        self.batch_size = (self.target_batch_size // self.trainer_cfg.num_workers) * self.trainer_cfg.num_workers

        self.vecenv = make_vecenv(
            self.cfg,
            num_envs = self.batch_size * self.trainer_cfg.async_factor,
            batch_size = self.batch_size,
            num_workers=self.trainer_cfg.num_workers,
            zero_copy=self.trainer_cfg.zero_copy)


class AbortingTrainer(PufferTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_train_step(self):
        if self.wandb_run is None:
            return

        if "abort" not in wandb.Api().run(self.wandb_run.path).tags:
            return

        logger.info("Abort tag detected. Stopping the run.")
        self.cfg.trainer.total_timesteps = int(self.global_step)
        self.wandb_run.config.update({
            "trainer.total_timesteps": self.cfg.trainer.total_timesteps
        }, allow_val_change=True)
