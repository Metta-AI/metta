import time
from collections import defaultdict

import numpy as np
import pufferlib
import pufferlib.pytorch
import pufferlib.utils
import torch
from omegaconf import OmegaConf
import wandb
import os

from rl.pufferlib.dashboard import Dashboard
from rl.pufferlib.experience import Experience
from rl.pufferlib.profile import Profile
from rl.pufferlib.vecenv import make_vecenv
from rl.pufferlib.policy import load_policy_from_uri

from . import puffer_agent_wrapper

torch.set_float32_matmul_precision('high')

from fast_gae import fast_gae


class PufferTrainer:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.device = cfg.device

        self._make_vecenv()

        self.profile = Profile()
        self.dashboard = Dashboard(cfg, self.profile)

        self.vecenv.async_reset(self.cfg.seed)
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        atn_dtype = self.vecenv.single_action_space.dtype
        total_agents = self.vecenv.num_agents

        self.uncompiled_policy = None
        if self.cfg.train.init_policy_uri is None:
            self.uncompiled_policy = puffer_agent_wrapper.make_policy(self.vecenv.driver_env, self.cfg)
        else:
            self.dashboard.log(f"Loading policy from {self.cfg.train.init_policy_uri}")
            self.uncompiled_policy = load_policy_from_uri(self.cfg.train.init_policy_uri, self.cfg)
            self.dashboard.log(f"Initialized policy from {self.cfg.train.init_policy_uri}")
        self.dashboard.set_policy(self.uncompiled_policy)
        self.policy = self.uncompiled_policy
        if self.cfg.train.compile:
            self.policy = torch.compile(self.policy, mode=self.cfg.train.compile_mode)

        lstm = self.policy.lstm if hasattr(self.policy, 'lstm') else None
        self.experience = Experience(self.cfg.train.batch_size, self.cfg.train.bptt_horizon,
            self.cfg.train.minibatch_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
            self.cfg.train.cpu_offload, self.device, lstm, total_agents)

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
            lr=self.cfg.train.learning_rate, eps=1e-5)
        self.global_step = 0
        self.epoch = 0
        self.stats = defaultdict(list)
        self.last_log_time = 0

        if self.cfg.train.resume:
            self._try_load_checkpoint()

        if self.uncompiled_policy._action_names != self.vecenv.driver_env.action_names():
            raise ValueError(
                "Action names do not match between policy and environment: "
                f"{self.uncompiled_policy._action_names} != {self.vecenv.driver_env.action_names()}")

    def train(self):
        train_start = time.time()

        self.dashboard.log("Starting training")

        while self.global_step < self.cfg.train.total_timesteps:
            self._evaluate()
            self._train()

        train_time = time.time() - train_start

        num_evals = 0
        while len(self.stats) == 0:
            num_evals += 1
            self._evaluate()
            self.dashboard.log(f"Running final evaluation: {num_evals}")

        self.close()
        return self.stats, train_time

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
                self.dashboard.global_step = self.global_step

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
                o = o if self.cfg.train.cpu_offload else o_device
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
        self.dashboard.losses = self.losses

        with profile.train_misc:
            idxs = experience.sort_training_data()
            dones_np = experience.dones_np[idxs]
            values_np = experience.values_np[idxs]
            rewards_np = experience.rewards_np[idxs]
            # TODO: bootstrap between segment bounds
            advantages_np = fast_gae.compute_gae(dones_np, values_np,
                rewards_np, self.cfg.train.gamma, self.cfg.train.gae_lambda)
            experience.flatten_batch(advantages_np)

        # Optimizing the policy and value network
        total_minibatches = experience.num_minibatches * self.cfg.train.update_epochs
        for epoch in range(self.cfg.train.update_epochs):
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
                        clipfrac = ((ratio - 1.0).abs() > self.cfg.train.clip_coef).float().mean()

                    adv = adv.reshape(-1)
                    if self.cfg.train.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - self.cfg.train.clip_coef, 1 + self.cfg.train.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.cfg.train.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -self.cfg.train.vf_clip_coef,
                            self.cfg.train.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.cfg.train.ent_coef * entropy_loss + v_loss * self.cfg.train.vf_coef

                with profile.learn:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.max_grad_norm)
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

            if self.cfg.train.target_kl is not None:
                if approx_kl > self.cfg.train.target_kl:
                    break

        with profile.train_misc:
            if self.cfg.train.anneal_lr:
                frac = 1.0 - self.global_step / self.cfg.train.total_timesteps
                lrnow = frac * self.cfg.train.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            y_pred = experience.values_np
            y_true = experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.losses.explained_variance = explained_var
            self.epoch += 1
            self.dashboard.epoch = self.epoch
            profile.update(
                self.global_step,
                self.cfg.train.total_timesteps,
                self._timers
            )
            self._process_stats()
            self._maybe_save_checkpoint()

    def _maybe_save_checkpoint(self):
        done_training = self.global_step >= self.cfg.train.total_timesteps
        if self.epoch % self.cfg.train.checkpoint_interval != 0 and not done_training:
            return

        self.dashboard.log(f"Saving checkpoint at step {self.global_step}")
        path = os.path.join(self.cfg.data_dir, "pufferlib", self.cfg.experiment)
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            self.dashboard.log(f"Checkpoint already exists. Skipping save.")
            return model_path

        torch.save(self.uncompiled_policy, model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'epoch': self.epoch,
            'model_name': model_name,
            'exp_id': self.cfg.experiment,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.rename(state_path + '.tmp', state_path)

        self.dashboard.log(f"Checkpoint Saved. Model: {model_path}")

        if self.cfg.wandb.enabled:
            artifact_name = f"{self.cfg.experiment}_model"
            artifact = wandb.Artifact(
                artifact_name,
                type="model",
                metadata={
                    "model_name": model_name,
                    "agent_step": self.global_step,
                    "epoch": self.epoch,
                    "exp_id": self.cfg.experiment,
                }
            )
            artifact.add_file(model_path)
            artifact = wandb.run.log_artifact(artifact)
            artifact.wait()
            self.dashboard.log(f"Wandb Model Saved: {artifact.name}")

        return model_path


    def _try_load_checkpoint(self):
        path = os.path.join(self.cfg.data_dir, "pufferlib", self.cfg.experiment)
        if not os.path.exists(path):
            print('No checkpoints found. Assuming new experiment')
            return

        trainer_path = os.path.join(path, 'trainer_state.pt')
        resume_state = torch.load(trainer_path)
        model_path = os.path.join(path, resume_state['model_name'])
        self.global_step = resume_state['global_step']
        self.epoch = resume_state['epoch']
        self.uncompiled_policy = torch.load(model_path, map_location=self.device)
        self.policy = self.uncompiled_policy
        if self.cfg.train.compile:
            self.policy = torch.compile(self.policy, mode=self.cfg.train.compile_mode)
        self.optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        self.dashboard.log(f'Loaded checkpoint {resume_state["model_name"]}')

    def _process_stats(self):
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        self.dashboard.update_stats(self.stats)

        if self.cfg.wandb.track:
            wandb.log({
                '0verview/SPS': self.profile.SPS,
                '0verview/agent_steps': self.global_step,
            '0verview/epoch': self.epoch,
            '0verview/learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v for k, v in self.profile},
        })
        self.stats = defaultdict(list)

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
        self.target_batch_size = self.cfg.train.forward_pass_minibatch_target_size // self.cfg.env.game.num_agents
        if self.target_batch_size < 2: # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2
        self.batch_size = (self.target_batch_size // self.cfg.train.num_workers) * self.cfg.train.num_workers

        self.vecenv = make_vecenv(
            self.cfg,
            num_envs = self.batch_size * self.cfg.train.async_factor,
            batch_size = self.batch_size,
            num_workers=self.cfg.train.num_workers,
            zero_copy=self.cfg.train.zero_copy)
