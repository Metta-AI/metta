import os
import time
from collections import defaultdict

import numpy as np
import pufferlib
import pufferlib.pytorch
import pufferlib.utils
import torch
from omegaconf import OmegaConf

from rl.pufferlib.dashboard import print_dashboard
from rl.pufferlib.profile import Profile
from rl.pufferlib.utilization import Utilization
from rl.pufferlib.vecenv import make_vecenv

from . import puffer_agent_wrapper

torch.set_float32_matmul_precision('high')

from fast_gae import fast_gae

import wandb


class PufferTrainer:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.pcfg = cfg.framework.pufferlib
        self.target_batch_size = self.pcfg.train.forward_pass_minibatch_target_size // cfg.env.game.num_agents
        if self.target_batch_size < 2: # pufferlib bug requires batch size >= 2
            self.target_batch_size = 2
        self.batch_size = (self.target_batch_size // self.pcfg.train.num_workers) * self.pcfg.train.num_workers

        self.vecenv = make_vecenv(
            cfg,
            num_envs = self.batch_size * self.pcfg.train.async_factor,
            batch_size = self.batch_size,
            num_workers=self.pcfg.train.num_workers,
            zero_copy=self.pcfg.train.zero_copy)

        self.policy = puffer_agent_wrapper.make_policy(self.vecenv.driver_env, self.cfg)

        self.profile = Profile()
        self.losses = self.make_losses()

        utilization = Utilization()
        msg = f'Model Size: {abbreviate(count_params(policy))} parameters'
        if config.dashboard:
            print_dashboard(config.env, utilization, 0, 0, profile, losses, {}, msg, clear=True)

        vecenv.async_reset(config.seed)
        obs_shape = vecenv.single_observation_space.shape
        obs_dtype = vecenv.single_observation_space.dtype
        atn_shape = vecenv.single_action_space.shape
        atn_dtype = vecenv.single_action_space.dtype
        total_agents = vecenv.num_agents

        lstm = policy.lstm if hasattr(policy, 'lstm') else None
        experience = Experience(config.batch_size, config.bptt_horizon,
            config.minibatch_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
            config.cpu_offload, config.device, lstm, total_agents)

        uncompiled_policy = policy

        if config.compile:
            policy = torch.compile(policy, mode=config.compile_mode)

        optimizer = torch.optim.Adam(policy.parameters(),
            lr=config.learning_rate, eps=1e-5)

        wandb_or_none = None
        if wandb.run is not None:
            wandb_or_none = wandb

        return pufferlib.namespace(
            config=config,
            vecenv=vecenv,
            policy=policy,
            uncompiled_policy=uncompiled_policy,
            optimizer=optimizer,
            experience=experience,
            profile=profile,
            losses=losses,
            wandb=wandb_or_none,
            global_step=0,
            epoch=0,
            stats=defaultdict(list),
            last_stats=defaultdict(list),
            msg=msg,
            last_msg="",
            last_log_time=0,
            utilization=utilization,
        )

    def load_checkpoint(self):
        clean_pufferl.try_load_checkpoint(self.data)

    def train(self):
        train_start = time.time()

        print(f"Starting training: {self.data.global_step}/{self.pcfg.train.total_timesteps} timesteps")

        while self.data.global_step < self.pcfg.train.total_timesteps:
            clean_pufferl.evaluate(self.data)
            clean_pufferl.train(self.data)

        train_time = time.time() - train_start
        clean_pufferl.close(self.data)
        return self.data.last_stats, train_time

    @pufferlib.utils.profile
    def evaluate(self):
        config, profile, experience = data.config, data.profile, data.experience

        with profile.eval_misc:
            policy = data.policy
            infos = defaultdict(list)
            lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

        while not experience.full:
            with profile.env:
                o, r, d, t, info, env_id, mask = data.vecenv.recv()
                env_id = env_id.tolist()

            with profile.eval_misc:
                data.global_step += sum(mask)

                o = torch.as_tensor(o)
                o_device = o.to(config.device)
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

                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.eval_misc:
                value = value.flatten()
                actions = actions.cpu().numpy()
                mask = torch.as_tensor(mask)# * policy.mask)
                o = o if config.cpu_offload else o_device
                experience.store(o, value, actions, logprob, r, d, env_id, mask)

                for i in info:
                    for k, v in pufferlib.utils.unroll_nested_dict(i):
                        infos[k].append(v)

            with profile.env:
                data.vecenv.send(actions)

        with profile.eval_misc:
            for k, v in infos.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                try:
                    iter(v)
                except TypeError:
                    data.stats[k].append(v)
                else:
                    data.stats[k] += v

        # TODO: Better way to enable multiple collects
        data.experience.ptr = 0
        data.experience.step = 0
        return data.stats, infos

    @pufferlib.utils.profile
    def train(data):
        config, profile, experience = data.config, data.profile, data.experience
        data.losses = make_losses()
        losses = data.losses

        with profile.train_misc:
            idxs = experience.sort_training_data()
            dones_np = experience.dones_np[idxs]
            values_np = experience.values_np[idxs]
            rewards_np = experience.rewards_np[idxs]
            # TODO: bootstrap between segment bounds
            advantages_np = fast_gae.compute_gae(dones_np, values_np,
                rewards_np, config.gamma, config.gae_lambda)
            experience.flatten_batch(advantages_np)

        # Optimizing the policy and value network
        total_minibatches = experience.num_minibatches * config.update_epochs
        mean_pg_loss, mean_v_loss, mean_entropy_loss = 0, 0, 0
        mean_old_kl, mean_kl, mean_clipfrac = 0, 0, 0
        for epoch in range(config.update_epochs):
            lstm_state = None
            for mb in range(experience.num_minibatches):
                with profile.train_misc:
                    obs = experience.b_obs[mb]
                    obs = obs.to(config.device)
                    atn = experience.b_actions[mb]
                    log_probs = experience.b_logprobs[mb]
                    val = experience.b_values[mb]
                    adv = experience.b_advantages[mb]
                    ret = experience.b_returns[mb]

                with profile.train_forward:
                    if experience.lstm_h is not None:
                        _, newlogprob, entropy, newvalue, lstm_state = data.policy(
                            obs, state=lstm_state, action=atn)
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        _, newlogprob, entropy, newvalue = data.policy(
                            obs.reshape(-1, *data.vecenv.single_observation_space.shape),
                            action=atn,
                        )

                    if config.device == 'cuda':
                        torch.cuda.synchronize()

                with profile.train_misc:
                    logratio = newlogprob - log_probs.reshape(-1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

                    adv = adv.reshape(-1)
                    if config.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - config.clip_coef, 1 + config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -config.vf_clip_coef,
                            config.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                with profile.learn:
                    data.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)
                    data.optimizer.step()
                    if config.device == 'cuda':
                        torch.cuda.synchronize()

                with profile.train_misc:
                    losses.policy_loss += pg_loss.item() / total_minibatches
                    losses.value_loss += v_loss.item() / total_minibatches
                    losses.entropy += entropy_loss.item() / total_minibatches
                    losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                    losses.approx_kl += approx_kl.item() / total_minibatches
                    losses.clipfrac += clipfrac.item() / total_minibatches

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        with profile.train_misc:
            if config.anneal_lr:
                frac = 1.0 - data.global_step / config.total_timesteps
                lrnow = frac * config.learning_rate
                data.optimizer.param_groups[0]["lr"] = lrnow

            y_pred = experience.values_np
            y_true = experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            losses.explained_variance = explained_var
            data.epoch += 1

            done_training = data.global_step >= config.total_timesteps
            # TODO: beter way to get episode return update without clogging dashboard
            # TODO: make this appear faster
            if profile.update(data):
                mean_and_log(data)

                if len(data.msg):
                    data.last_msg = data.msg
                if len(data.stats) > 0:
                    data.last_stats = data.stats

                if config.dashboard:
                    print_dashboard(config.env, data.utilization, data.global_step, data.epoch,
                        profile, data.losses, data.last_stats, data.last_msg)

                elif data.msg:
                    print(data.global_step, data.msg)

                data.msg = ""
                data.stats = defaultdict(list)

            if data.epoch % config.checkpoint_interval == 0 or done_training:
                model_path = save_checkpoint(data)
                data.msg = f'Checkpoint saved {model_path}'

    def mean_and_log(self):
        for k in list(data.stats.keys()):
            v = data.stats[k]
            try:
                v = np.mean(v)
            except:
                del data.stats[k]

            data.stats[k] = v

        if data.wandb is None:
            return

        data.last_log_time = time.time()
        data.wandb.log({
            '0verview/SPS': data.profile.SPS,
            '0verview/agent_steps': data.global_step,
            '0verview/epoch': data.epoch,
            '0verview/learning_rate': data.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in data.stats.items()},
            **{f'losses/{k}': v for k, v in data.losses.items()},
            **{f'performance/{k}': v for k, v in data.profile},
        })

    def close(self):
        data.vecenv.close()
        data.utilization.stop()
        if data.wandb is not None:
            data.wandb.finish()


    def make_losses(self):
        return pufferlib.namespace(
            policy_loss=0,
            value_loss=0,
            entropy=0,
            old_approx_kl=0,
            approx_kl=0,
            clipfrac=0,
            explained_variance=0,
        )

    def count_params(policy):
        return sum(p.numel() for p in policy.parameters() if p.requires_grad)

    def rollout(self, cfg: OmegaConf, env_creator, env_kwargs, agent_creator, agent_kwargs,
            backend, render_mode='auto', model_path=None, device='cuda'):

        if render_mode != 'auto':
            env_kwargs['render_mode'] = render_mode

        # We are just using Serial vecenv to give a consistent
        # single-agent/multi-agent API for evaluation
        env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs, backend=backend)

        if model_path is None:
            agent = agent_creator(env.driver_env, agent_kwargs).to(device)
        else:
            agent = torch.load(model_path, map_location=device)

        ob, info = env.reset()
        driver = env.driver_env
        os.system('clear')
        state = None

        frames = []
        tick = 0
        while tick <= 1000:
            if tick % 1 == 0:
                render = driver.render()
                if driver.render_mode == 'ansi':
                    print('\033[0;0H' + render + '\n')
                    time.sleep(0.05)
                elif driver.render_mode == 'rgb_array':
                    frames.append(render)
                    import cv2
                    render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                    cv2.imshow('frame', render)
                    cv2.waitKey(1)
                    time.sleep(1/24)
                elif driver.render_mode in ('human', 'raylib') and render is not None:
                    frames.append(render)

            with torch.no_grad():
                ob = torch.as_tensor(ob).to(device)
                if hasattr(agent, 'lstm'):
                    action, _, _, _, state = agent(ob, state)
                else:
                    action, _, _, _ = agent(ob)

                action = action.cpu().numpy().reshape(env.action_space.shape)

            ob, reward = env.step(action)[:2]
            reward = reward.mean()
            if tick % 128 == 0:
                print(f'Reward: {reward:.4f}, Tick: {tick}')
            tick += 1

        # Save frames as gif
        #import imageio
        #imageio.mimsave('../docker/eval.gif', frames, fps=15, loop=0)

