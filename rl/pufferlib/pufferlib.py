from math import inf
from pyexpat import model
from typing import Dict
from omegaconf import OmegaConf
from rl.rl_framework import RLFramework
import os
import torch
import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl
import hydra
import time
from rich.console import Console
import numpy as np

from . import puffer_agent_wrapper

from . import clean_pufferl

def make_env_func(cfg: OmegaConf, render_mode='rgb_array'):
    env = hydra.utils.instantiate(cfg, render_mode=render_mode)
    env.emulated = None
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.num_agents = env.player_count

    return env

class PufferLibFramework(RLFramework):
    def __init__(self, cfg: Dict, **puffer_args):
        cfg = OmegaConf.create(cfg)
        super().__init__(cfg)
        self.puffer_cfg = cfg.framework.pufferlib

        self._train_start = time.time()

    def train(self):
        pcfg = self.puffer_cfg
        target_batch_size = pcfg.train.forward_pass_minibatch_target_size // self.cfg.env.game.num_agents
        if target_batch_size < 2: # pufferlib bug requires batch size >= 2
            target_batch_size = 2
        batch_size = (target_batch_size // pcfg.train.num_workers) * pcfg.train.num_workers
        vecenv = self._make_vecenv(
            num_envs = batch_size * pcfg.train.async_factor,
            batch_size = batch_size,
            num_workers=pcfg.train.num_workers,
            zero_copy=pcfg.train.zero_copy)

        policy = puffer_agent_wrapper.make_policy(vecenv.driver_env, self.cfg)
        self.data = clean_pufferl.create(pcfg.train, vecenv, policy)
        clean_pufferl.try_load_checkpoint(self.data)
        while self.data.global_step < pcfg.train.total_timesteps:
            try:
                clean_pufferl.evaluate(self.data)
                self.process_stats(self.data)
                clean_pufferl.train(self.data)
            except KeyboardInterrupt:
                self.close()
                os._exit(0)
            except Exception:
                Console().print_exception()
                os._exit(0)

        clean_pufferl.evaluate(self.data)
        self.process_stats(self.data)
        self.train_time = time.time() - self._train_start

    def process_stats(self, data):
        if len(data.stats) == 0:
            return
        new_stats = {}
        for k, v in data.stats.items():
            new_stats[k] = np.array(v).mean()
        self.last_stats = new_stats

    def close(self):
        clean_pufferl.close(self.data)

    def evaluate(self):
        num_envs = self.cfg.eval.num_envs
        device = self.cfg.framework.pufferlib.device
        vecenv = self._make_vecenv(num_envs=num_envs)

        run_path = os.path.join(self.cfg.framework.pufferlib.train_dir, self.cfg.experiment)
        trainer_state = torch.load(os.path.join(run_path, 'trainer_state.pt'))
        model_path = os.path.join(run_path, trainer_state["model_name"])
        print(f'Loaded model from {model_path}')
        policy = torch.load(model_path, map_location=device)
        opponents = [policy]

        # paths = glob.glob(f'{checkpoint_dir}/model_*.pt', recursive=True)
        # names = [path.split('/')[-1] for path in paths]
        # print(f'Loaded {len(paths)} models')
        # paths.remove(f'{checkpoint_dir}/{checkpoint}')
        # print(f'Removed {checkpoint} from paths')
        # elos[checkpoint] = 1000

        # Sample with replacement if not enough models
        # print(f'Sampling {num_opponents} opponents')
        # n_models = len(paths)
        # if n_models < num_opponents:
        #     idxs = random.choices(range(n_models), k=num_opponents)
        # else:
        #     idxs = random.sample(range(n_models), num_opponents)
        # print(f'Sampled {num_opponents} opponents')

        # opponent_names = [names[i] for i in idxs]
        # opponents = [torch.load(paths[i], map_location='cuda') for i in idxs]
        # print(f'Loaded {num_opponents} opponents')
        obs, _ = vecenv.reset()

        num_opponents = len(opponents)
        envs_per_opponent = num_envs // num_opponents
        my_state = None
        opp_states = [None for _ in range(num_opponents)]

        num_agents = self.cfg.env.game.num_agents
        num_my_agents = max(1, int(self.cfg.env.game.num_agents * self.cfg.eval.policy_agents_pct))
        num_opponent_agents = num_agents - num_my_agents
        print(f'Policy Agents: {num_my_agents}, Opponent Agents: {num_opponent_agents}')
        slice_idxs = torch.arange(vecenv.num_agents).reshape(num_envs, num_agents).to(device=device)
        my_idxs = slice_idxs[:, :num_my_agents].reshape(vecenv.num_agents//2)
        opp_idxs = slice_idxs[:, num_my_agents:].reshape(num_envs*num_opponent_agents).split(num_opponent_agents*envs_per_opponent)

        start = time.time()
        episodes = 0
        step = 0
        scores = []
        total_rewards = np.zeros(vecenv.num_agents)
        while episodes < self.cfg.eval.num_episodes and time.time() - start < self.cfg.eval.max_time_s:
            step += 1
            opp_actions = []
            with torch.no_grad():
                obs = torch.as_tensor(obs).to(device=device)
                my_obs = obs[my_idxs]

                # Parallelize across opponents
                if hasattr(policy, 'lstm'):
                    my_actions, _, _, _, my_state = policy(my_obs, my_state)
                else:
                    my_actions, _, _, _ = policy(my_obs)

                # Iterate opponent policies
                for i in range(num_opponents):
                    opp_obs = obs[opp_idxs[i]]
                    opp_state = opp_states[i]

                    opponent = opponents[i]
                    if hasattr(policy, 'lstm'):
                        opp_atn, _, _, _, opp_states[i] = opponent(opp_obs, opp_state)
                    else:
                        opp_atn, _, _, _ = opponent(opp_obs)

                    opp_actions.append(opp_atn)

            opp_actions = torch.cat(opp_actions)
            actions = torch.cat([
                my_actions.view(num_envs, num_my_agents, -1),
                opp_actions.view(num_envs, num_opponent_agents, -1),
            ], dim=1).view(num_envs*num_agents, -1)

            obs, rewards, dones, truncated, infos = vecenv.step(actions.cpu().numpy())
            total_rewards += rewards
            episodes += sum([e.done for e in vecenv.envs])

            # for i in range(num_envs):
            #     c = envs.c_envs[i]
            #     opp_idx = i // envs_per_opponent
            #     if c.radiant_victories > prev_radiant_victories[i]:
            #         prev_radiant_victories[i] = c.radiant_victories
            #         scores.append((opp_idx, 1))
            #         games_played += 1
            #         print('Radiant Victory')
            #     elif c.dire_victories > prev_dire_victories[i]:
            #         prev_dire_victories[i] = c.dire_victories
            #         scores.append((opp_idx, 0))
            #         games_played += 1
            #         print('Dire Victory')

        return scores

    def _make_vecenv(self, num_envs=1, batch_size=None, num_workers=1, **kwargs):
        pcfg = self.puffer_cfg
        vec = pcfg.vectorization
        if vec == 'serial' or num_workers == 1:
            vec = pufferlib.vector.Serial
        elif vec == 'multiprocessing':
            vec = pufferlib.vector.Multiprocessing
        elif vec == 'ray':
            vec = pufferlib.vector.Ray
        else:
            raise ValueError('Invalid --vector (serial/multiprocessing/ray).')

        vecenv_args = dict(
            env_kwargs=dict(cfg = dict(**self.cfg.env)),
            num_envs=num_envs,
            num_workers=num_workers,
            batch_size=batch_size or num_envs,
            backend=vec,
            **kwargs
        )
        print("Vectorization Settings: ", vecenv_args)
        return pufferlib.vector.make(make_env_func, **vecenv_args)
