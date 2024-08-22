from typing import Dict
from omegaconf import OmegaConf
from rl.rl_framework import RLFramework
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl
import hydra
import time
from rich.console import Console

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
        self.wandb = None

    def train(self):
        pcfg = self.puffer_cfg
        vec = pcfg.vectorization
        if vec == 'serial':
            vec = pufferlib.vector.Serial
        elif vec == 'multiprocessing':
            vec = pufferlib.vector.Multiprocessing
        elif vec == 'ray':
            vec = pufferlib.vector.Ray
        else:
            raise ValueError('Invalid --vector (serial/multiprocessing/ray).')

        target_batch_size = pcfg.train.forward_pass_minibatch_target_size // self.cfg.env.game.num_agents
        if target_batch_size < 2: # pufferlib bug requires batch size >= 2
            target_batch_size = 2
        batch_size = (target_batch_size // pcfg.train.num_workers) * pcfg.train.num_workers

        vecenv_args = dict(
        env_kwargs=dict(cfg = dict(**self.cfg.env)),
            num_envs=batch_size * pcfg.train.async_factor,
            num_workers=pcfg.train.num_workers,
            batch_size=batch_size,
            zero_copy=pcfg.train.zero_copy,
            backend=vec,
        )
        print("Vectorization Settings: ", vecenv_args)
        vecenv = pufferlib.vector.make(make_env_func, **vecenv_args)
        policy = puffer_agent_wrapper.make_policy(vecenv.driver_env, self.cfg)
        data = clean_pufferl.create(pcfg.train, vecenv, policy, wandb=self.wandb)

        while data.global_step < pcfg.train.total_timesteps:
            try:
                clean_pufferl.evaluate(data)
                self.process_stats(data)
                clean_pufferl.train(data)
            except KeyboardInterrupt:
                clean_pufferl.close(data)
                os._exit(0)
            except Exception:
                Console().print_exception()
                os._exit(0)

        clean_pufferl.evaluate(data)
        self.process_stats(data)
        clean_pufferl.close(data)
        self.train_time = time.time() - self._train_start

    def evaluate(self):
        # model_dir = os.path.join(self.puffer_cfg.train_dir, self.cfg.experiment)
        # latest_model_cp = [p for p in os.listdir(model_dir) if p.endswith(".pt")][-1]
        # print(f"Loading model from {latest_model_cp}")

        result = clean_pufferl.rollout(
            self.cfg,
            make_env_func,
            env_kwargs=dict(cfg = dict(**self.cfg.env)),
            agent_creator=puffer_agent_wrapper.make_policy,
            agent_kwargs=self.cfg,
            render_mode=self.puffer_cfg.render_mode,
            device=self.puffer_cfg.device,
            backend=pufferlib.vector.Serial,
            # model_path=os.path.join(model_dir, latest_model_cp),
        )
        # return EvaluationResult(
        #     reward=result['reward'],
        #     frames=result['frames']
        # )

    def process_stats(self, data):
        self.stats = data.stats
        # new_stats = {}
        # for k, v in data.stats.items():
        #     new_stats["avg_" + k] = v
        # if "episode_return" in data.stats:
        #     new_stats["episode_return"] = data.stats["episode_return"]
        #     new_stats["episode_length"] = data.stats["episode_length"]
        # data.stats = new_stats


