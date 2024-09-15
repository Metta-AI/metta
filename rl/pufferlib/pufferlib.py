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
import numpy as np
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.play import play
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
        self.policy = None

    def train(self, load_checkpoint=True):
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
        if load_checkpoint:
            clean_pufferl.try_load_checkpoint(self.data)

        print(f"Starting training: {self.data.global_step}/{pcfg.train.total_timesteps} timesteps")

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

        print("Training complete. Evaluating final model...")
        clean_pufferl.evaluate(self.data)
        self.process_stats(self.data)
        self.train_time = time.time() - self._train_start

    def process_stats(self, data):
        if len(data.stats) == 0:
            return
        self.last_stats = data.stats

    def close(self):
        clean_pufferl.close(self.data)

    def evaluate(self):
        vecenv = self._make_vecenv(num_envs=self.cfg.eval.num_envs)
        return evaluate(self.cfg, vecenv)

    def play(self):
        vecenv = self._make_vecenv(num_envs=1, render_mode="human")
        return play(self.cfg, vecenv)

    def _make_vecenv(self, num_envs=1, batch_size=None, num_workers=1, render_mode=None, **kwargs):
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
            env_kwargs=dict(cfg = dict(**self.cfg.env), render_mode=render_mode),
            num_envs=num_envs,
            num_workers=num_workers,
            batch_size=batch_size or num_envs,
            backend=vec,
            **kwargs
        )
        print("Vectorization Settings: ", vecenv_args)
        return pufferlib.vector.make(make_env_func, **vecenv_args)
