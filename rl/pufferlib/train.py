from omegaconf import OmegaConf
import os
import time
from rich.console import Console
from . import puffer_agent_wrapper
from rl.pufferlib.vecenv import make_vecenv

from . import clean_pufferl

def train(cfg: OmegaConf, load_checkpoint: bool = True):
    train_start = time.time()
    pcfg = cfg.framework.pufferlib
    target_batch_size = pcfg.train.forward_pass_minibatch_target_size // cfg.env.game.num_agents
    if target_batch_size < 2: # pufferlib bug requires batch size >= 2
        target_batch_size = 2
    batch_size = (target_batch_size // pcfg.train.num_workers) * pcfg.train.num_workers

    vecenv = make_vecenv(
        cfg,
        num_envs = batch_size * pcfg.train.async_factor,
        batch_size = batch_size,
        num_workers=pcfg.train.num_workers,
        zero_copy=pcfg.train.zero_copy)

    policy = puffer_agent_wrapper.make_policy(vecenv.driver_env, cfg)
    data = clean_pufferl.create(pcfg.train, vecenv, policy)
    if load_checkpoint:
        clean_pufferl.try_load_checkpoint(data)

    print(f"Starting training: {data.global_step}/{pcfg.train.total_timesteps} timesteps")

    while data.global_step < pcfg.train.total_timesteps:
        try:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)
        except KeyboardInterrupt:
            clean_pufferl.close(data)
            os._exit(0)
        except Exception:
            Console().print_exception()
            os._exit(0)

    print("Training complete. Evaluating final model...")
    clean_pufferl.evaluate(data)
    train_time = time.time() - train_start
    clean_pufferl.close(data)
    return data.last_stats, train_time

