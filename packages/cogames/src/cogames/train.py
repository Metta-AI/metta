import logging
import multiprocessing
import platform
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

import pufferlib.pytorch
import pufferlib.vector
from cogames.policy import TrainablePolicy
from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.module import load_symbol
from pufferlib import pufferl
from pufferlib.pufferlib import set_buffers

logger = logging.getLogger("cogames.pufferlib")


class EnvConfigIterator:
    """Selects configs for vector environments, cycling when needed."""

    def __init__(self, configs: Sequence[MettaGridConfig]) -> None:
        if not configs:
            msg = "At least one MettaGridConfig is required to create environments."
            raise ValueError(msg)
        # Work with deep copies to avoid shared state across workers
        self._configs = tuple(cfg.model_copy(deep=True) for cfg in configs)
        self._next_index = 0

    def take(self, seed: Optional[int] = None) -> MettaGridConfig:
        if seed is not None:
            index = seed % len(self._configs)
            cfg = self._configs[index].model_copy(deep=True)
            return cfg
        index = self._next_index
        self._next_index = (self._next_index + 1) % len(self._configs)
        cfg = self._configs[index].model_copy(deep=True)
        return cfg


def env_creator(
    cfg_iterator: EnvConfigIterator,
    buf: Optional[Any] = None,
    seed: Optional[int] = None,
):
    cfg = cfg_iterator.take(seed=seed)
    env = MettaGridEnv(env_cfg=cfg)
    set_buffers(env, buf)
    return env


def train(
    env_cfgs: Sequence[MettaGridConfig],
    policy_class_path: str,
    device: torch.device,
    initial_weights_path: Optional[Path],
    num_steps: int,
    checkpoints_path: Path,
    seed: int,
    batch_size: int,
    minibatch_size: int,
    num_envs: int,
    num_workers: int,
    use_rnn: bool,
    checkpoint_interval: int,
    vector_backend: str,
):
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    cfg_iterator = EnvConfigIterator(env_cfgs)

    backend_options = {
        "multiprocessing": pufferlib.vector.Multiprocessing,
        "serial": pufferlib.vector.Serial,
        "ray": getattr(pufferlib.vector, "Ray", pufferlib.vector.Multiprocessing),
    }

    backend = backend_options[vector_backend]

    if platform.system() == "Darwin" and backend is pufferlib.vector.Multiprocessing:
        multiprocessing.set_start_method("spawn", force=True)
        backend = pufferlib.vector.Serial

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=num_envs,
        backend=backend,
        env_kwargs={
            "cfg_iterator": cfg_iterator,
        },
    )

    policy_class = load_symbol(policy_class_path)
    policy = policy_class(vecenv.driver_env, device)
    assert isinstance(policy, TrainablePolicy)
    if initial_weights_path:
        policy.load_checkpoint(str(initial_weights_path))

    env_name = "cogames.cogs_vs_clips"

    train_args = dict(
        env=env_name,
        device=device.type,
        total_timesteps=num_steps,
        minibatch_size=minibatch_size,
        batch_size=batch_size,
        data_dir=str(checkpoints_path),
        checkpoint_interval=checkpoint_interval,
        bptt_horizon=16 if use_rnn else 1,
        seed=seed,
        use_rnn=use_rnn,
        # Defaults
        torch_deterministic=True,
        cpu_offload=False,
        optimizer="muon",
        anneal_lr=True,
        precision="float32",
        learning_rate=0.015,
        gamma=0.995,
        gae_lambda=0.90,
        update_epochs=1,
        clip_coef=0.2,
        vf_coef=2.0,
        vf_clip_coef=0.2,
        max_grad_norm=1.5,
        ent_coef=0.001,
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_eps=1e-12,
        max_minibatch_size=32768,
        compile=False,
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
        prio_alpha=0.8,
        prio_beta0=0.2,
    )

    trainer = pufferl.PuffeRL(train_args, vecenv, policy.network())

    while trainer.global_step < num_steps:
        trainer.evaluate()
        trainer.train()

    trainer.print_dashboard()
    trainer.close()
