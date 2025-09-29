from __future__ import annotations

import logging
import multiprocessing
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from cogames.policy import TrainablePolicy
from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("cogames.pufferlib")


def train(
    env_cfg: MettaGridConfig,
    policy_class_path: str,
    device: "torch.device",
    initial_weights_path: Optional[str],
    num_steps: int,
    checkpoints_path: Path,
    seed: int,
    batch_size: int,
    minibatch_size: int,
) -> None:
    import pufferlib.pytorch  # noqa: F401 - ensure modules register with torch
    import pufferlib.vector
    from pufferlib import pufferl
    from pufferlib.pufferlib import set_buffers

    def env_creator(cfg: MettaGridConfig, buf: Optional[Any] = None, seed: Optional[int] = None):
        env = MettaGridEnv(env_cfg=cfg)
        set_buffers(env, buf)
        return env

    backend = pufferlib.vector.Multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)
        # TODO(jsuarez): Fix multiprocessing backend
        backend = pufferlib.vector.Serial

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=256,
        num_workers=8,
        batch_size=128,
        backend=backend,
        env_kwargs={
            "cfg": env_cfg,
        },
    )

    policy_class = load_symbol(policy_class_path)
    policy = policy_class(vecenv.driver_env, device)
    assert isinstance(policy, TrainablePolicy)
    if initial_weights_path:
        policy.load_checkpoint(initial_weights_path)

    env_name = "cogames.cogs_vs_clips"

    train_args = dict(
        env=env_name,
        device=device.type,
        total_timesteps=num_steps,
        minibatch_size=minibatch_size,
        batch_size=batch_size,
        data_dir=str(checkpoints_path),
        checkpoint_interval=200,
        bptt_horizon=1,
        seed=seed,
        use_rnn=False,
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
