from __future__ import annotations

import logging
import math
import multiprocessing
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from cogames import serialization
from cogames.policy import TrainablePolicy
from mettagrid import MettaGridConfig

from cogames.env import HierarchicalActionMettaGridEnv, make_hierarchical_env

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = logging.getLogger("cogames.pufferlib")


class EnvConfigIterator:
    """Cycles through MettaGridConfigs, cloning for isolation when reused."""

    def __init__(self, configs: Sequence[MettaGridConfig]) -> None:
        if not configs:
            msg = "At least one MettaGridConfig is required to create environments."
            raise ValueError(msg)
        self._configs = tuple(cfg.model_copy(deep=True) for cfg in configs)
        self._next = 0

    def take(self, seed: Optional[int] = None) -> MettaGridConfig:
        if seed is not None:
            index = seed % len(self._configs)
        else:
            index = self._next
            self._next = (self._next + 1) % len(self._configs)
        return self._configs[index].model_copy(deep=True)


def _normalize_env_configs(
    env_cfgs: Sequence[MettaGridConfig] | MettaGridConfig | None,
    env_cfg: Optional[MettaGridConfig],
) -> Tuple[MettaGridConfig, ...]:
    if env_cfgs is not None:
        if isinstance(env_cfgs, MettaGridConfig):
            candidates: Iterable[MettaGridConfig] = (env_cfgs,)
        else:
            candidates = env_cfgs
    elif env_cfg is not None:
        candidates = (env_cfg,)
    else:
        raise ValueError("Either env_cfgs or env_cfg must be provided to train a policy")

    normalized = tuple(cfg for cfg in candidates)
    if not normalized:
        raise ValueError("No MettaGridConfig instances provided for training")
    return normalized


def _cpu_core_count() -> Optional[int]:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    except Exception:  # pragma: no cover - best effort fallback
        return None


def train(
    env_cfgs: Sequence[MettaGridConfig] | MettaGridConfig | None = None,
    *,
    env_cfg: Optional[MettaGridConfig] = None,
    policy_class_path: str,
    device: "torch.device",
    initial_weights_path: Optional[Path | str],
    num_steps: int,
    checkpoints_path: Path,
    seed: int,
    batch_size: int,
    minibatch_size: int,
    num_envs: int = 1,
    num_workers: Optional[int] = None,
    use_rnn: bool = False,
    checkpoint_interval: int = 200,
    vector_backend: str = "multiprocessing",
    game_name: Optional[str] = None,
    vector_num_envs: Optional[int] = None,
    vector_batch_size: Optional[int] = None,
    vector_num_workers: Optional[int] = None,
) -> None:
    import torch
    import torch.distributed

    import pufferlib.pytorch  # noqa: F401 - ensure modules register with torch
    import pufferlib.vector
    from pufferlib import pufferl
    from pufferlib.pufferlib import set_buffers

    checkpoints_path.mkdir(parents=True, exist_ok=True)

    env_sequence = _normalize_env_configs(env_cfgs, env_cfg)
    cfg_iterator = EnvConfigIterator(env_sequence)

    backend_options = {
        "multiprocessing": pufferlib.vector.Multiprocessing,
        "serial": pufferlib.vector.Serial,
        "ray": getattr(pufferlib.vector, "Ray", pufferlib.vector.Multiprocessing),
    }
    backend_key = vector_backend.lower()
    try:
        backend = backend_options[backend_key]
    except KeyError as exc:  # pragma: no cover - guarded by CLI validation
        raise ValueError(f"Unsupported vector backend: {vector_backend}") from exc

    if platform.system() == "Darwin" and backend is pufferlib.vector.Multiprocessing:
        multiprocessing.set_start_method("spawn", force=True)
        backend = pufferlib.vector.Serial

    requested_envs = vector_num_envs if vector_num_envs is not None else num_envs
    requested_envs = max(1, requested_envs)

    requested_workers = vector_num_workers if vector_num_workers is not None else num_workers
    default_workers = requested_workers if requested_workers is not None else 8

    cpu_cores = _cpu_core_count()
    if cpu_cores is not None:
        default_workers = min(default_workers, max(1, cpu_cores))

    if backend is pufferlib.vector.Multiprocessing and device.type != "cuda":
        backend = pufferlib.vector.Serial

    effective_num_workers = requested_workers if requested_workers is not None else default_workers
    if backend is pufferlib.vector.Serial:
        effective_num_workers = 1
    effective_num_workers = max(1, min(effective_num_workers, requested_envs))

    envs_per_worker = max(1, math.ceil(requested_envs / effective_num_workers))
    base_batch_size = vector_batch_size if vector_batch_size is not None else 128
    vector_batch = max(base_batch_size, requested_envs, envs_per_worker)
    remainder = vector_batch % envs_per_worker
    if remainder:
        vector_batch += envs_per_worker - remainder

    if backend is pufferlib.vector.Serial:
        vector_batch = max(requested_envs, envs_per_worker)
    elif backend is pufferlib.vector.Multiprocessing:
        # PufferLib's zero-copy path requires num_envs to be divisible by batch_size.
        # For modest env counts (our default heuristics pick 8 envs), the
        # 128-sample fallback would violate that constraint and trigger an
        # APIUsageError. Keep the batch aligned with the env count unless the
        # caller explicitly overrides vector_batch_size.
        if vector_batch > requested_envs or requested_envs % vector_batch != 0:
            vector_batch = requested_envs

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        seed += local_rank

    def env_creator(
        buf: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> HierarchicalActionMettaGridEnv:
        cfg = cfg_iterator.take(seed=seed)
        env = make_hierarchical_env(cfg, buf=buf)
        set_buffers(env, buf)
        return env

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=requested_envs,
        num_workers=effective_num_workers,
        batch_size=vector_batch,
        backend=backend,
    )

    weights_path: Optional[Path]
    if initial_weights_path is None:
        weights_path = None
    else:
        weights_path = Path(initial_weights_path)

    artifact = serialization.PolicyArtifact(
        policy_class=policy_class_path,
        weights_path=weights_path,
    )
    policy = serialization.load_policy(artifact, vecenv.driver_env, device)
    assert isinstance(policy, TrainablePolicy), (
        f"Policy class {policy_class_path} must implement TrainablePolicy interface"
    )

    auto_use_rnn = "lstm" in policy_class_path.lower() or "rnn" in policy_class_path.lower()
    use_rnn = use_rnn or auto_use_rnn

    env_name = "cogames.cogs_vs_clips"

    if use_rnn:
        learning_rate = 0.0003
        bptt_horizon = 1
        optimizer = "adam"
        adam_eps = 1e-8
        logger.info("Using RNN-specific hyperparameters: lr=0.0003, bptt=1, optimizer=adam")
    else:
        learning_rate = 0.015
        bptt_horizon = 1
        optimizer = "muon"
        adam_eps = 1e-12

    total_agents = max(1, getattr(vecenv, "num_agents", 1))
    realised_envs = max(1, getattr(vecenv, "num_envs", requested_envs))
    realised_workers = max(1, getattr(vecenv, "num_workers", effective_num_workers))
    realised_envs_per_worker = max(1, realised_envs // realised_workers)

    original_batch_size = batch_size
    amended_batch_size = max(original_batch_size, total_agents * bptt_horizon)
    remainder = amended_batch_size % realised_envs_per_worker
    if remainder:
        amended_batch_size += realised_envs_per_worker - remainder
    if amended_batch_size != original_batch_size:
        logger.info(
            "Adjusted batch_size from %s to %s (agents=%s, horizon=%s, envs/worker=%s)",
            original_batch_size,
            amended_batch_size,
            total_agents,
            bptt_horizon,
            realised_envs_per_worker,
        )

    amended_minibatch_size = min(minibatch_size, amended_batch_size)
    if amended_minibatch_size != minibatch_size:
        logger.info(
            "Reducing minibatch_size from %s to %s to keep it <= batch_size",
            minibatch_size,
            amended_minibatch_size,
        )

    effective_timesteps = max(num_steps, amended_batch_size)
    if effective_timesteps != num_steps:
        logger.info(
            "Raising total_timesteps from %s to %s to keep it >= batch_size",
            num_steps,
            effective_timesteps,
        )

    train_args = dict(
        env=env_name,
        device=device.type,
        total_timesteps=effective_timesteps,
        minibatch_size=amended_minibatch_size,
        batch_size=amended_batch_size,
        data_dir=str(checkpoints_path),
        checkpoint_interval=checkpoint_interval,
        bptt_horizon=bptt_horizon,
        seed=seed,
        use_rnn=use_rnn,
        torch_deterministic=True,
        cpu_offload=False,
        optimizer=optimizer,
        anneal_lr=True,
        precision="float32",
        learning_rate=learning_rate,
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
        adam_eps=adam_eps,
        max_minibatch_size=32768,
        compile=False,
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
        prio_alpha=0.8,
        prio_beta0=0.2,
    )

    trainer = pufferl.PuffeRL(train_args, vecenv, policy.network())
    training_diverged = False

    while trainer.global_step < num_steps:
        trainer.evaluate()
        trainer.train()

        rewards_tensor = getattr(trainer, "rewards", None)
        if rewards_tensor is None:
            continue

        if hasattr(rewards_tensor, "detach") and hasattr(rewards_tensor, "numel"):
            if rewards_tensor.numel() == 0:
                continue
            rewards_array = rewards_tensor.detach().cpu().float().numpy()
        else:
            rewards_array = np.asarray(rewards_tensor, dtype=np.float32)
            if rewards_array.size == 0:
                continue

        trainer.stats["reward_mean"] = float(np.mean(rewards_array))
        trainer.stats["reward_std"] = float(np.std(rewards_array, ddof=0))
        trainer.stats["reward_sum"] = float(np.sum(rewards_array))

        network: nn.Module = policy.network()
        has_nan = False
        for name, param in network.named_parameters():
            if param.grad is not None and not param.grad.isfinite().all():
                logger.error("NaN/Inf detected in gradients for parameter: %s", name)
                has_nan = True
            if not param.isfinite().all():
                logger.error("NaN/Inf detected in parameter: %s", name)
                has_nan = True

        if has_nan:
            logger.error(
                "Training diverged at step %s! Stopping early to prevent saving corrupted checkpoint.",
                trainer.global_step,
            )
            training_diverged = True
            break

    trainer.print_dashboard()
    trainer.close()

    from rich.console import Console

    console = Console()

    console.print()
    if training_diverged:
        console.print("=" * 80, style="bold red")
        console.print("Training diverged (NaN detected)! Stopped early.", style="bold red")
        console.print(f"Checkpoints saved to: [cyan]{checkpoints_path}[/cyan]", style="bold red")
        console.print("=" * 80, style="bold red")
        console.print()
        console.print("[yellow]Warning: The latest checkpoint may contain NaN values.[/yellow]")
        console.print("[yellow]Try using an earlier checkpoint or retraining with lower learning rate.[/yellow]")
    else:
        console.print("=" * 80, style="bold green")
        console.print(
            f"Training complete. Checkpoints saved to: [cyan]{checkpoints_path}[/cyan]",
            style="bold green",
        )
        console.print("=" * 80, style="bold green")

    checkpoint_dir = checkpoints_path / env_name
    checkpoints = []

    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))

    if not checkpoints and checkpoints_path.exists():
        checkpoints = sorted(checkpoints_path.glob("*.pt"))

    console.print()
    if checkpoints and not training_diverged:
        final_checkpoint = checkpoints[-1]
        console.print(f"Final checkpoint: [cyan]{final_checkpoint}[/cyan]")
        policy_shorthand = {
            "cogames.policy.random.RandomPolicy": "random",
            "cogames.policy.simple.SimplePolicy": "simple",
            "cogames.policy.lstm.LSTMPolicy": "lstm",
        }.get(policy_class_path)
        game_arg = f" {game_name}" if game_name else ""
        policy_arg = policy_shorthand if policy_shorthand else policy_class_path
        console.print()
        console.print("To play with this policy:", style="bold")
        console.print(
            f"  [yellow]cogames play{game_arg} --policy {policy_arg} --policy-data {final_checkpoint}[/yellow]"
        )
    elif checkpoints and training_diverged:
        console.print(f"[yellow]Found {len(checkpoints)} checkpoint(s). The most recent may be corrupted.[/yellow]")
        console.print("[yellow]Try using an earlier checkpoint or retraining.[/yellow]")
    else:
        console.print(f"[yellow]No checkpoint files found. Check {checkpoints_path} for saved models.[/yellow]")

    console.print("=" * 80, style="bold green")
    console.print()
