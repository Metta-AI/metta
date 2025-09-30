from __future__ import annotations

import logging
import multiprocessing
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from cogames.policy import TrainablePolicy
from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    import torch

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


def train(
    env_cfgs: Sequence[MettaGridConfig] | MettaGridConfig | None = None,
    *,
    policy_class_path: str,
    device: "torch.device",
    initial_weights_path: Optional[Path],
    num_steps: int,
    checkpoints_path: Path,
    seed: int,
    batch_size: int,
    minibatch_size: int,
    num_envs: int = 1,
    num_workers: int = 1,
    use_rnn: bool = False,
    checkpoint_interval: int = 200,
    vector_backend: str = "multiprocessing",
    game_name: Optional[str] = None,
    env_cfg: Optional[MettaGridConfig] = None,
) -> None:
    import torch
    import torch.distributed

    import pufferlib.pytorch  # noqa: F401 - ensure modules register with torch
    import pufferlib.vector
    from pufferlib import pufferl
    from pufferlib.pufferlib import set_buffers

    checkpoints_path.mkdir(parents=True, exist_ok=True)

    if env_cfgs is not None and isinstance(env_cfgs, MettaGridConfig):
        env_sequence: Sequence[MettaGridConfig] = (env_cfgs,)
    elif env_cfgs is not None:
        env_sequence = tuple(env_cfgs)
    elif env_cfg is not None:
        env_sequence = (env_cfg,)
    else:
        raise ValueError("Either env_cfgs or env_cfg must be provided to train a policy")

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

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        seed = seed + local_rank

    def env_creator(buf: Optional[Any] = None, seed: Optional[int] = None) -> MettaGridEnv:
        cfg = cfg_iterator.take(seed=seed)
        env = MettaGridEnv(env_cfg=cfg)
        set_buffers(env, buf)
        return env

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=num_envs,
        backend=backend,
    )

    # Load the TrainablePolicy class using the new API
    policy_class = load_symbol(policy_class_path)
    policy = policy_class(vecenv.driver_env, device)

    # Ensure it implements the TrainablePolicy interface
    assert isinstance(policy, TrainablePolicy), (
        f"Policy class {policy_class_path} must implement TrainablePolicy interface"
    )

    # Load initial weights if provided
    if initial_weights_path:
        policy.load_policy_data(str(initial_weights_path))

    auto_use_rnn = "lstm" in policy_class_path.lower() or "rnn" in policy_class_path.lower()
    use_rnn = use_rnn or auto_use_rnn

    env_name = "cogames.cogs_vs_clips"

    # Use RNN-specific hyperparameters if needed
    if use_rnn:
        learning_rate = 0.0003  # Much lower LR for RNN stability
        bptt_horizon = 1  # Use bptt=1 for now (TODO: fix bptt>1 observation reshaping)
        optimizer = "adam"  # Adam is more stable for RNNs than Muon
        adam_eps = 1e-8  # Standard eps value, not too small
        logger.info("Using RNN-specific hyperparameters: lr=0.0003, bptt=1, optimizer=adam")
    else:
        learning_rate = 0.015
        bptt_horizon = 1
        optimizer = "muon"
        adam_eps = 1e-12

    train_args = dict(
        env=env_name,
        device=device.type,
        total_timesteps=num_steps,
        minibatch_size=minibatch_size,
        batch_size=batch_size,
        data_dir=str(checkpoints_path),
        checkpoint_interval=checkpoint_interval,
        bptt_horizon=bptt_horizon,
        seed=seed,
        use_rnn=use_rnn,
        # Defaults
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

    # Pass the neural network from TrainablePolicy to PuffeRL for training
    # The network() method is part of the new TrainablePolicy API
    trainer = pufferl.PuffeRL(train_args, vecenv, policy.network())

    # Track if training diverged
    training_diverged = False

    while trainer.global_step < num_steps:
        trainer.evaluate()
        trainer.train()
        rewards_tensor = getattr(trainer, "rewards", None)
        if rewards_tensor is None:
            continue
        if not hasattr(rewards_tensor, "numel") or rewards_tensor.numel() == 0:
            continue

        rewards = rewards_tensor.detach() if hasattr(rewards_tensor, "detach") else torch.as_tensor(rewards_tensor)
        trainer.stats["reward_mean"] = float(rewards.mean().item())
        trainer.stats["reward_std"] = float(rewards.std(unbiased=False).item())
        trainer.stats["reward_sum"] = float(rewards.sum().item())

        # Check for NaN in network parameters after each training step
        network = policy.network()
        has_nan = False
        for name, param in network.named_parameters():
            if param.grad is not None and not param.grad.isfinite().all():
                logger.error(f"NaN/Inf detected in gradients for parameter: {name}")
                has_nan = True
            if not param.isfinite().all():
                logger.error(f"NaN/Inf detected in parameter: {name}")
                has_nan = True

        if has_nan:
            logger.error(
                f"Training diverged at step {trainer.global_step}! "
                "Stopping early to prevent saving corrupted checkpoint."
            )
            training_diverged = True
            break

    trainer.print_dashboard()
    trainer.close()

    # Print checkpoint path and usage commands with colored output
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

    # Try to find the final checkpoint
    # PufferLib saves checkpoints in data_dir/env_name/
    checkpoint_dir = checkpoints_path / env_name
    checkpoints = []

    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))

    # Fallback: also check directly in checkpoints_path
    if not checkpoints and checkpoints_path.exists():
        checkpoints = sorted(checkpoints_path.glob("*.pt"))

    if checkpoints and not training_diverged:
        final_checkpoint = checkpoints[-1]
        console.print()
        console.print(f"Final checkpoint: [cyan]{final_checkpoint}[/cyan]")

        # Show shorthand version if available
        policy_shorthand = {
            "cogames.policy.random.RandomPolicy": "random",
            "cogames.policy.simple.SimplePolicy": "simple",
            "cogames.policy.lstm.LSTMPolicy": "lstm",
        }.get(policy_class_path)

        # Build the command with game name if provided
        game_arg = f" {game_name}" if game_name else ""
        policy_arg = policy_shorthand if policy_shorthand else policy_class_path

        console.print()
        console.print("To play with this policy:", style="bold")
        console.print(
            f"  [yellow]cogames play{game_arg} --policy {policy_arg} --policy-data {final_checkpoint}[/yellow]"
        )
    elif checkpoints and training_diverged:
        console.print()
        console.print(f"[yellow]Found {len(checkpoints)} checkpoint(s). The most recent may be corrupted.[/yellow]")
        console.print("[yellow]Try using an earlier checkpoint or retraining.[/yellow]")
    else:
        console.print()
        console.print(f"[yellow]No checkpoint files found. Check {checkpoints_path} for saved models.[/yellow]")

    console.print("=" * 80, style="bold green")
    console.print()
