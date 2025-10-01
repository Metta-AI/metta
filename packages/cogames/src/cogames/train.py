from __future__ import annotations

import logging
import multiprocessing
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from cogames.policy import TrainablePolicy
from cogames.policy.loader import instantiate_or_load_policy
from mettagrid import MettaGridConfig, MettaGridEnv

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
    game_name: Optional[str] = None,
    *,
    vector_num_envs: Optional[int] = None,
    vector_batch_size: Optional[int] = None,
    vector_num_workers: Optional[int] = None,
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

    desired_workers = vector_num_workers if vector_num_workers is not None else 8
    cpu_cores = None
    try:
        import psutil

        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    except Exception:  # pragma: no cover - best effort fallback
        cpu_cores = None

    if cpu_cores is not None:
        adjusted_workers = min(desired_workers, max(1, cpu_cores))
        if adjusted_workers < desired_workers:
            logger.info(
                "Reducing num_workers from %s to %s to match available CPU cores",
                desired_workers,
                adjusted_workers,
            )
        num_workers = adjusted_workers
    else:
        num_workers = desired_workers

    if backend is pufferlib.vector.Multiprocessing and device.type != "cuda":
        backend = pufferlib.vector.Serial
        num_workers = 1

    num_envs = vector_num_envs if vector_num_envs is not None else 256

    envs_per_worker = max(1, num_envs // num_workers)
    base_batch_size = vector_batch_size if vector_batch_size is not None else 128
    vector_batch_size = max(base_batch_size, envs_per_worker)
    remainder = vector_batch_size % envs_per_worker
    if remainder:
        vector_batch_size += envs_per_worker - remainder

    logger.debug(
        "Vec env config: num_envs=%s, num_workers=%s, batch_size=%s (envs/worker=%s)",
        num_envs,
        num_workers,
        vector_batch_size,
        envs_per_worker,
    )

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=vector_batch_size,
        backend=backend,
        env_kwargs={
            "cfg": env_cfg,
        },
    )

    policy = instantiate_or_load_policy(policy_class_path, initial_weights_path, vecenv.driver_env, device)
    # Ensure it implements the TrainablePolicy interface
    assert isinstance(policy, TrainablePolicy), (
        f"Policy class {policy_class_path} must implement TrainablePolicy interface"
    )

    # Detect if policy uses RNN (e.g., LSTM)
    use_rnn = "lstm" in policy_class_path.lower() or "rnn" in policy_class_path.lower()

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

    total_agents = max(1, getattr(vecenv, "num_agents", 1))
    num_envs = max(1, getattr(vecenv, "num_envs", 1))
    num_workers = max(1, getattr(vecenv, "num_workers", 1))
    envs_per_worker = max(1, num_envs // num_workers)

    # PuffeRL enforces two simple rules:
    # 1. batch_size >= num_agents * bptt_horizon
    # 2. batch_size % (num_envs / num_workers) == 0
    original_batch_size = batch_size
    amended_batch_size = max(original_batch_size, total_agents * bptt_horizon)
    remainder = amended_batch_size % envs_per_worker
    if remainder:
        amended_batch_size += envs_per_worker - remainder

    if amended_batch_size != original_batch_size:
        logger.info(
            "Adjusted batch_size from %s to %s (agents=%s, horizon=%s, envs/worker=%s)",
            original_batch_size,
            amended_batch_size,
            total_agents,
            bptt_horizon,
            envs_per_worker,
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
        checkpoint_interval=200,
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
