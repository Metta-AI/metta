from __future__ import annotations

import json
import logging
import multiprocessing
import platform
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np

from cogames.aws_storage import maybe_upload_checkpoint
from cogames.env import make_hierarchical_env
from cogames.policy import TrainablePolicy
from mettagrid import MettaGridConfig
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("cogames.pufferlib")


def train(
    env_cfg: Optional[MettaGridConfig],
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
    env_cfg_supplier: Optional[Callable[[], MettaGridConfig]] = None,
    logits_debug_path: Optional[Path] = None,
) -> None:
    import pufferlib.pytorch  # noqa: F401 - ensure modules register with torch
    import pufferlib.vector
    from pufferlib import pufferl
    from pufferlib.pufferlib import set_buffers

    if env_cfg_supplier is None and env_cfg is None:
        raise ValueError("Either env_cfg or env_cfg_supplier must be provided to train a policy")

    def _next_config() -> MettaGridConfig:
        if env_cfg_supplier is not None:
            cfg = env_cfg_supplier()
            if not isinstance(cfg, MettaGridConfig):
                raise TypeError("Curriculum callable must return a MettaGridConfig instance")
            return cfg.model_copy(deep=True)
        assert env_cfg is not None
        return env_cfg.model_copy(deep=True)

    def env_creator(buf: Optional[Any] = None, seed: Optional[int] = None):
        cfg = _next_config()
        env = make_hierarchical_env(cfg, buf=buf)
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
    )

    debug_file: Optional[IO[str]] = None
    instrumentation_env: Optional[Any] = None
    instrumentation_actions: Optional[np.ndarray] = None
    instrumentation_rollout_steps = 4

    if logits_debug_path is not None:
        import torch

        logits_debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = logits_debug_path.open("w", encoding="utf-8")
        instrumentation_cfg = vecenv.driver_env.mg_config.model_copy(deep=True)
        instrumentation_env = make_hierarchical_env(instrumentation_cfg)
        action_dim = int(instrumentation_env.single_action_space.nvec.size)
        instrumentation_actions = np.zeros((instrumentation_env.num_agents, action_dim), dtype=np.int32)

        def _write_logit_snapshot(global_step: int) -> None:
            if debug_file is None or instrumentation_env is None or instrumentation_actions is None:
                return

            obs, _ = instrumentation_env.reset(seed=seed + int(global_step))
            entry: dict[str, Any] = {"global_step": int(global_step), "steps": []}
            for rollout_step in range(instrumentation_rollout_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.uint8, device=device)
                network = policy.network()
                network.eval()
                with torch.no_grad():
                    logits_raw, _ = network.forward_eval(obs_tensor)

                logits_cpu = [tensor.detach().to("cpu") for tensor in logits_raw]
                step_dims: list[dict[str, Union[float, int]]] = []
                for dim_index, logits in enumerate(logits_cpu):
                    stats: dict[str, Union[float, int]] = {
                        "dim": int(dim_index),
                        "logit_mean": float(logits.mean().item()),
                        "logit_std": float(logits.std(unbiased=False).item()),
                    }

                    try:
                        dist = torch.distributions.Categorical(logits=logits)
                        entropy = float(dist.entropy().mean().item())
                        top_prob = float(dist.probs.max(dim=1).values.mean().item())
                    except ValueError:
                        entropy = 0.0
                        top_prob = 1.0

                    stats["mean_entropy"] = entropy
                    stats["mean_top_prob"] = top_prob
                    step_dims.append(stats)

                entry["steps"].append({"rollout_step": rollout_step, "dims": step_dims})

                for agent_id in range(instrumentation_env.num_agents):
                    for dim_index, logits in enumerate(logits_cpu):
                        dist = torch.distributions.Categorical(logits=logits[agent_id])
                        instrumentation_actions[agent_id, dim_index] = int(dist.sample().item())

                obs, _, done, truncated, _ = instrumentation_env.step(instrumentation_actions)
                if all(done) or all(truncated):
                    break

            debug_file.write(json.dumps(entry) + "\n")
            debug_file.flush()

    else:

        def _write_logit_snapshot(global_step: int) -> None:
            return

    # Load the TrainablePolicy class using the new API
    policy_class = load_symbol(policy_class_path)
    policy = policy_class(vecenv.driver_env, device)

    # Ensure it implements the TrainablePolicy interface
    assert isinstance(policy, TrainablePolicy), (
        f"Policy class {policy_class_path} must implement TrainablePolicy interface"
    )

    # Load initial weights if provided
    if initial_weights_path:
        policy.load_policy_data(initial_weights_path)

    # Detect if policy uses RNN (e.g., LSTM)
    use_rnn = policy.is_recurrent()

    if logits_debug_path is not None:
        _write_logit_snapshot(0)

    env_name = "cogames.cogs_vs_clips"

    # Use RNN-specific hyperparameters if needed
    if use_rnn:
        learning_rate = 1e-4  # Slightly gentler LR to keep critic stable
        bptt_horizon = 1  # TODO: revisit once observation reshaping supports >1
        optimizer = "adam"
        adam_eps = 1e-8
        logger.info("Using RNN hyperparameters: lr=1e-4, bptt=1, optimizer=adam")
    else:
        learning_rate = 1e-4
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
        gamma=0.97,
        gae_lambda=0.90,
        update_epochs=1,
        clip_coef=0.1,
        vf_coef=0.1,
        vf_clip_coef=0.1,
        max_grad_norm=0.5,
        ent_coef=0.01,
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

        if logits_debug_path is not None:
            _write_logit_snapshot(trainer.global_step)

    trainer.print_dashboard()
    trainer.close()

    if debug_file is not None:
        debug_file.close()

    if instrumentation_env is not None:
        instrumentation_env.close()

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
            "cogames.policy.token.TokenPolicy": "token",
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

        maybe_upload_checkpoint(
            final_checkpoint=final_checkpoint,
            game_name=game_name,
            policy_class_path=policy_class_path,
            console=console,
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
