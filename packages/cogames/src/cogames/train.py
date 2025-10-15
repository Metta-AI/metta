from __future__ import annotations

import logging
import math
import multiprocessing
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import psutil
from rich.console import Console

from cogames.cli.mission import MAP_MISSION_DELIMITER
from cogames.policy.interfaces import TrainablePolicy
from cogames.policy.signal_handler import DeferSigintContextManager
from cogames.policy.utils import (
    find_policy_checkpoints,
    get_policy_class_shorthand,
    initialize_or_load_policy,
    resolve_policy_data_path,
)
from mettagrid import MettaGridConfig, MettaGridEnv
from pufferlib import pufferl
from pufferlib import vector as pvector
from pufferlib.pufferlib import set_buffers

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("cogames.pufferlib")


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _resolve_vector_counts(
    num_envs: int,
    num_workers: int,
    *,
    envs_user_supplied: bool,
    workers_user_supplied: bool,
) -> tuple[int, int]:
    """Adjust counts so num_envs stays divisible by num_workers."""

    num_envs = max(1, num_envs)
    num_workers = max(1, num_workers)

    if envs_user_supplied and workers_user_supplied:
        return num_envs, num_workers

    if envs_user_supplied:
        adjusted_workers = _largest_divisor_at_most(num_envs, min(num_workers, num_envs))
        return num_envs, max(1, adjusted_workers)

    if workers_user_supplied:
        num_envs = max(num_envs, num_workers)
        if num_envs % num_workers != 0:
            num_envs = num_workers * math.ceil(num_envs / num_workers)
        return num_envs, num_workers

    if num_envs < num_workers:
        num_envs = num_workers
        return num_envs, num_workers

    if num_envs % num_workers != 0:
        num_envs = num_workers * math.ceil(num_envs / num_workers)

    return num_envs, num_workers


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
    missions_arg: Optional[list[str]] = None,
    vector_num_envs: Optional[int] = None,
    vector_batch_size: Optional[int] = None,
    vector_num_workers: Optional[int] = None,
    env_cfg_supplier: Optional[Callable[[], MettaGridConfig]] = None,
) -> None:
    import pufferlib.pytorch  # noqa: F401 - ensure modules register with torch

    console = Console()

    if env_cfg is None and env_cfg_supplier is None:
        raise ValueError("Either env_cfg or env_cfg_supplier must be provided")

    backend = pvector.Multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)
        backend = pvector.Serial

    try:
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    except Exception:  # pragma: no cover - best effort fallback
        cpu_cores = None

    desired_workers = vector_num_workers or cpu_cores or 4
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

    if backend is pvector.Multiprocessing and device.type != "cuda":
        backend = pvector.Serial
        num_workers = 1

    num_envs = vector_num_envs or 256

    original_envs = num_envs
    original_workers = num_workers

    adjusted_envs, adjusted_workers = _resolve_vector_counts(
        num_envs,
        num_workers,
        envs_user_supplied=vector_num_envs is not None,
        workers_user_supplied=vector_num_workers is not None,
    )

    if adjusted_envs != original_envs:
        log_fn = logger.warning if vector_num_envs is not None else logger.info
        log_fn(
            "Auto-adjusting num_envs from %s to %s so num_workers=%s divides evenly",
            original_envs,
            adjusted_envs,
            adjusted_workers,
        )
        num_envs = adjusted_envs

    if adjusted_workers != original_workers:
        log_fn = logger.warning if vector_num_workers is not None else logger.info
        log_fn(
            "Auto-adjusting num_workers from %s to %s to evenly divide num_envs=%s",
            original_workers,
            adjusted_workers,
            num_envs,
        )
        num_workers = adjusted_workers

    envs_per_worker = max(1, num_envs // num_workers)
    base_batch_size = vector_batch_size or 128
    vector_batch_size = max(base_batch_size, envs_per_worker)
    remainder = vector_batch_size % envs_per_worker
    if remainder:
        vector_batch_size += envs_per_worker - remainder

    if backend is pvector.Serial:
        vector_batch_size = num_envs

    logger.debug(
        "Vec env config: num_envs=%s, num_workers=%s, batch_size=%s (envs/worker=%s)",
        num_envs,
        num_workers,
        vector_batch_size,
        envs_per_worker,
    )

    def _clone_cfg() -> MettaGridConfig:
        if env_cfg_supplier is not None:
            supplied = env_cfg_supplier()
            if not isinstance(supplied, MettaGridConfig):
                raise TypeError("env_cfg_supplier must return a MettaGridConfig")
            return supplied.model_copy(deep=True)
        assert env_cfg is not None
        return env_cfg.model_copy(deep=True)

    base_cfg = _clone_cfg()

    def env_creator(
        cfg: Optional[MettaGridConfig] = None,
        buf: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        target_cfg = cfg.model_copy(deep=True) if cfg is not None else _clone_cfg()
        env = MettaGridEnv(env_cfg=target_cfg)
        set_buffers(env, buf)
        return env

    vecenv = pvector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=vector_batch_size,
        backend=backend,
        env_kwargs={"cfg": base_cfg},
    )

    resolved_initial_weights = initial_weights_path
    if initial_weights_path is not None:
        try:
            resolved_initial_weights = resolve_policy_data_path(initial_weights_path)
        except FileNotFoundError as exc:
            console.print(f"[yellow]Initial weights not found ({exc}). Continuing with random initialization.[/yellow]")
            resolved_initial_weights = None

    policy = initialize_or_load_policy(
        policy_class_path,
        resolved_initial_weights,
        vecenv.driver_env,
        device,
    )
    assert isinstance(policy, TrainablePolicy), (
        f"Policy class {policy_class_path} must implement TrainablePolicy interface"
    )

    use_rnn = getattr(policy, "is_recurrent", lambda: False)()
    if not use_rnn:
        lowered = policy_class_path.lower()
        if "lstm" in lowered or "rnn" in lowered:
            use_rnn = True

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
    num_envs = max(1, getattr(vecenv, "num_envs", 1))
    num_workers = max(1, getattr(vecenv, "num_workers", 1))
    envs_per_worker = max(1, num_envs // num_workers)

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

    checkpoint_interval = 200
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

    with DeferSigintContextManager():
        try:
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
        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt received at step %s; stopping training gracefully.",
                trainer.global_step,
            )

        trainer.print_dashboard()
        trainer.close()

        console.print()
        if training_diverged:
            console.print("=" * 80, style="bold red")
            console.print("Training diverged (NaN detected)! Stopped early.", style="bold red")
            console.print(f"Checkpoints saved to: [cyan]{checkpoints_path}[/cyan]", style="bold red")
            console.print("=" * 80, style="bold red")
            console.print()
            console.print("[yellow]Warning: The latest checkpoint may contain NaN values.[/yellow]")
            console.print("[yellow]Try using an earlier checkpoint or retraining with lower learning rate.[/yellow]")
        elif trainer.epoch >= checkpoint_interval:
            console.print("=" * 80, style="bold green")
            console.print("Training complete")
            console.print(
                f"Checkpoints saved to: [cyan]{checkpoints_path}[/cyan]",
                style="bold green",
            )
            console.print("=" * 80, style="bold green")

        checkpoints = find_policy_checkpoints(checkpoints_path, env_name)

        if checkpoints and not training_diverged:
            final_checkpoint = checkpoints[-1]
            console.print()
            console.print(f"Final checkpoint: [cyan]{final_checkpoint}[/cyan]")
            if trainer.epoch < checkpoint_interval:
                console.print(
                    "This checkpoint has initialized weights but does not reflect training. \n"
                    "Training was cut off before the first meaningful checkpoint would have been saved"
                    f" (epoch {checkpoint_interval}).",
                    style="yellow",
                )

            # Show shorthand version if available
            policy_shorthand = get_policy_class_shorthand(policy_class_path)

            # Build the command with game name if provided
            policy_class_arg = policy_shorthand if policy_shorthand else policy_class_path
            policy_arg = f"{policy_class_arg}{MAP_MISSION_DELIMITER}{final_checkpoint}"

            first_mission = missions_arg[0] if missions_arg else "training_facility_1"
            all_missions = " ".join(f"-m {m}" for m in (missions_arg or ["training_facility_1"]))

            console.print()
            console.print("To continue training this policy:", style="bold")
            console.print(f"  [yellow]cogames train {all_missions} -p {policy_arg}[/yellow]")
            console.print()
            console.print("To play with this policy:", style="bold")
            console.print(f"  [yellow]cogames play -m {first_mission} -p {policy_arg}[/yellow]")
            console.print()
            console.print("To evaluate this policy:", style="bold")
            console.print(f"  [yellow]cogames eval -m {first_mission} -p {policy_arg}[/yellow]")
        elif checkpoints and training_diverged:
            console.print()
            console.print(f"[yellow]Found {len(checkpoints)} checkpoint(s). The most recent may be corrupted.[/yellow]")
            console.print("[yellow]Try using an earlier checkpoint or retraining.[/yellow]")
        else:
            console.print()
            console.print(f"[yellow]No checkpoint files found. Check {checkpoints_path} for saved models.[/yellow]")

        console.print("=" * 80, style="bold green")
        console.print()
