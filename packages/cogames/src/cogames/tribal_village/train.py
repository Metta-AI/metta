"""Training loop for the Tribal Village environment using PufferLib."""

from __future__ import annotations

import logging
import multiprocessing
import platform
from pathlib import Path
from typing import Any, Optional

import psutil
import torch
from rich.console import Console
from tribal_village_env.build import ensure_nim_library_current
from tribal_village_env.environment import TribalVillageEnv

from cogames.policy.signal_handler import DeferSigintContextManager
from cogames.policy.tribal_village_policy import TribalPolicyEnvInfo
from cogames.train import _resolve_vector_counts
from mettagrid.policy.loader import (
    find_policy_checkpoints,
    get_policy_class_shorthand,
    initialize_or_load_policy,
    resolve_policy_data_path,
)
from mettagrid.policy.policy import PolicySpec
from pufferlib import pufferl
from pufferlib import vector as pvector
from pufferlib.pufferlib import set_buffers

logger = logging.getLogger("cogames.tribal_village.train")


class _TribalEnvCreator:
    """Picklable factory for vectorized Tribal Village environments."""

    def __init__(self, base_config: dict[str, Any]):
        self._base_config = base_config

    def clone_cfg(self) -> dict[str, Any]:
        return dict(self._base_config)

    def __call__(
        self,
        cfg: Optional[dict[str, Any]] = None,
        buf: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> TribalVillageEnv:
        merged_cfg = dict(self._base_config)
        if cfg is not None:
            merged_cfg.update(cfg)
        if seed is not None and "seed" not in merged_cfg:
            merged_cfg["seed"] = seed

        env = TribalVillageEnv(config=merged_cfg)
        set_buffers(env, buf)
        return env


def train(
    config: Optional[dict[str, Any]],
    policy_class_path: str,
    device: torch.device,
    initial_weights_path: Optional[str],
    num_steps: int,
    checkpoints_path: Path,
    seed: int,
    batch_size: int,
    minibatch_size: int,
    *,
    vector_num_envs: Optional[int] = None,
    vector_batch_size: Optional[int] = None,
    vector_num_workers: Optional[int] = None,
    log_outputs: bool = False,
) -> None:
    """Run PPO training for Tribal Village."""

    ensure_nim_library_current()

    console = Console()

    backend = pvector.Multiprocessing
    if platform.system() == "Darwin":
        # macOS requires spawn for Multiprocessing to work reliably
        multiprocessing.set_start_method("spawn", force=True)

    try:
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    except Exception:
        cpu_cores = None

    desired_workers = vector_num_workers or cpu_cores or 4
    num_workers = min(desired_workers, max(1, cpu_cores or desired_workers))

    num_envs = vector_num_envs or 64

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

    def _divisible_batch(min_size: int) -> int:
        for candidate in range(num_envs, min_size - 1, -1):
            if num_envs % candidate == 0:
                return candidate
        return num_envs

    target_min = max(envs_per_worker, vector_batch_size or envs_per_worker)
    vector_batch_size = _divisible_batch(target_min)

    logger.debug(
        "Vec env config: num_envs=%s, num_workers=%s, batch_size=%s (envs/worker=%s)",
        num_envs,
        num_workers,
        vector_batch_size,
        envs_per_worker,
    )

    base_config = {"render_mode": "ansi", "render_scale": 1}
    if config:
        base_config.update(config)

    env_creator = _TribalEnvCreator(base_config)
    base_cfg = env_creator.clone_cfg()

    vecenv = pvector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=vector_batch_size,
        backend=backend,
        env_kwargs={"cfg": base_cfg},
    )

    driver_env = getattr(vecenv, "driver_env", None)
    if driver_env is None:
        raise RuntimeError("Vectorized environment did not expose driver_env for shape inference.")

    policy_env_info = TribalPolicyEnvInfo(
        observation_space=driver_env.single_observation_space,
        action_space=driver_env.single_action_space,
        num_agents=max(1, getattr(driver_env, "num_agents", 1)),
    )

    resolved_initial_weights: Optional[str] = None
    if initial_weights_path is not None:
        try:
            resolved_initial_weights = resolve_policy_data_path(initial_weights_path)
        except FileNotFoundError as exc:
            console.print(f"[yellow]Initial weights not found ({exc}). Continuing with random initialization.[/yellow]")

    policy_spec = PolicySpec(class_path=policy_class_path, data_path=resolved_initial_weights)
    policy = initialize_or_load_policy(policy_env_info, policy_spec)
    network = policy.network()
    assert network is not None, f"Policy {policy_class_path} must be trainable (network() returned None)"
    network.to(device)

    use_rnn = getattr(policy, "is_recurrent", lambda: False)()
    if not use_rnn and "lstm" in policy_class_path.lower():
        use_rnn = True

    env_name = "tribal_village"

    learning_rate = 0.0005
    bptt_horizon = 64 if use_rnn else 1
    optimizer = "adam"
    adam_eps = 1e-8

    total_agents = max(1, getattr(vecenv, "num_agents", getattr(driver_env, "num_agents", 1)))
    num_envs = max(1, getattr(vecenv, "num_envs", num_envs))
    num_workers = max(1, getattr(vecenv, "num_workers", num_workers))
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

    trainer: Optional[pufferl.PuffeRL] = None
    training_diverged = False

    try:
        trainer = pufferl.PuffeRL(train_args, vecenv, network)
        if log_outputs:
            console.clear()
            console.print("[dim]Evaluation stats will stream below; disabling Rich dashboard.[/dim]")
            trainer.print_dashboard = lambda *_, **__: None  # type: ignore[assignment]

        with DeferSigintContextManager():
            while trainer.global_step < num_steps:
                eval_stats = trainer.evaluate()
                if log_outputs and eval_stats:
                    console.log("Evaluation", eval_stats)
                trainer_stats = trainer.train()
                if log_outputs and trainer_stats:
                    console.log("Training", trainer_stats)

    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        logger.warning("Training interrupted by user. Saving latest model if available.")
    except RuntimeError as exc:  # pragma: no cover - runtime failure
        training_diverged = True
        logger.exception("Training diverged with runtime error: %s", exc)
    finally:
        if trainer is not None:
            try:
                trainer.print_dashboard()
            except Exception:
                pass
            try:
                trainer.close()
            except Exception:
                pass

        try:
            vecenv.close()
        except Exception:
            pass

        console.rule("[bold green]Training Summary")
        checkpoints = find_policy_checkpoints(checkpoints_path, env_name)

        if checkpoints and not training_diverged:
            final_checkpoint = checkpoints[-1]
            console.print(f"Final checkpoint: [cyan]{final_checkpoint}[/cyan]")
            if trainer is not None and trainer.epoch < checkpoint_interval:
                console.print(
                    f"Training stopped before first scheduled checkpoint (epoch {checkpoint_interval}). "
                    "Latest weights may be near-random.",
                    style="yellow",
                )

            policy_shorthand = get_policy_class_shorthand(policy_class_path)
            policy_arg = policy_shorthand if policy_shorthand else policy_class_path
            policy_with_checkpoint = f"{policy_arg}:{final_checkpoint}"

            console.print()
            console.print("To continue training this policy:", style="bold")
            console.print(f"  [yellow]cogames train-tribal -p {policy_with_checkpoint}[/yellow]")

        elif checkpoints and training_diverged:
            console.print()
            console.print(f"[yellow]Found {len(checkpoints)} checkpoint(s). The most recent may be corrupted.[/yellow]")
            console.print("[yellow]Try using an earlier checkpoint or retraining.[/yellow]")
        else:
            console.print()
            console.print(f"[yellow]No checkpoint files found. Check {checkpoints_path} for saved models.[/yellow]")

        console.rule("[bold green]End Training")
