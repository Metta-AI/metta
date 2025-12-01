"""Training loop for the Tribal Village environment using PufferLib."""

from __future__ import annotations

import logging
import multiprocessing
import platform
from typing import Optional

import psutil
import torch
from rich.console import Console

from cogames.policy.signal_handler import DeferSigintContextManager
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

from tribal_village_env.cogames.config import TribalTrainSettings
from tribal_village_env.cogames.policy import TribalPolicyEnvInfo
from tribal_village_env.cogames.vector_env import FlattenVecEnv, TribalEnvFactory

logger = logging.getLogger("cogames.tribal_village.train")


def _resolve_worker_counts(settings: TribalTrainSettings) -> tuple[int, int]:
    try:
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    except Exception:
        cpu_cores = None

    desired_workers = settings.vector_num_workers or cpu_cores or 4
    num_workers = min(desired_workers, max(1, cpu_cores or desired_workers))

    num_envs = settings.vector_num_envs or 64

    adjusted_envs, adjusted_workers = _resolve_vector_counts(
        num_envs,
        num_workers,
        envs_user_supplied=settings.vector_num_envs is not None,
        workers_user_supplied=settings.vector_num_workers is not None,
    )

    if adjusted_envs != num_envs:
        log_fn = logger.warning if settings.vector_num_envs is not None else logger.info
        log_fn(
            "Auto-adjusting num_envs from %s to %s so num_workers=%s divides evenly",
            num_envs,
            adjusted_envs,
            adjusted_workers,
        )
        num_envs = adjusted_envs

    if adjusted_workers != num_workers:
        log_fn = logger.warning if settings.vector_num_workers is not None else logger.info
        log_fn(
            "Auto-adjusting num_workers from %s to %s to evenly divide num_envs=%s",
            num_workers,
            adjusted_workers,
            num_envs,
        )
        num_workers = adjusted_workers

    return num_envs, num_workers


def _resolve_vector_batch_size(num_envs: int, num_workers: int, vector_batch_size: Optional[int]) -> int:
    envs_per_worker = max(1, num_envs // num_workers)
    if vector_batch_size is None:
        return num_envs
    if num_envs % vector_batch_size != 0:
        logger.warning(
            "vector_batch_size=%s does not evenly divide num_envs=%s; resetting to %s",
            vector_batch_size,
            num_envs,
            num_envs,
        )
        return num_envs
    return vector_batch_size


def train(settings: TribalTrainSettings) -> None:
    """Run PPO training for Tribal Village using the provided settings."""

    from tribal_village_env.build import ensure_nim_library_current

    ensure_nim_library_current()

    console = Console()

    backend = pvector.Multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)

    num_envs, num_workers = _resolve_worker_counts(settings)
    vector_batch_size = _resolve_vector_batch_size(num_envs, num_workers, settings.vector_batch_size)

    base_config = {"render_mode": "ansi", "render_scale": 1}
    base_config.update(settings.env_config)

    env_creator = TribalEnvFactory(base_config)
    base_cfg = env_creator.clone_cfg()

    vecenv = pvector.make(
        env_creator,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=vector_batch_size,
        backend=backend,
        env_kwargs={"cfg": base_cfg},
    )
    agents_per_batch = getattr(vecenv, "agents_per_batch", None)
    if agents_per_batch is not None:
        vecenv.num_agents = agents_per_batch
    vecenv = FlattenVecEnv(vecenv)

    driver_env = getattr(vecenv, "driver_env", None)
    if driver_env is None:
        raise RuntimeError("Vectorized environment did not expose driver_env for shape inference.")

    policy_env_info = TribalPolicyEnvInfo(
        observation_space=driver_env.single_observation_space,
        action_space=driver_env.single_action_space,
        num_agents=max(1, getattr(driver_env, "num_agents", 1)),
    )

    resolved_initial_weights: Optional[str] = None
    if settings.initial_weights_path is not None:
        try:
            resolved_initial_weights = resolve_policy_data_path(settings.initial_weights_path)
        except FileNotFoundError as exc:
            console.print(f"[yellow]Initial weights not found ({exc}). Continuing with random initialization.[/yellow]")

    policy_spec = PolicySpec(class_path=settings.policy_class_path, data_path=resolved_initial_weights)
    policy = initialize_or_load_policy(policy_env_info, policy_spec)
    network = policy.network()
    assert network is not None, f"Policy {settings.policy_class_path} must be trainable (network() returned None)"
    network.to(settings.device)

    use_rnn = getattr(policy, "is_recurrent", lambda: False)()
    if not use_rnn and "lstm" in settings.policy_class_path.lower():
        use_rnn = True

    env_name = "tribal_village"

    learning_rate = 0.0005
    bptt_horizon = 64 if use_rnn else 1
    optimizer = "adam"
    adam_eps = 1e-8

    total_agents = max(1, getattr(vecenv, "num_agents", getattr(driver_env, "num_agents", 1)))
    env_count = max(1, getattr(vecenv, "num_environments", num_envs))
    num_workers = max(1, getattr(vecenv, "num_workers", num_workers))
    envs_per_worker = max(1, getattr(vecenv, "envs_per_worker", env_count // num_workers))

    effective_agents_per_batch = agents_per_batch or total_agents
    amended_batch_size = effective_agents_per_batch
    if settings.batch_size != amended_batch_size:
        logger.warning(
            "batch_size=%s overridden to %s to match agents_per_batch; larger batches not yet supported",
            settings.batch_size,
            amended_batch_size,
        )

    amended_minibatch_size = min(settings.minibatch_size, amended_batch_size)
    if amended_minibatch_size != settings.minibatch_size:
        logger.info(
            "Reducing minibatch_size from %s to %s to keep it <= batch_size",
            settings.minibatch_size,
            amended_minibatch_size,
        )

    effective_timesteps = max(settings.steps, amended_batch_size)
    if effective_timesteps != settings.steps:
        logger.info(
            "Raising total_timesteps from %s to %s to keep it >= batch_size",
            settings.steps,
            effective_timesteps,
        )

    checkpoint_interval = 200
    train_args = dict(
        env=env_name,
        device=settings.device.type,
        total_timesteps=effective_timesteps,
        minibatch_size=amended_minibatch_size,
        batch_size=amended_batch_size,
        data_dir=str(settings.checkpoints_path),
        checkpoint_interval=checkpoint_interval,
        bptt_horizon=bptt_horizon,
        seed=settings.seed,
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

        with DeferSigintContextManager():
            while trainer.global_step < settings.steps:
                trainer.evaluate()
                trainer.train()

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
        checkpoints = find_policy_checkpoints(settings.checkpoints_path, env_name)

        if checkpoints and not training_diverged:
            final_checkpoint = checkpoints[-1]
            console.print(f"Final checkpoint: [cyan]{final_checkpoint}[/cyan]")
            if trainer is not None and trainer.epoch < checkpoint_interval:
                console.print(
                    f"Training stopped before first scheduled checkpoint (epoch {checkpoint_interval}). "
                    "Latest weights may be near-random.",
                    style="yellow",
                )

            policy_shorthand = get_policy_class_shorthand(settings.policy_class_path)
            policy_arg = policy_shorthand if policy_shorthand else settings.policy_class_path
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
            console.print(f"[yellow]No checkpoint files found. Check {settings.checkpoints_path} for saved models.[/yellow]")

        console.rule("[bold green]End Training")
