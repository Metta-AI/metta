from __future__ import annotations

import logging
import multiprocessing
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import psutil
from rich.console import Console

from cogames.env import make_hierarchical_env
from cogames.policy import TrainablePolicy
from cogames.policy.utils import resolve_policy_data_path
from cogames.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig
from pufferlib import pufferl
from pufferlib import vector as pvector
from pufferlib.pufferlib import set_buffers

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

    cpu_cores = None
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

    envs_per_worker = max(1, num_envs // num_workers)
    base_batch_size = vector_batch_size or 128
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
        env = make_hierarchical_env(env_cfg=target_cfg, buf=buf)
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
            resolved_initial_weights = resolve_policy_data_path(
                initial_weights_path,
                policy_class_path=policy_class_path,
                game_name=game_name,
                console=console,
            )
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
        learning_rate = 0.0006
        bptt_horizon = 1
        optimizer = "adam"
        adam_eps = 1e-8
        logger.info("Using RNN-specific hyperparameters: lr=0.0006, bptt=1, optimizer=adam")
    else:
        learning_rate = 0.03
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

        network = policy.network()
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

    if checkpoints and not training_diverged:
        final_checkpoint = checkpoints[-1]
        console.print()
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
        console.print()
        console.print(f"[yellow]Found {len(checkpoints)} checkpoint(s). The most recent may be corrupted.[/yellow]")
        console.print("[yellow]Try using an earlier checkpoint or retraining.[/yellow]")
    else:
        console.print()
        console.print(f"[yellow]No checkpoint files found. Check {checkpoints_path} for saved models.[/yellow]")

    console.print("=" * 80, style="bold green")
    console.print()
