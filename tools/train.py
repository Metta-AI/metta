#!/usr/bin/env -S uv run
import os
import sys
from logging import Logger
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.metta_agent import DistributedMettaAgent
from metta.agent.policy_store import PolicyStore
from metta.common.stopwatch import Stopwatch
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functional_trainer import rollout, train_ppo
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import Config
from metta.util.heartbeat import record_heartbeat
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.stats_client_cfg import get_stats_client
from metta.util.wandb.wandb_context import WandbContext, WandbRun

try:
    from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None

torch.set_float32_matmul_precision("high")


# TODO: populate this more
class TrainJob(Config):
    __init__ = Config.__init__
    evals: SimulationSuiteConfig
    map_preview_uri: Optional[str] = None


def functional_train(cfg: ListConfig | DictConfig, wandb_run: WandbRun | None, logger: Logger):
    """Functional training loop implementation."""
    overrides_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    if os.path.exists(overrides_path):
        logger.info(f"Loading train config overrides from {overrides_path}")
        override_cfg = OmegaConf.load(overrides_path)

        # Set struct flag to False to allow accessing undefined fields
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, override_cfg)
        # Optionally, restore struct behavior after merge
        OmegaConf.set_struct(cfg, True)

    if os.environ.get("RANK", "0") == "0":
        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    TrainJob(cfg.train_job)

    policy_store = PolicyStore(cfg, wandb_run)
    get_stats_client(cfg, logger)

    # Extract trainer config
    trainer_cfg = cfg.trainer
    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device

    # Handle distributed training
    is_master = True
    world_size = 1
    batch_size = trainer_cfg.batch_size
    minibatch_size = trainer_cfg.minibatch_size

    if torch.distributed.is_initialized():
        is_master = int(os.environ["RANK"]) == 0
        world_size = torch.distributed.get_world_size()
        batch_size = trainer_cfg.batch_size // world_size
        minibatch_size = trainer_cfg.minibatch_size // world_size
        trainer_cfg.forward_pass_minibatch_target_size = trainer_cfg.forward_pass_minibatch_target_size // world_size

    # Setup timer
    timer = Stopwatch(logger)
    timer.start()

    # Create curriculum
    curriculum_config = trainer_cfg.get("curriculum", trainer_cfg.get("env", {}))
    env_overrides = DictConfig({"env_overrides": trainer_cfg.env_overrides})
    curriculum = curriculum_from_config_path(curriculum_config, env_overrides)

    # Create vecenv
    num_agents = curriculum.get_task().env_cfg().game.num_agents
    target_batch_size = trainer_cfg.forward_pass_minibatch_target_size // num_agents
    if target_batch_size < 2:  # pufferlib bug requires batch size >= 2
        target_batch_size = 2

        # Use num_workers with fallback default
    num_workers = trainer_cfg.get("num_workers", 1)
    env_batch_size = (target_batch_size // num_workers) * num_workers
    if env_batch_size < 1:
        env_batch_size = num_workers
    num_envs = env_batch_size * trainer_cfg.async_factor

    # Ensure batch_size is aligned properly for pufferlib
    # For serial execution (num_workers=1), batch_size must equal num_envs
    if num_workers == 1:
        env_batch_size = num_envs

    logger.info(f"Creating vecenv with {num_envs} environments, batch_size={env_batch_size}, num_workers={num_workers}")
    vecenv = make_vecenv(
        curriculum,
        cfg.vectorization,
        num_envs=num_envs,
        batch_size=env_batch_size,
        num_workers=num_workers,
        zero_copy=trainer_cfg.zero_copy,
    )

    # Reset environment
    if cfg.seed is None:
        cfg.seed = np.random.randint(0, 1000000)
    rank = int(os.environ.get("RANK", 0))
    vecenv.async_reset(cfg.seed + rank)

    # Get environment
    metta_grid_env: MettaGridEnv = vecenv.driver_env
    assert isinstance(metta_grid_env, MettaGridEnv), (
        f"vecenv.driver_env type {type(metta_grid_env).__name__} is not MettaGridEnv"
    )

    # Load checkpoint
    logger.info("Loading checkpoint")
    os.makedirs(trainer_cfg.checkpoint_dir, exist_ok=True)
    checkpoint = TrainerCheckpoint.load(cfg.run_dir)

    # Load policy
    policy_record = _load_policy(checkpoint, policy_store, metta_grid_env)
    assert policy_record is not None, "No policy found"

    if is_master:
        logger.info(f"Loaded policy: {policy_record.policy()}")

    policy = policy_record.policy().to(device)

    # Activate actions
    actions_names = metta_grid_env.action_names
    actions_max_params = metta_grid_env.max_action_args
    policy.activate_actions(actions_names, actions_max_params, device)

    # Store uncompiled policy reference for saving
    uncompiled_policy = policy

    # Compile policy if requested
    if trainer_cfg.compile:
        logger.info("Compiling policy")
        policy = torch.compile(policy, mode=trainer_cfg.compile_mode)

    # Setup kickstarter
    kickstarter = Kickstarter(cfg, policy_store, actions_names, actions_max_params)

    # Handle distributed training
    if torch.distributed.is_initialized():
        logger.info(f"Initializing DistributedDataParallel on device {device}")
        policy = DistributedMettaAgent(uncompiled_policy, device)

    # Create experience buffer
    obs_space = vecenv.single_observation_space
    atn_space = vecenv.single_action_space
    total_agents = vecenv.num_agents
    hidden_size = getattr(policy, "hidden_size", 256)
    num_lstm_layers = 2  # Default value

    experience = Experience(
        total_agents=total_agents,
        batch_size=batch_size,
        bptt_horizon=trainer_cfg.bptt_horizon,
        minibatch_size=minibatch_size,
        max_minibatch_size=trainer_cfg.get("max_minibatch_size", minibatch_size),
        obs_space=obs_space,
        atn_space=atn_space,
        device=device,
        hidden_size=hidden_size,
        cpu_offload=trainer_cfg.cpu_offload,
        num_lstm_layers=num_lstm_layers,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # Initialize training state
    agent_step = checkpoint.agent_step
    epoch = checkpoint.epoch

    # Create optimizer
    optimizer_type = getattr(trainer_cfg.optimizer, "type", "adam")
    assert optimizer_type in ("adam", "muon"), f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}"

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=trainer_cfg.optimizer.learning_rate,
            betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
            eps=trainer_cfg.optimizer.eps,
            weight_decay=trainer_cfg.optimizer.weight_decay,
        )
    else:
        from heavyball import ForeachMuon

        optimizer = ForeachMuon(
            policy.parameters(),
            lr=trainer_cfg.optimizer.learning_rate,
            betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
            eps=trainer_cfg.optimizer.eps,
            weight_decay=trainer_cfg.optimizer.weight_decay,
        )

    # Load optimizer state if available
    if checkpoint.agent_step > 0:
        try:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info("Successfully loaded optimizer state from checkpoint")
        except ValueError as e:
            logger.warning(f"Failed to load optimizer state: {e}")

    # Create lr_scheduler if enabled
    lr_scheduler = None
    if hasattr(trainer_cfg, "lr_scheduler") and getattr(trainer_cfg.lr_scheduler, "enabled", False):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=trainer_cfg.total_timesteps // trainer_cfg.batch_size
        )

    # Create losses tracker
    losses = Losses()

    # Define wandb metrics if master
    if wandb_run and is_master:
        metrics = ["agent_step", "epoch", "total_time", "train_time"]
        for metric in metrics:
            wandb_run.define_metric(f"metric/{metric}")
        wandb_run.define_metric("*", step_metric="metric/agent_step")

    logger.info(f"Starting functional training loop on device: {device}")

    # Main training loop
    while agent_step < trainer_cfg.total_timesteps:
        steps_before = agent_step

        with timer("rollout"):
            agent_step, stats = rollout(policy, vecenv, experience, device, agent_step, timer)

        with timer("train"):
            epoch = train_ppo(
                policy=policy,
                optimizer=optimizer,
                experience=experience,
                device=device,
                losses=losses,
                epoch=epoch,
                cfg=cfg,
                lr_scheduler=lr_scheduler,
                timer=timer,
                gamma=trainer_cfg.gamma,
                gae_lambda=trainer_cfg.gae_lambda,
                clip_coef=trainer_cfg.clip_coef,
                ent_coef=trainer_cfg.ent_coef,
                vf_coef=trainer_cfg.vf_coef,
                max_grad_norm=trainer_cfg.max_grad_norm,
                norm_adv=trainer_cfg.norm_adv,
                clip_vloss=trainer_cfg.clip_vloss,
                vf_clip_coef=trainer_cfg.vf_clip_coef,
                update_epochs=trainer_cfg.update_epochs,
                target_kl=trainer_cfg.target_kl,
                kickstarter=kickstarter,
                agent_step=agent_step,
                l2_reg_loss_coef=trainer_cfg.l2_reg_loss_coef,
                l2_init_loss_coef=trainer_cfg.l2_init_loss_coef,
                clip_range=cfg.agent.clip_range,
                prio_alpha=trainer_cfg.prioritized_experience_replay.get("prio_alpha", 0.0),
                prio_beta0=trainer_cfg.prioritized_experience_replay.get("prio_beta0", 0.6),
                total_timesteps=trainer_cfg.total_timesteps,
                batch_size=trainer_cfg.batch_size,
                vtrace_rho_clip=trainer_cfg.vtrace.get("vtrace_rho_clip", 1.0),
                vtrace_c_clip=trainer_cfg.vtrace.get("vtrace_c_clip", 1.0),
            )

        # Calculate timing
        rollout_time = timer.get_last_elapsed("rollout")
        train_time = timer.get_last_elapsed("train")
        total_time = train_time + rollout_time
        steps_calculated = agent_step - steps_before
        steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

        train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
        rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

        logger.info(
            f"Epoch {epoch} - "
            f"{steps_per_sec * world_size:.0f} steps/sec "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout)"
        )
        record_heartbeat()

        # Log to wandb
        if wandb_run and is_master and stats:
            # Process stats - convert lists to means
            mean_stats = {}
            for k, v in stats.items():
                try:
                    mean_stats[k] = np.mean(v) if isinstance(v, list) else v
                except (TypeError, ValueError):
                    pass

            # Extract losses
            loss_stats = losses.stats()

            # Log metrics
            wandb_run.log(
                {
                    "metric/agent_step": agent_step * world_size,
                    "metric/epoch": epoch,
                    "metric/total_time": timer.get_elapsed(),
                    "overview/sps": steps_per_sec * world_size,
                    **{f"losses/{k}": v for k, v in loss_stats.items()},
                    **{f"env_{k}": v for k, v in mean_stats.items()},
                    **{f"experience/{k}": v for k, v in experience.stats().items()},
                }
            )

            # Checkpointing
        if epoch % trainer_cfg.checkpoint_interval == 0 and is_master:
            logger.info("Saving checkpoint...")
            # Save policy
            name = policy_store.make_model_name(epoch)
            path = os.path.join(trainer_cfg.checkpoint_dir, name)
            pr = policy_store.save(
                name=name, path=path, policy=uncompiled_policy, metadata={"epoch": epoch, "agent_step": agent_step}
            )

            # Save trainer checkpoint
            checkpoint = TrainerCheckpoint(
                agent_step=agent_step,
                epoch=epoch,
                total_agent_step=agent_step * world_size,
                optimizer_state_dict=optimizer.state_dict(),
                policy_path=pr.uri if pr else None,
                extra_args={},
            )
            checkpoint.save(cfg.run_dir)

        # Update L2 init weights
        if cfg.agent.l2_init_weight_update_interval != 0 and epoch % cfg.agent.l2_init_weight_update_interval == 0:
            policy.update_l2_init_weight_copy()

    logger.info("Training complete!")

    # Final checkpoint
    if is_master:
        name = policy_store.make_model_name(epoch)
        path = os.path.join(trainer_cfg.checkpoint_dir, name)
        pr = policy_store.save(
            name=name, path=path, policy=uncompiled_policy, metadata={"epoch": epoch, "agent_step": agent_step}
        )

        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            total_agent_step=agent_step * world_size,
            optimizer_state_dict=optimizer.state_dict(),
            policy_path=pr.uri if pr else None,
            extra_args={},
        )
        checkpoint.save(cfg.run_dir)

    vecenv.close()


def _load_policy(checkpoint, policy_store, metta_grid_env):
    """Load policy from checkpoint or create new one."""
    if checkpoint.policy_path:
        return policy_store.load_from_uri(checkpoint.policy_path)
    else:
        # Create new policy
        return policy_store.create(metta_grid_env)


@record
@hydra.main(config_path="../configs", config_name="train_job", version_base=None)
def main(cfg: ListConfig | DictConfig) -> int:
    setup_mettagrid_environment(cfg)

    record_heartbeat()

    logger = setup_mettagrid_logger("train")
    logger.info(f"Train job config: {OmegaConf.to_yaml(cfg, resolve=True)}")

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    if "LOCAL_RANK" in os.environ and cfg.device.startswith("cuda"):
        logger.info(f"Initializing distributed training with {os.environ['LOCAL_RANK']} {cfg.device}")
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f"{cfg.device}:{local_rank}"
        dist.init_process_group(backend="nccl")

    logger.info(f"Training {cfg.run} on {cfg.device}")
    if os.environ.get("RANK", "0") == "0":
        with WandbContext(cfg.wandb, cfg) as wandb_run:
            functional_train(cfg, wandb_run, logger)
    else:
        functional_train(cfg, None, logger)

    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
