#!/usr/bin/env python3
"""
Functional Training Demo

This demonstrates how to use Metta's functional training approach without Hydra.
All objects are created directly, and configuration is done through Pydantic models.

This demo shows the core training loop using simple while/for loops,
calling the functional rollout() and train_ppo() functions directly.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import yaml
from pydantic import BaseModel

# Add metta to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.stopwatch import Stopwatch
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functional_trainer import rollout, train_ppo
from metta.rl.losses import Losses
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv

# Pufferlib C extension for advantage computation
try:
    from pufferlib import _C  # noqa: F401
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None


# Pydantic Configs for structured configuration
class PPOConfig(BaseModel):
    """PPO algorithm parameters"""

    gamma: float = 0.977
    gae_lambda: float = 0.916
    clip_coef: float = 0.1
    ent_coef: float = 0.0021
    vf_coef: float = 0.44
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    vf_clip_coef: float = 0.1
    target_kl: Optional[float] = None
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""

    type: str = "adam"  # "adam" or "muon"
    learning_rate: float = 0.0004573146765703167
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-12
    weight_decay: float = 0.0


class TrainingConfig(BaseModel):
    """Training loop configuration"""

    total_timesteps: int = 1_000_000
    batch_size: int = 6144
    minibatch_size: int = 256
    bptt_horizon: int = 64
    update_epochs: int = 1
    checkpoint_interval: int = 10
    evaluate_interval: int = 100
    cpu_offload: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    zero_copy: bool = True
    l2_reg_loss_coef: float = 0.0
    l2_init_loss_coef: float = 0.0


class EnvironmentConfig(BaseModel):
    """Environment configuration"""

    curriculum: str = "env/mettagrid/curriculum/simple"
    num_workers: int = 1
    async_factor: int = 2
    forward_pass_minibatch_target_size: int = 32
    seed: Optional[int] = 42


class PrioritizedReplayConfig(BaseModel):
    """Prioritized experience replay configuration"""

    prio_alpha: float = 0.0
    prio_beta0: float = 0.6


def load_yaml_config(path: str, config_class):
    """Load YAML config into Pydantic model"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return config_class(**data)


def create_optimizer(optimizer_cfg: OptimizerConfig, policy: torch.nn.Module):
    """Create optimizer from config"""
    if optimizer_cfg.type == "adam":
        return torch.optim.Adam(
            policy.parameters(),
            lr=optimizer_cfg.learning_rate,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
            eps=optimizer_cfg.eps,
            weight_decay=optimizer_cfg.weight_decay,
        )
    elif optimizer_cfg.type == "muon":
        from heavyball import ForeachMuon

        return ForeachMuon(
            policy.parameters(),
            lr=optimizer_cfg.learning_rate,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
            eps=optimizer_cfg.eps,
            weight_decay=optimizer_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_cfg.type}")


def main():
    """
    Main functional training loop demonstrating direct object creation
    without Hydra or complex configuration management.
    """
    print("🚀 Metta Functional Training Demo")
    print("=" * 50)

    # 1. Create configurations directly (or load from YAML)
    ppo_config = PPOConfig()
    optimizer_config = OptimizerConfig()
    training_config = TrainingConfig(
        total_timesteps=10_000,  # Short demo
        checkpoint_interval=5,
    )
    env_config = EnvironmentConfig()
    replay_config = PrioritizedReplayConfig()

    # Optional: Load from YAML files if they exist
    # ppo_config = load_yaml_config("configs/ppo/default.yaml", PPOConfig)
    # training_config = load_yaml_config("configs/training/default.yaml", TrainingConfig)

    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 3. Create directories
    run_dir = "./demo_run"
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 4. Create curriculum
    # For demo, we need to use Hydra temporarily just for curriculum loading
    # This is the simplest way to get a working environment without rewriting all configs

    from hydra import initialize_config_dir

    from metta.util.resolvers import register_resolvers

    # Register custom resolvers that configs might use
    register_resolvers()

    # Initialize Hydra temporarily just for config loading
    config_path = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # Load a simple curriculum config
        from metta.mettagrid.curriculum.util import curriculum_from_config_path

        curriculum = curriculum_from_config_path("env/mettagrid/curriculum/simple", DictConfig({}))

    # 5. Create vectorized environment
    num_agents = curriculum.get_task().env_cfg().game.num_agents
    target_batch_size = env_config.forward_pass_minibatch_target_size // num_agents
    target_batch_size = max(2, target_batch_size)  # pufferlib requires batch_size >= 2

    env_batch_size = (target_batch_size // env_config.num_workers) * env_config.num_workers
    env_batch_size = max(env_config.num_workers, env_batch_size)
    num_envs = env_batch_size * env_config.async_factor

    if env_config.num_workers == 1:
        env_batch_size = num_envs

    print(f"Creating {num_envs} environments...")
    vecenv = make_vecenv(
        curriculum,
        "serial",  # Use serial for demo
        num_envs=num_envs,
        batch_size=env_batch_size,
        num_workers=env_config.num_workers,
        zero_copy=training_config.zero_copy,
    )

    # Reset environment
    seed = env_config.seed or np.random.randint(0, 1000000)
    vecenv.async_reset(seed)

    # Get driver environment
    metta_grid_env: MettaGridEnv = vecenv.driver_env
    assert isinstance(metta_grid_env, MettaGridEnv)

    # 6. Create or load policy
    print("Creating policy...")

    # Simple policy store without wandb
    simple_cfg = DictConfig(
        {
            "device": str(device),
            "data_dir": run_dir,
        }
    )
    policy_store = PolicyStore(simple_cfg, wandb_run=None)

    # Load checkpoint if exists
    checkpoint = TrainerCheckpoint.load(run_dir)

    if checkpoint.policy_path:
        policy_record = policy_store.load_from_uri(checkpoint.policy_path)
        print("Loaded existing policy from checkpoint")
    else:
        policy_record = policy_store.create(metta_grid_env)
        print("Created new policy")

    policy = policy_record.policy().to(device)

    # Activate actions
    actions_names = metta_grid_env.action_names
    actions_max_params = metta_grid_env.max_action_args
    policy.activate_actions(actions_names, actions_max_params, device)

    # Store uncompiled reference for saving
    uncompiled_policy = policy

    # Compile if requested
    if training_config.compile:
        print("Compiling policy...")
        policy = torch.compile(policy, mode=training_config.compile_mode)

    # 7. Create experience buffer
    obs_space = vecenv.single_observation_space
    atn_space = vecenv.single_action_space
    total_agents = vecenv.num_agents
    hidden_size = getattr(policy, "hidden_size", 256)
    num_lstm_layers = 2

    experience = Experience(
        total_agents=total_agents,
        batch_size=training_config.batch_size,
        bptt_horizon=training_config.bptt_horizon,
        minibatch_size=training_config.minibatch_size,
        max_minibatch_size=training_config.minibatch_size,
        obs_space=obs_space,
        atn_space=atn_space,
        device=device,
        hidden_size=hidden_size,
        cpu_offload=training_config.cpu_offload,
        num_lstm_layers=num_lstm_layers,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # 8. Create optimizer
    optimizer = create_optimizer(optimizer_config, policy)

    # Load optimizer state if resuming
    if checkpoint.agent_step > 0:
        try:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            print("Loaded optimizer state from checkpoint")
        except ValueError as e:
            print(f"Could not load optimizer state: {e}")

    # 9. Initialize training state
    agent_step = checkpoint.agent_step
    epoch = checkpoint.epoch
    losses = Losses()
    timer = Stopwatch(None)  # No logger for simple demo
    timer.start()

    # Create minimal config for agent clip range
    agent_cfg = DictConfig({"agent": {"clip_range": 0}})

    # Optional: Kickstarter (set to None for demo)
    kickstarter = None

    print(f"\nStarting training from epoch {epoch}, step {agent_step}")
    print("=" * 50)

    # 10. Main training loop
    while agent_step < training_config.total_timesteps:
        steps_before = agent_step

        # Rollout phase
        with timer("rollout"):
            agent_step, stats = rollout(
                policy=policy, vecenv=vecenv, experience=experience, device=device, agent_step=agent_step, timer=timer
            )

        # Training phase
        with timer("train"):
            epoch = train_ppo(
                policy=policy,
                optimizer=optimizer,
                experience=experience,
                device=device,
                losses=losses,
                epoch=epoch,
                cfg=agent_cfg,
                lr_scheduler=None,
                timer=timer,
                # PPO parameters
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda,
                clip_coef=ppo_config.clip_coef,
                ent_coef=ppo_config.ent_coef,
                vf_coef=ppo_config.vf_coef,
                max_grad_norm=ppo_config.max_grad_norm,
                norm_adv=ppo_config.norm_adv,
                clip_vloss=ppo_config.clip_vloss,
                vf_clip_coef=ppo_config.vf_clip_coef,
                update_epochs=training_config.update_epochs,
                target_kl=ppo_config.target_kl,
                kickstarter=kickstarter,
                agent_step=agent_step,
                l2_reg_loss_coef=training_config.l2_reg_loss_coef,
                l2_init_loss_coef=training_config.l2_init_loss_coef,
                clip_range=0,  # From agent config
                # Prioritized replay
                prio_alpha=replay_config.prio_alpha,
                prio_beta0=replay_config.prio_beta0,
                total_timesteps=training_config.total_timesteps,
                batch_size=training_config.batch_size,
                # V-trace
                vtrace_rho_clip=ppo_config.vtrace_rho_clip,
                vtrace_c_clip=ppo_config.vtrace_c_clip,
            )

        # Calculate and display metrics
        rollout_time = timer.get_last_elapsed("rollout")
        train_time = timer.get_last_elapsed("train")
        total_time = train_time + rollout_time
        steps_calculated = agent_step - steps_before
        steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

        train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
        rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

        # Get loss statistics
        loss_stats = losses.stats()

        print(
            f"Epoch {epoch:4d} | "
            f"Steps: {agent_step:6d}/{training_config.total_timesteps} | "
            f"SPS: {steps_per_sec:5.0f} | "
            f"Loss: {loss_stats.get('policy_loss', 0):.4f} | "
            f"Value: {loss_stats.get('value_loss', 0):.4f} | "
            f"Time: {train_pct:.0f}% train, {rollout_pct:.0f}% rollout"
        )

        # Checkpoint
        if epoch % training_config.checkpoint_interval == 0:
            print(f"Saving checkpoint at epoch {epoch}...")

            # Save policy
            name = policy_store.make_model_name(epoch)
            path = os.path.join(checkpoint_dir, name)
            pr = policy_store.save(
                name=name,
                path=path,
                policy=uncompiled_policy,
                metadata={
                    "epoch": epoch,
                    "agent_step": agent_step,
                    "loss": loss_stats.get("policy_loss", 0),
                },
            )

            # Save trainer state
            checkpoint = TrainerCheckpoint(
                agent_step=agent_step,
                epoch=epoch,
                total_agent_step=agent_step,
                optimizer_state_dict=optimizer.state_dict(),
                policy_path=pr.uri if pr else None,
                extra_args={},
            )
            checkpoint.save(run_dir)

    # Final checkpoint
    print("\nTraining complete! Saving final checkpoint...")
    name = policy_store.make_model_name(epoch)
    path = os.path.join(checkpoint_dir, name)
    pr = policy_store.save(
        name=name,
        path=path,
        policy=uncompiled_policy,
        metadata={
            "epoch": epoch,
            "agent_step": agent_step,
            "final": True,
        },
    )

    checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        total_agent_step=agent_step,
        optimizer_state_dict=optimizer.state_dict(),
        policy_path=pr.uri if pr else None,
        extra_args={},
    )
    checkpoint.save(run_dir)

    # Cleanup
    vecenv.close()

    print(f"\n✅ Demo complete! Final model saved to: {path}")
    print(f"Total training time: {timer.get_elapsed():.1f}s")


if __name__ == "__main__":
    main()
