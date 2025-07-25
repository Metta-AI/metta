#!/usr/bin/env -S uv run
"""Simplified run.py using the component-based Trainer class."""

import logging
import os

import torch
from omegaconf import DictConfig

from metta.interface.directories import setup_run_directories
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.rl.components import Trainer
from metta.rl.trainer_config import (
    CheckpointConfig,
    InitialPolicyConfig,
    KickstartConfig,
    OptimizerConfig,
    PPOConfig,
    PrioritizedExperienceReplayConfig,
    SimulationConfig,
    TorchProfilerConfig,
    TrainerConfig,
    VTraceConfig,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set up directories
dirs = setup_run_directories()

# Core training parameters
num_workers = 4
total_timesteps = 10_000_000
batch_size = 524288 if torch.cuda.is_available() else 16384  # 512k for GPU, 16k for CPU
minibatch_size = 16384 if torch.cuda.is_available() else 1024  # 16k for GPU, 1k for CPU
curriculum = "/env/mettagrid/curriculum/navigation/bucketed"
bptt_horizon = 64
update_epochs = 1
forward_pass_minibatch_target_size = 4096 if torch.cuda.is_available() else 256

# Adjust defaults based on vectorization mode
vectorization_mode = "multiprocessing"
if vectorization_mode == "serial":
    async_factor = 1
    zero_copy = False
else:
    async_factor = 2
    zero_copy = True

grad_mean_variance_interval = 150
scale_batches_by_world_size = False
cpu_offload = False

# Individual component configs
ppo_config = PPOConfig(
    clip_coef=0.1,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
)

optimizer_config = OptimizerConfig(
    type="adam",
    learning_rate=3e-4,
)

checkpoint_config = CheckpointConfig(
    checkpoint_dir=dirs.checkpoint_dir,
    checkpoint_interval=300,
    wandb_checkpoint_interval=0,
)

simulation_config = SimulationConfig(
    evaluate_interval=300,
    replay_dir=dirs.replay_dir,
)

profiler_config = TorchProfilerConfig(
    interval_epochs=0,
    profile_dir=os.path.join(dirs.run_dir, "torch_traces"),
)

prioritized_replay_config = PrioritizedExperienceReplayConfig()
vtrace_config = VTraceConfig()
kickstart_config = KickstartConfig()

# Check for initial policy URI from environment variable
initial_policy_uri = os.environ.get("INITIAL_POLICY_URI", None)
initial_policy_config = InitialPolicyConfig(uri=initial_policy_uri)

# Create trainer config
trainer_config = TrainerConfig(
    num_workers=num_workers,
    total_timesteps=total_timesteps,
    batch_size=batch_size,
    minibatch_size=minibatch_size,
    curriculum=curriculum,
    bptt_horizon=bptt_horizon,
    update_epochs=update_epochs,
    forward_pass_minibatch_target_size=forward_pass_minibatch_target_size,
    async_factor=async_factor,
    grad_mean_variance_interval=grad_mean_variance_interval,
    scale_batches_by_world_size=scale_batches_by_world_size,
    cpu_offload=cpu_offload,
    zero_copy=zero_copy,
    ppo=ppo_config,
    optimizer=optimizer_config,
    checkpoint=checkpoint_config,
    simulation=simulation_config,
    profiler=profiler_config,
    prioritized_experience_replay=prioritized_replay_config,
    vtrace=vtrace_config,
    kickstart=kickstart_config,
    initial_policy=initial_policy_config,
)

# WandB configuration
wandb_enabled = os.environ.get("WANDB_DISABLED", "").lower() != "true"

if wandb_enabled:
    wandb_config = DictConfig(
        {
            "enabled": True,
            "project": os.environ.get("WANDB_PROJECT", "metta"),
            "entity": os.environ.get("WANDB_ENTITY", "metta-research"),
            "group": dirs.run_name,
            "name": dirs.run_name,
            "run_id": dirs.run_name,
            "data_dir": dirs.run_dir,
            "job_type": "train",
            "tags": [],
            "notes": "",
        }
    )
else:
    wandb_config = DictConfig({"enabled": False})

global_config = DictConfig(
    {
        "run": dirs.run_name,
        "run_dir": dirs.run_dir,
        "cmd": "train",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 1,
        "trainer": trainer_config.model_dump(),
        "train_job": {"evals": {}},
        "wandb": wandb_config,
    }
)

# Create and run trainer
trainer = Trainer(
    trainer_config=trainer_config,
    run_dir=dirs.run_dir,
    run_name=dirs.run_name,
    checkpoint_dir=dirs.checkpoint_dir,
    replay_dir=dirs.replay_dir,
    stats_dir=dirs.stats_dir,
    wandb_config=wandb_config,
    global_config=global_config,
)

try:
    # Set up trainer components
    trainer.setup(vectorization=vectorization_mode)

    # Run training
    trainer.train()

finally:
    # Clean up
    trainer.cleanup()
