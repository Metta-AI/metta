"""Figure of Merit hardware scaling sweep using simplified API.

The FoM experiment sweeps over different hardware configurations (GPUs x nodes),
with each configuration getting its own optimized hyperparameter sweep.
"""

import math
import os

from metta.sweep.core import make_sweep, SweepParameters as SP, Distribution as D
from metta.tools.sweep import SweepTool


def calculate_hardware_specific_batch_range(
    gpus: int,
    nodes: int,
    num_agents: int = 24,
    bptt_horizon: int = 64,
    forward_pass_minibatch_target_size: int = 4096,
    async_factor: int = 2,
) -> tuple[float, float, float]:
    """Calculate min, max, and default batch size for given hardware.

    Uses the exact calculations from optimized_sweep_config.py.
    """
    # Calculate minimum batch size for this hardware
    total_gpus = gpus * nodes

    # CPU-related (environment parallelization)
    cpu_count = os.cpu_count() or 8  # Assume 8 CPUs per node if not available
    num_workers = max(1, cpu_count // 2)  # Half CPUs for workers

    # Environment calculation (CPU-side) per GPU
    target_batch = forward_pass_minibatch_target_size // num_agents
    batch_size_per_gpu = (target_batch // num_workers) * num_workers
    num_envs = batch_size_per_gpu * async_factor

    # Total agents PER GPU
    total_agents_per_gpu = num_envs * num_agents

    # Minimum batch size PER GPU
    min_batch_size_per_gpu = total_agents_per_gpu * bptt_horizon

    # Total minimum batch size across all GPUs
    min_batch_size = min_batch_size_per_gpu * total_gpus

    # Round up to nearest power of 2
    min_batch_size = 2 ** math.ceil(math.log2(min_batch_size))

    # Max batch size is 8x minimum (as specified in original requirements)
    # Cap at 8192 for memory stability
    max_batch_size = min_batch_size * 8

    # Default batch is 2x minimum (from optimized config)
    default_batch = min_batch_size * 2

    return float(min_batch_size), float(max_batch_size), float(default_batch)


def sweep(
    sweep_name: str,
    gpus: int,
    nodes: int,
    num_trials: int = 50,
    total_timesteps: int = 300_000_000,
) -> SweepTool:
    """Create a FoM sweep for specific hardware configuration.

    Args:
        sweep_name: Name for this sweep
        gpus: GPUs per node
        nodes: Number of nodes
        num_trials: Trials to run
        total_timesteps: Timesteps per trial

    Returns:
        Configured SweepTool
    """
    # Calculate hardware-specific batch constraints
    min_batch, max_batch, default_batch = calculate_hardware_specific_batch_range(
        gpus, nodes
    )

    # Define parameters to sweep - using exact ranges from optimized_sweep_config.py
    parameters = [
        # Hardware-scaled batch size
        SP.param(
            "trainer.batch_size",
            D.UNIFORM_POW2,
            min=min_batch,
            max=max_batch,
            search_center=default_batch,
        ),
        # Learning rate (narrowed around the default from TrainerConfig)
        SP.param(
            "trainer.optimizer.learning_rate",
            D.LOG_NORMAL,
            min=5.768185e-4,  # ~0.5x default
            max=2.307274e-3,  # ~2x default
            search_center=0.001153637,  # Exact default from TrainerConfig
        ),
        # PPO stability/efficiency parameters (narrowed ranges from optimized config)
        SP.param(
            "trainer.losses.loss_configs.ppo.clip_coef",
            D.UNIFORM,
            min=0.15,
            max=0.30,
            search_center=0.24,
        ),
        SP.param(
            "trainer.losses.loss_configs.ppo.vf_clip_coef",
            D.UNIFORM,
            min=0.10,
            max=0.30,
            search_center=0.20,
        ),
        SP.param(
            "trainer.losses.loss_configs.ppo.gae_lambda",
            D.UNIFORM,
            min=0.90,
            max=0.97,
            search_center=0.94,
        ),
        SP.param(
            "trainer.losses.loss_configs.ppo.ent_coef",
            D.LOG_NORMAL,
            min=0.001,
            max=0.03,
            search_center=0.01,
        ),
        SP.param(
            "trainer.losses.loss_configs.ppo.vf_coef",
            D.UNIFORM,
            min=0.5,
            max=1.2,
            search_center=0.9,
        ),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        objective="evaluator/eval_sweep/score",
        parameters=parameters,
        num_trials=num_trials,
        num_parallel_trials=4,
        resources={"gpus": gpus, "nodes": nodes},
        train_overrides={"trainer.total_timesteps": total_timesteps},
        # FoM-specific settings
        max_concurrent_evals=2,
    )
