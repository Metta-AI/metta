"""Optimized configuration for hardware scaling sweeps with proper defaults.

This configuration uses:
- No random samples (starts from search center immediately)
- Seed with search center (start from the mean values)
- Exact default values from the codebase for non-batch parameters
"""

import math
import os

from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings


def create_optimized_protein_config(
    gpus: int,
    nodes: int,
    num_agents: int = 24,
    bptt_horizon: int = 64,  # Default from TrainerConfig
    forward_pass_minibatch_target_size: int = 4096,
    async_factor: int = 2,
    metric_path: str = "evaluator/eval_arena/score",
) -> ProteinConfig:
    """Create an optimized Protein configuration for a specific hardware setup.

    Uses the exact default values from the codebase as search centers for
    non-batch parameters, and calculates appropriate batch size constraints
    based on the hardware configuration.

    Args:
        gpus: Number of GPUs
        nodes: Number of nodes
        num_agents: Number of agents per environment
        bptt_horizon: BPTT horizon for sequence processing (default 64 from TrainerConfig)
        forward_pass_minibatch_target_size: Target size for forward pass
        async_factor: Async factor for environment parallelization
        metric_path: Metric path to optimize

    Returns:
        ProteinConfig with appropriate hyperparameter search space and defaults
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
    max_batch_size = max(min_batch_size * 8, 8192)

    # Minibatch constraints
    # Must be divisible by bptt_horizon and reasonable for training
    min_minibatch = max(128, bptt_horizon * 16)  # At least 16 segments
    max_minibatch = min(
        max_batch_size // 4, 65536
    )  # At most 1/4 of batch size, cap at 64k

    # Default minibatch is typically batch_size / 4 or batch_size / 8
    default_minibatch = min(max(min_batch_size // 4, min_minibatch), max_minibatch)
    # Round to power of 2
    default_minibatch = 2 ** round(math.log2(default_minibatch))

    return ProteinConfig(
        metric=metric_path,
        goal="maximize",
        method="bayes",
        parameters={
            # Batch sizes - hardware dependent
            "trainer.batch_size": ParameterConfig(
                min=float(min_batch_size),
                max=float(max_batch_size),
                distribution="uniform_pow2",  # Powers of 2
                mean=float(min_batch_size * 2),  # Start at 2x minimum
                scale="auto",
            ),
            # "trainer.minibatch_size": ParameterConfig(
            #     min=float(min_minibatch),
            #     max=float(max_minibatch),
            #     distribution="uniform_pow2",  # Powers of 2
            #     mean=float(default_minibatch),
            #     scale="auto",
            # ),
            # Learning rate (narrowed around the default)
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=5.768185e-4,     # ~0.5x default
                max=2.307274e-3,     # ~2x default
                distribution="log_normal",
                mean=0.001153637,    # Exact default from TrainerConfig
                scale="auto",
            ),

            # PPO stability/efficiency (narrowed)
            "trainer.losses.loss_configs.ppo.clip_coef": ParameterConfig(
                min=0.15,
                max=0.30,
                distribution="uniform",
                mean=0.24,
                scale="auto",
            ),
            "trainer.losses.loss_configs.ppo.vf_clip_coef": ParameterConfig(
                min=0.10,
                max=0.30,
                distribution="uniform",
                mean=0.20,
                scale="auto",
            ),
            "trainer.losses.loss_configs.ppo.gae_lambda": ParameterConfig(
                min=0.90,
                max=0.97,
                distribution="uniform",
                mean=0.94,
                scale="auto",
            ),
            "trainer.losses.loss_configs.ppo.ent_coef": ParameterConfig(
                min=0.001,
                max=0.03,
                distribution="log_normal",
                mean=0.01,
                scale="auto",
            ),
            "trainer.losses.loss_configs.ppo.vf_coef": ParameterConfig(
                min=0.5,
                max=1.2,
                distribution="uniform",
                mean=0.9,
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            # Key settings as requested
            num_random_samples=0,  # No random samples - use Bayesian optimization immediately
            seed_with_search_center=True,  # Start from the mean values
            # Other Protein settings with good defaults
            global_search_scale=1.0,
            # Cost and suggestion settings
            max_suggestion_cost=10800,  # 3 hours max per run
            resample_frequency=5,  # Don't resample failed runs
            # Batch suggestion settings
            random_suggestions=256,  # Reduced from default 1024 since we're not using random
            suggestions_per_pareto=128,  # Reduced from default 256 for faster iteration
        ),
    )
