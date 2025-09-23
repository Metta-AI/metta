"""Optimized configuration for hardware scaling sweeps with proper defaults.

This configuration uses:
- No random samples (starts from search center immediately)
- Seed with search center (start from the mean values)
- Exact default values from the codebase for non-batch parameters
"""

import math
import os
from typing import Optional

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
    max_batch_size = min_batch_size * 8

    # Minibatch constraints
    # Must be divisible by bptt_horizon and reasonable for training
    min_minibatch = max(256, bptt_horizon * 16)  # At least 16 segments
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
            "trainer.minibatch_size": ParameterConfig(
                min=float(min_minibatch),
                max=float(max_minibatch),
                distribution="uniform_pow2",  # Powers of 2
                mean=float(default_minibatch),
                scale="auto",
            ),
            # Learning rate - exact default from trainer_config.py
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-2,
                distribution="log_normal",
                mean=0.001153637,  # Exact default from TrainerConfig
                scale="auto",
            ),
            # PPO clip coefficient - exact default from ppo.py
            "trainer.losses.ppo.clip_coef": ParameterConfig(
                min=0.1,
                max=0.4,
                distribution="uniform",
                mean=0.264407,  # Exact default from PPOLossConfig
                scale="auto",
            ),
            # Value function clip coefficient - exact default from ppo.py
            "trainer.losses.ppo.vf_clip_coef": ParameterConfig(
                min=0.05,
                max=1.0,
                distribution="uniform",
                mean=0.1,  # Exact default from PPOLossConfig
                scale="auto",
            ),
            # GAE lambda - exact default from ppo.py
            "trainer.losses.ppo.gae_lambda": ParameterConfig(
                min=0.8,
                max=0.99,
                distribution="uniform",
                mean=0.891477,  # Exact default from PPOLossConfig
                scale="auto",
            ),
            # Additional parameters you might want to sweep
            # Entropy coefficient - exact default from ppo.py
            "trainer.losses.ppo.ent_coef": ParameterConfig(
                min=0.0001,
                max=0.1,
                distribution="log_normal",
                mean=0.01,  # Exact default from PPOLossConfig
                scale="auto",
            ),
            # Value function coefficient - exact default from ppo.py
            "trainer.losses.ppo.vf_coef": ParameterConfig(
                min=0.1,
                max=2.0,
                distribution="uniform",
                mean=0.897619,  # Exact default from PPOLossConfig
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            # Key settings as requested
            num_random_samples=0,  # No random samples - use Bayesian optimization immediately
            seed_with_search_center=True,  # Start from the mean values
            # Other Protein settings with good defaults
            global_search_scale=1.0,
            acquisition_fn="ei",  # Expected Improvement is often better than naive
            ucb_beta=2.0,  # For UCB acquisition (if used)
            randomize_acquisition=False,  # Deterministic for reproducibility
            # Cost and suggestion settings
            max_suggestion_cost=10800,  # 3 hours max per run
            resample_frequency=0,  # Don't resample failed runs
            expansion_rate=0.25,  # For naive acquisition
            # Batch suggestion settings
            random_suggestions=256,  # Reduced from default 1024 since we're not using random
            suggestions_per_pareto=128,  # Reduced from default 256 for faster iteration
        ),
    )


def get_simplified_protein_config(
    gpus: int,
    nodes: int,
    num_agents: int = 24,
    bptt_horizon: int = 16,
) -> ProteinConfig:
    """Get a simplified Protein config with fewer parameters to sweep.

    This version only sweeps the most important parameters:
    - batch_size
    - minibatch_size
    - learning_rate
    - ppo_clip_coef
    - gae_lambda
    """
    # Calculate batch size constraints
    total_gpus = gpus * nodes
    min_batch_size = num_agents * bptt_horizon * 64 * total_gpus
    min_batch_size = 2 ** math.ceil(math.log2(min_batch_size))
    max_batch_size = min_batch_size * 8

    min_minibatch = max(256, bptt_horizon * 16)
    max_minibatch = min(max_batch_size // 4, 65536)
    default_minibatch = 2 ** round(
        math.log2(min(max(min_batch_size // 4, min_minibatch), max_minibatch))
    )

    return ProteinConfig(
        metric="evaluator/eval_arena/score",
        goal="maximize",
        method="bayes",
        parameters={
            "trainer.batch_size": ParameterConfig(
                min=float(min_batch_size),
                max=float(max_batch_size),
                distribution="uniform_pow2",
                mean=float(min_batch_size * 2),
                scale="auto",
            ),
            "trainer.minibatch_size": ParameterConfig(
                min=float(min_minibatch),
                max=float(max_minibatch),
                distribution="uniform_pow2",
                mean=float(default_minibatch),
                scale="auto",
            ),
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5,
                max=1e-2,
                distribution="log_normal",
                mean=0.001153637,  # Exact default
                scale="auto",
            ),
            "trainer.losses.ppo.clip_coef": ParameterConfig(
                min=0.1,
                max=0.4,
                distribution="uniform",
                mean=0.264407,  # Exact default
                scale="auto",
            ),
            "trainer.losses.ppo.gae_lambda": ParameterConfig(
                min=0.8,
                max=0.99,
                distribution="uniform",
                mean=0.891477,  # Exact default
                scale="auto",
            ),
        },
        settings=ProteinSettings(
            num_random_samples=0,
            seed_with_search_center=True,
            global_search_scale=1.0,
            acquisition_fn="ei",
            max_suggestion_cost=10800,
            random_suggestions=256,
            suggestions_per_pareto=128,
        ),
    )
