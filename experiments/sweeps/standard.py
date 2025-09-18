"""Standard sweep configurations using the adaptive module."""

from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.tools.sweep import SweepTool
from metta.tools.sweep import DispatcherType


def protein(
    recipe: str = "experiments.recipes.arena",
    train: str = "train",
    eval: str = "evaluate",
    protein_config: ProteinConfig | None = None,
    max_trials: int = 300,
    max_parallel_jobs: int = 6,
    gpus: int = 1,
    batch_size: int = 4,
    local_test: bool = False,
) -> SweepTool:
    """Create PPO hyperparameter sweep using adaptive infrastructure.

    Args:
        recipe: Recipe module to use for training and evaluation
        train: Training entrypoint name
        eval: Evaluation entrypoint name
        protein_cfg: Protein configuration
        max_trials: Maximum number of trials to run
        max_parallel_jobs: Maximum parallel jobs
        gpus: Number of GPUs per job
        batch_size: Number of suggestions per batch
        local_test: If True, use local dispatcher with 50k timesteps for testing

    Returns:
        Configured SweepTool for PPO hyperparameter optimization
    """

    # Define the 6 PPO parameters to sweep over
    # Import DispatcherType for local testing
    assert protein_config is not None, "protein_config must be provided"

    # Configure based on local_test flag
    if local_test:
        # Local testing configuration
        dispatcher_type = DispatcherType.LOCAL
        total_timesteps = 50000  # Quick 50k timesteps for testing
        monitoring_interval = 30  # Check more frequently for local testing

        # We let the batch size be set in training for the quick run
        if "batch_size" in protein_config.parameters:
            protein_config.parameters["batch_size"] = None

        if "minibatch_size" in protein_config.parameters:
            protein_config.parameters["minibatch_size"] = None
    else:
        # Production configuration
        dispatcher_type = DispatcherType.SKYPILOT
        total_timesteps = 2000000000  # 2B timesteps for production
        monitoring_interval = 60

    # Create and return the sweep tool using adaptive infrastructure
    return SweepTool(
        protein_config=protein_config,
        max_trials=max_trials,
        batch_size=batch_size,
        recipe_module=recipe,
        train_entrypoint=train,
        eval_entrypoint=eval,
        monitoring_interval=monitoring_interval,
        max_parallel_jobs=max_parallel_jobs,
        gpus=gpus,
        dispatcher_type=dispatcher_type,
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )
