"""Standard sweep configurations using the adaptive module."""

from experiments.sweeps.protein_configs import PPO_BASIC
from metta.sweep.protein_config import ProteinConfig
from metta.tools.sweep import SweepTool


def protein_sweep(
    recipe: str = "experiments.recipes.arena",
    train: str = "train",
    eval: str = "evaluate",
    protein_config: ProteinConfig = PPO_BASIC,
    total_timesteps: int = 1000000,
) -> SweepTool:
    """Create PPO hyperparameter sweep using adaptive infrastructure.

    Args:
        recipe: Recipe module to use for training and evaluation
        train: Training entrypoint name
        eval: Evaluation entrypoint name
        protein_cfg: Protein configuration
        total_timesteps: Total timesteps for training

    Returns:
        Configured SweepTool for PPO hyperparameter optimization
    """

    # Define the 6 PPO parameters to sweep over
    # Import DispatcherType for local testing
    assert protein_config is not None, "protein_config must be provided"

    # Create and return the sweep tool using adaptive infrastructure
    return SweepTool(
        protein_config=protein_config,
        recipe_module=recipe,
        train_entrypoint=train,
        eval_entrypoint=eval,
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )
