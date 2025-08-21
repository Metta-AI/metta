"""Sweep experiments for arena environment."""

from experiments.recipes.arena import make_evals, train as arena_train_factory
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.sweep_config import SweepConfig
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool


def sweep_hpo(sweep_name: str, num_trials: int = 10) -> SweepTool:
    """Create a sweep for optimizing PPO hyperparameters on arena.

    Args:
        sweep_name: Name for this sweep
        num_trials: Number of trials to run

    Returns:
        Configured SweepTool for PPO hyperparameter optimization
    """
    # Define parameters to optimize
    protein_config = ProteinConfig(
        metric="eval_arena",  # Optimize for arena performance
        goal="maximize",
        method="bayes",
        parameters={
            "trainer": {
                "optimizer": {
                    "learning_rate": ParameterConfig(
                        min=1e-5,
                        max=1e-2,
                        distribution="log_normal",
                        mean=1e-3,  # Geometric mean
                        scale="auto",
                    ),
                },
                "ppo": {
                    "clip_coef": ParameterConfig(
                        min=0.05,
                        max=0.3,
                        distribution="uniform",
                        mean=0.175,
                        scale="auto",
                    ),
                    "ent_coef": ParameterConfig(
                        min=0.0001,
                        max=0.01,
                        distribution="log_normal",
                        mean=0.001,  # Geometric mean
                        scale="auto",
                    ),
                    "gae_lambda": ParameterConfig(
                        min=0.8,
                        max=0.99,
                        distribution="uniform",
                        mean=0.895,
                        scale="auto",
                    ),
                    "gamma": ParameterConfig(
                        min=0.95,
                        max=0.999,
                        distribution="uniform",
                        mean=0.9745,
                        scale="auto",
                    ),
                    "vf_coef": ParameterConfig(
                        min=0.1,
                        max=1.0,
                        distribution="uniform",
                        mean=0.55,
                        scale="auto",
                    ),
                },
            }
        },
    )

    # Create evaluation simulations for end-of-training
    # Run arena evaluations for 5 episodes
    eval_simulations = make_evals()
    for sim in eval_simulations:
        sim.num_episodes = 5  # Run 5 episodes for thorough evaluation

    # Create sweep configuration
    sweep_config = SweepConfig(
        num_trials=num_trials,
        protein=protein_config,
        evaluation_simulations=eval_simulations,  # Arena evals with 5 episodes
    )

    # Create the factory function that generates TrainTools
    def train_factory(run_name: str) -> TrainTool:
        # Use the arena train factory as base
        train_tool = arena_train_factory()
        # Set the run name
        train_tool.run = run_name
        return train_tool

    return SweepTool(
        sweep=sweep_config,
        sweep_name=sweep_name,
        train_tool_factory=train_factory,
    )


def sweep_hpo_quick(sweep_name: str, num_trials: int = 10) -> SweepTool:
    """Create a quick sweep for optimizing PPO hyperparameters on arena.

    This version runs training for only 10,000 timesteps for rapid testing.

    Args:
        sweep_name: Name for this sweep
        num_trials: Number of trials to run

    Returns:
        Configured SweepTool for PPO hyperparameter optimization
    """
    # Define parameters to optimize
    protein_config = ProteinConfig(
        metric="eval_arena",  # Optimize for arena performance
        goal="maximize",
        method="bayes",
        parameters={
            "trainer": {
                "optimizer": {
                    "learning_rate": ParameterConfig(
                        min=1e-5,
                        max=1e-2,
                        distribution="log_normal",
                        mean=1e-3,  # Geometric mean
                        scale="auto",
                    ),
                },
                "ppo": {
                    "clip_coef": ParameterConfig(
                        min=0.05,
                        max=0.3,
                        distribution="uniform",
                        mean=0.175,
                        scale="auto",
                    ),
                    "ent_coef": ParameterConfig(
                        min=0.0001,
                        max=0.01,
                        distribution="log_normal",
                        mean=0.001,  # Geometric mean
                        scale="auto",
                    ),
                    "gae_lambda": ParameterConfig(
                        min=0.8,
                        max=0.99,
                        distribution="uniform",
                        mean=0.895,
                        scale="auto",
                    ),
                    "gamma": ParameterConfig(
                        min=0.95,
                        max=0.999,
                        distribution="uniform",
                        mean=0.9745,
                        scale="auto",
                    ),
                    "vf_coef": ParameterConfig(
                        min=0.1,
                        max=1.0,
                        distribution="uniform",
                        mean=0.55,
                        scale="auto",
                    ),
                },
            }
        },
    )

    # Create evaluation simulations for end-of-training
    # Run arena evaluations for 5 episodes
    eval_simulations = make_evals()
    for sim in eval_simulations:
        sim.num_episodes = 5  # Run 5 episodes for thorough evaluation
        sim.max_time_s = 120

    # Create sweep configuration
    sweep_config = SweepConfig(
        num_trials=num_trials,
        protein=protein_config,
        evaluation_simulations=eval_simulations,  # Arena evals with 5 episodes
    )

    # Create the factory function that generates TrainTools
    def train_factory(run_name: str) -> TrainTool:
        # Use the arena train factory as base
        train_tool = arena_train_factory()
        # Set the run name
        train_tool.run = run_name
        # Override to run for only 10,000 timesteps for quick testing
        train_tool.trainer.total_timesteps = 10000
        return train_tool

    return SweepTool(
        sweep=sweep_config,
        sweep_name=sweep_name,
        train_tool_factory=train_factory,
    )
