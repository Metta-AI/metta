"""Sweep experiments for arena environment."""

from experiments.recipes.arena import (
    make_evals,
    train as arena_train_factory,
    evaluate as default_arena_eval,
)
from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.sweep.sweep_config import SweepConfig
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from metta.tools.sim import SimTool


def sweep_hpo_1b(sweep_name: str, num_trials: int = 50) -> SweepTool:
    """Create a full sweep for optimizing PPO hyperparameters on arena with 1B timesteps.

    This version runs training for 1 billion timesteps (~2.3 hours at 120k sps).
    Suitable for serious hyperparameter optimization.

    Args:
        sweep_name: Name for this sweep
        num_trials: Number of trials to run (default 50 for thorough search)

    Returns:
        Configured SweepTool for PPO hyperparameter optimization
    """
    # Define parameters to optimize - same as quick but with adjusted settings
    protein_config = PPO_STANDARD_SWEEP

    # Create evaluation simulations for end-of-training
    eval_simulations = make_evals()
    for sim in eval_simulations:
        sim.num_episodes = 10  # More episodes for accurate evaluation
        sim.max_time_s = 300  # Longer timeout

    # Create the factory function that generates TrainTools
    def train_factory(run_name: str) -> TrainTool:
        # Use the arena train factory as base
        train_tool = arena_train_factory()
        # Set the run name
        train_tool.run = run_name
        # Set to 1 billion timesteps
        train_tool.trainer.total_timesteps = 1_000_000_000

        # Adjust checkpoint intervals for long runs
        train_tool.trainer.checkpoint.checkpoint_interval = (
            50_000_000  # Every 50M steps
        )
        train_tool.trainer.checkpoint.wandb_checkpoint_interval = (
            50_000_000  # Every 50M steps
        )

        # ENSURE LOCAL EVALUATIONS ONLY
        if train_tool.trainer.evaluation:
            train_tool.trainer.evaluation.evaluate_remote = False
            train_tool.trainer.evaluation.evaluate_local = True
            train_tool.trainer.evaluation.evaluate_interval = (
                50_000_000  # Evaluate every 50M steps
            )

        return train_tool

    # Run arena evaluations for 5 episodes
    eval_simulations = make_evals()
    for sim in eval_simulations:
        sim.num_episodes = 5  # Run 5 episodes for thorough evaluation
        sim.max_time_s = 120
    # Create sweep configuration
    sweep_config = SweepConfig(
        num_trials=num_trials,
        protein=protein_config,
        sweep_name=sweep_name,
        max_observations_to_load=500,  # Load more observations for longer runs
    )

    return SweepTool(
        sweep=sweep_config,
        sweep_name=sweep_name,
        train_tool_factory=train_factory,
        simulations=eval_simulations,
    )


def sweep_hpo_quick(sweep_name: str, num_trials: int = 10) -> SweepTool:
    """Create a quick sweep for optimizing PPO hyperparameters on arena.

    This version runs training for only 100,000 timesteps for rapid testing.

    Args:
        sweep_name: Name for this sweep
        num_trials: Number of trials to run

    Returns:
        Configured SweepTool for PPO hyperparameter optimization
    """
    # Define parameters to optimize
    protein_config = PPO_STANDARD_SWEEP

    # Create evaluation simulations for end-of-training
    # Run arena evaluations for 5 episodes
    eval_simulations = make_evals()
    for sim in eval_simulations:
        sim.num_episodes = 5  # Run 5 episodes for thorough evaluation
        sim.max_time_s = 120

    # Create the factory function that generates TrainTools
    # TODO We can actually use this factory to change the parameters here directly if we wantted.
    def train_factory(run_name: str) -> TrainTool:
        # Use the arena train factory as base
        train_tool = arena_train_factory()
        # Set the run name
        train_tool.run = run_name
        # Override to run for only 10,000 timesteps for quick testing
        train_tool.trainer.total_timesteps = 10000

        # ENSURE LOCAL EVALUATIONS ONLY
        if train_tool.trainer.evaluation:
            train_tool.trainer.evaluation.evaluate_remote = False
            train_tool.trainer.evaluation.evaluate_local = True

        return train_tool

    # Create sweep configuration
    sweep_config = SweepConfig(
        num_trials=num_trials,
        protein=protein_config,
        sweep_name=sweep_name,
    )
    return SweepTool(
        sweep=sweep_config,
        sweep_name=sweep_name,
        train_tool_factory=train_factory,
        simulations=eval_simulations,
    )


# Protein Conf igs
PPO_STANDARD_SWEEP = ProteinConfig(
    metric="eval_arena",  # Optimize for arena performance (this is the actual metric name in evaluator output)
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
    settings=ProteinSettings(
        num_random_samples=30,  # More random samples for longer runs
        max_suggestion_cost=3600,  # 1 hour max per trial
    ),
)
