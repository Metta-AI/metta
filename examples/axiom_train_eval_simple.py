"""Simple, clean example of train and eval experiments."""

from metta.mettagrid import EnvConfig
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.axiom.experiment_spec import AxiomControls
from metta.sweep.axiom.train_and_eval import (
    TrainAndEvalExperiment,
    TrainAndEvalSpec,
)


def example_basic_usage():
    """Simplest possible usage."""
    print("\n" + "=" * 60)
    print("Basic Train + Eval")
    print("=" * 60)
    
    # Create spec with typed configs
    trainer = TrainerConfig()
    trainer.total_timesteps = 100000
    trainer.rollout_workers = 4
    
    spec = TrainAndEvalSpec(
        name="my_experiment",
        trainer_config=trainer,
        eval_configs=[
            SimulationConfig(
                name="navigation",
                env=EnvConfig(),
                num_episodes=50,
            ),
        ],
    )
    
    # Create experiment from spec
    exp = TrainAndEvalExperiment(spec)
    
    # Prepare and run
    exp.prepare()
    result = exp.run()
    
    print(f"✓ Experiment complete: {result.manifest()['experiment']}")


def example_custom_configs():
    """Example with custom configurations."""
    print("\n" + "=" * 60)
    print("Custom Configuration")
    print("=" * 60)
    
    # Define exactly what you want
    trainer = TrainerConfig()
    trainer.total_timesteps = 500000
    trainer.rollout_workers = 8
    trainer.batch_size = 256
    trainer.optimizer.learning_rate = 1e-4
    trainer.checkpoint.checkpoint_interval = 50
    
    spec = TrainAndEvalSpec(
        name="custom_ppo",
        description="PPO with custom hyperparameters",
        trainer_config=trainer,
        eval_configs=[
            SimulationConfig(
                name="navigation",
                env=EnvConfig(),
                num_episodes=100,
            ),
            SimulationConfig(
                name="arena",
                env=EnvConfig(),
                num_episodes=100,
            ),
        ],
        controls=AxiomControls(
            seed=1337,
            enforce_determinism=True,
        ),
    )
    
    # Single unit of experiment
    exp = TrainAndEvalExperiment(spec)
    exp.prepare()
    result = exp.run()
    
    print(f"✓ Custom experiment complete")


def example_eval_only():
    """Example of evaluation only (no training)."""
    print("\n" + "=" * 60)
    print("Evaluation Only")
    print("=" * 60)
    
    # Spec with existing policy path
    spec = TrainAndEvalSpec(
        name="eval_existing",
        policy_path="./models/pretrained_policy.pt",  # Skip training
        eval_configs=[
            SimulationConfig(
                name="navigation",
                env=EnvConfig(),
                num_episodes=200,
            ),
            SimulationConfig(
                name="arena",
                env=EnvConfig(),
                num_episodes=200,
            ),
            SimulationConfig(
                name="memory",
                env=EnvConfig(),
                num_episodes=200,
            ),
        ],
    )
    
    exp = TrainAndEvalExperiment(spec)
    exp.prepare()
    result = exp.run()
    
    print(f"✓ Evaluation complete")


def example_reproducible_experiment():
    """Example of fully reproducible experiment."""
    print("\n" + "=" * 60)
    print("Reproducible Experiment")
    print("=" * 60)
    
    # Save spec for exact reproduction
    trainer = TrainerConfig()
    trainer.total_timesteps = 1000000
    trainer.rollout_workers = 4
    trainer.batch_size = 128
    trainer.optimizer.learning_rate = 3e-4
    
    spec = TrainAndEvalSpec(
        name="reproducible_v1",
        description="Fully reproducible training run",
        trainer_config=trainer,
        eval_configs=[
            SimulationConfig(
                name="navigation",
                env=EnvConfig(),
                num_episodes=100,
            ),
        ],
        controls=AxiomControls(
            seed=42,
            enforce_determinism=True,
            single_factor_enforce=False,
        ),
        run_dir="./reproducible_experiments",
    )
    
    # Save spec for later
    from metta.sweep.axiom.experiment_spec import save_experiment_spec
    save_experiment_spec(spec, "./specs/reproducible_v1.json")
    print(f"Spec saved to ./specs/reproducible_v1.json")
    
    # Run experiment
    exp = TrainAndEvalExperiment(spec)
    exp.prepare()
    result = exp.run()
    
    # Later, load and reproduce exactly
    from metta.sweep.axiom.experiment_spec import load_experiment_spec
    loaded_spec = load_experiment_spec("./specs/reproducible_v1.json")
    
    print(f"✓ Can reproduce exactly from saved spec")


def example_comparison():
    """Example of comparing two experiments."""
    print("\n" + "=" * 60)
    print("Comparing Experiments")
    print("=" * 60)
    
    # Baseline spec
    baseline_trainer = TrainerConfig()
    baseline_trainer.total_timesteps = 100000
    baseline_trainer.optimizer.learning_rate = 3e-4
    
    baseline_spec = TrainAndEvalSpec(
        name="baseline",
        trainer_config=baseline_trainer,
        eval_configs=[
            SimulationConfig(
                name="navigation",
                env=EnvConfig(),
                num_episodes=50,
            ),
        ],
        controls=AxiomControls(seed=42),
    )
    
    # Variant with different learning rate
    variant_trainer = TrainerConfig()
    variant_trainer.total_timesteps = 100000
    variant_trainer.optimizer.learning_rate = 1e-3  # Higher LR
    
    variant_spec = TrainAndEvalSpec(
        name="high_lr",
        trainer_config=variant_trainer,
        eval_configs=[
            SimulationConfig(
                name="navigation",
                env=EnvConfig(),
                num_episodes=50,
            ),
        ],
        controls=AxiomControls(seed=42),
    )
    
    # Run both
    baseline_exp = TrainAndEvalExperiment(baseline_spec)
    baseline_exp.prepare()
    baseline_result = baseline_exp.run()
    
    variant_exp = TrainAndEvalExperiment(variant_spec)
    variant_exp.prepare()
    variant_result = variant_exp.run()
    
    # Compare
    print(f"\nBaseline: lr={baseline_spec.trainer_config.optimizer.learning_rate}")
    print(f"Variant:  lr={variant_spec.trainer_config.optimizer.learning_rate}")
    
    # Show diff
    diff = baseline_exp.diff(baseline_result, variant_result)
    print(f"\nKey differences in configurations")
    
    print(f"✓ Comparison complete")


def main():
    """Run all examples."""
    print("""
Train and Eval Experiment - Clean MVP
======================================

Key principles:
1. One spec <-> One experiment (exact mapping)
2. Typed configurations (no ambiguity)
3. Single run() function (no sub-experiments)
4. Clean, minimal, presentable code
    """)
    
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Basic usage (would run with implementation): {e}")
    
    try:
        example_custom_configs()
    except Exception as e:
        print(f"Custom configs (would run with implementation): {e}")
    
    try:
        example_eval_only()
    except Exception as e:
        print(f"Eval only (would run with implementation): {e}")
    
    try:
        example_reproducible_experiment()
    except Exception as e:
        print(f"Reproducible (would run with implementation): {e}")
    
    try:
        example_comparison()
    except Exception as e:
        print(f"Comparison (would run with implementation): {e}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The clean MVP provides:

1. **AxiomSpec**: Base template with configs and controls
2. **TrainAndEvalSpec**: Typed configs for training + eval
3. **TrainAndEvalExperiment**: Single experiment, single run()
4. **Clear mapping**: spec <-> experiment is 1:1

Simple, clean, and presentable!
    """)


if __name__ == "__main__":
    main()