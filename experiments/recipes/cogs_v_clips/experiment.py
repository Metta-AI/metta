"""Example usage of the CoGs vs Clips training recipe.

This file demonstrates how to use the recipe programmatically.
For CLI usage, see README.md
"""

import subprocess
import time

from experiments.recipes.cogs_v_clips.methods import (
    evaluate,
    make_curriculum,
    make_eval_suite,
    make_training_env,
    play,
    train,
    train_coordination,
    train_single_mission,
    train_small_maps,
)

# Define experiment configurations for skypilot jobs
experiment_configs = {
    # Quick experiments (for testing/debugging)
    "debug_single": {
        "function": "train_single_mission",
        "mission_name": "extractor_hub_30",
        "num_cogs": 2,
        "gpus": 1,
        "timesteps": 5_000_000,
    },
    # Small maps experiments
    "small_1cog": {
        "function": "train_small_maps",
        "num_cogs": 1,
        "gpus": 2,
        "timesteps": 20_000_000,
    },
    "small_2cogs": {
        "function": "train_small_maps",
        "num_cogs": 2,
        "gpus": 2,
        "timesteps": 20_000_000,
    },
    "small_4cogs": {
        "function": "train_small_maps",
        "num_cogs": 4,
        "gpus": 4,
        "timesteps": 30_000_000,
    },
    # Medium maps experiments
    "medium_4cogs": {
        "function": "train_medium_maps",
        "num_cogs": 4,
        "gpus": 4,
        "timesteps": 40_000_000,
    },
    # Coordination-focused
    "coordination_4cogs": {
        "function": "train_coordination",
        "num_cogs": 4,
        "gpus": 4,
        "timesteps": 40_000_000,
    },
    # Full curriculum (all missions)
    "full_1cog": {
        "function": "train",
        "num_cogs": 1,
        "gpus": 4,
        "timesteps": 50_000_000,
    },
    "full_4cogs": {
        "function": "train",
        "num_cogs": 4,
        "gpus": 8,
        "timesteps": 100_000_000,
    },
    "full_8cogs": {
        "function": "train",
        "num_cogs": 8,
        "gpus": 8,
        "timesteps": 100_000_000,
    },
}


def basic_training():
    """Train on small maps with 4 agents."""
    tool = train_small_maps(num_cogs=4)
    return tool


def custom_curriculum():
    """Create a custom curriculum with specific missions."""
    curriculum = make_curriculum(
        num_cogs=4,
        base_missions=["extractor_hub_30", "oxygen_bottleneck", "energy_starved"],
        enable_detailed_slice_logging=True,
    )
    tool = train(num_cogs=4, curriculum=curriculum)
    return tool


def single_mission_debug():
    """Quick debug training on a single mission."""
    tool = train_single_mission(mission_name="extractor_hub_30", num_cogs=2)
    return tool


def evaluation():
    """Evaluate a trained policy."""
    tool = evaluate(
        policy_uris=["file://./checkpoints/cvc_default/latest"],
        num_cogs=4,
        difficulty="standard",
    )
    return tool


def multi_agent_coordination():
    """Train specifically on multi-agent coordination."""
    tool = train_coordination(num_cogs=4)
    return tool


def custom_eval_suite():
    """Create a custom evaluation suite."""
    # Test only on hub missions with 8 agents
    suite = make_eval_suite(
        num_cogs=8,
        difficulty="standard",
        subset=["extractor_hub_30", "extractor_hub_50", "extractor_hub_70"],
    )
    print(f"Created suite with {len(suite)} simulations:")
    for sim in suite:
        print(f"  - {sim.name}")
    return suite


def play_trained_policy():
    """Play a trained policy interactively."""
    tool = play(
        policy_uri="file://./checkpoints/cvc_default/latest",
        mission_name="extractor_hub_30",
        num_cogs=4,
    )
    return tool


def inspect_training_env():
    """Inspect the configuration of a training environment."""
    env = make_training_env(num_cogs=4, mission_name="extractor_hub_30")

    print("Environment Configuration:")
    print(f"  Num agents: {env.game.num_agents}")
    print(f"  Max steps: {env.game.max_steps}")

    # Check station efficiencies
    print("\nStation Efficiencies:")
    for obj_name in [
        "charger",
        "carbon_extractor",
        "oxygen_extractor",
        "silicon_extractor",
    ]:
        if obj_name in env.game.objects:
            obj = env.game.objects[obj_name]
            if hasattr(obj, "efficiency"):
                print(f"  {obj_name}: {obj.efficiency}%")

    # Check reward structure
    print("\nReward Structure:")
    print(f"  Inventory rewards: {env.game.agent.rewards.inventory}")

    return env


def experiment(
    configs: list[str] | None = None,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
):
    """Launch skypilot jobs for multiple training configurations.

    Args:
        configs: List of config names to run (defaults to all non-debug configs)
        heartbeat_timeout: Timeout in seconds for heartbeat monitoring
        skip_git_check: Whether to skip git status check before launching

    Example:
        # Run all standard experiments
        experiment()

        # Run specific experiments
        experiment(configs=["small_4cogs", "medium_4cogs"])

        # Run debug experiment
        experiment(configs=["debug_single"])
    """
    # Default to non-debug configs if not specified
    if configs is None:
        configs = [k for k in experiment_configs.keys() if not k.startswith("debug")]

    print(f"Launching {len(configs)} skypilot jobs:")
    for config_name in configs:
        print(f"  - {config_name}")

    for config_name in configs:
        if config_name not in experiment_configs:
            print(f"Warning: Unknown config '{config_name}', skipping")
            continue

        config = experiment_configs[config_name]
        function_name = config["function"]
        num_cogs = config["num_cogs"]
        gpus = config["gpus"]
        timesteps = config["timesteps"]

        # Build run name with timestamp
        run_name = f"cvc_{config_name}.{time.strftime('%Y-%m-%d_%H%M')}"

        # Build command args
        cmd_args = [
            "./devops/skypilot/launch.py",
            f"experiments.recipes.cogs_v_clips.{function_name}",
            f"run={run_name}",
            f"num_cogs={num_cogs}",
            f"trainer.total_timesteps={timesteps}",
            f"--gpus={gpus}",
            f"--heartbeat-timeout={heartbeat_timeout}",
        ]

        # Add mission_name if it's a single mission experiment
        if "mission_name" in config:
            cmd_args.insert(3, f"mission_name={config['mission_name']}")

        if skip_git_check:
            cmd_args.append("--skip-git-check")

        print(f"\nLaunching: {config_name}")
        print(f"  Run: {run_name}")
        print(f"  Function: {function_name}")
        print(f"  Agents: {num_cogs}, GPUs: {gpus}, Steps: {timesteps:,}")

        subprocess.run(cmd_args)
        time.sleep(1)  # Brief delay between launches

    print(f"\n✓ Successfully launched {len(configs)} jobs")


if __name__ == "__main__":
    import sys

    # Allow passing config names as command line arguments
    # e.g., python experiment.py debug_single small_4cogs
    if len(sys.argv) > 1:
        configs = sys.argv[1:]
        print(f"Running experiments: {', '.join(configs)}")
        experiment(configs=configs)
    else:
        # Default: run all non-debug experiments
        print("Running all standard experiments (not debug)")
        experiment()
