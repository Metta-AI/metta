"""Example usage of the CoGs vs Clips training recipe.

This file demonstrates how to use the recipe programmatically.
For CLI usage, see README.md
"""

from experiments.recipes.cogs_v_clips import (
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


if __name__ == "__main__":
    # Example: Inspect a training environment
    print("=" * 60)
    print("Example: Inspecting Training Environment")
    print("=" * 60)
    inspect_training_env()

    print("\n" + "=" * 60)
    print("Example: Creating Custom Eval Suite")
    print("=" * 60)
    custom_eval_suite()

    print("\n" + "=" * 60)
    print("To run training, use the ./tools/run.py command:")
    print(
        "  uv run ./tools/run.py experiments.recipes.cogs_v_clips.train_small_maps run=test"
    )
    print("=" * 60)
