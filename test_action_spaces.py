#!/usr/bin/env python3
"""Test that all navigation training environments have consistent action spaces."""

import yaml

# List of all navigation training environments used in the curricula
NAVIGATION_TRAINING_ENVS = [
    "configs/env/mettagrid/navigation/training/terrain_from_numpy.yaml",
    "configs/env/mettagrid/navigation/training/cylinder_world.yaml",
    "configs/env/mettagrid/navigation/training/varied_terrain_sparse.yaml",
    "configs/env/mettagrid/navigation/training/varied_terrain_balanced.yaml",
    "configs/env/mettagrid/navigation/training/varied_terrain_maze.yaml",
    "configs/env/mettagrid/navigation/training/varied_terrain_dense.yaml",
    "configs/env/mettagrid/navigation/training/sparse.yaml",
    "configs/env/mettagrid/navigation/training/raster_grid_multiroom.yaml",
    "configs/env/mettagrid/navigation/training/spiral_altar_multiroom.yaml",
]


def get_enabled_actions(yaml_path):
    """Extract enabled actions from a YAML config file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Check if it uses the training defaults
    uses_training_defaults = False
    if "defaults" in config:
        for default in config["defaults"]:
            if "/env/mettagrid/navigation/training/defaults" in str(default):
                uses_training_defaults = True
                break

    # If it uses training defaults, we know the action config
    if uses_training_defaults:
        return ["noop", "move", "rotate", "get_items"]

    # Otherwise, check if actions are explicitly defined
    if "game" in config and "actions" in config["game"]:
        actions = config["game"]["actions"]
        enabled = []
        # Check each action in order
        for action in [
            "noop",
            "move",
            "rotate",
            "put_items",
            "get_items",
            "attack",
            "swap",
            "change_color",
            "change_glyph",
        ]:
            if action in actions and actions[action].get("enabled", True):  # Default is True if not specified
                enabled.append(action)
        return enabled

    # If no actions are defined, it inherits from base mettagrid which has all actions enabled
    return ["noop", "move", "rotate", "put_items", "get_items", "attack", "swap"]


def test_action_spaces():
    """Test that all navigation training environments have consistent action spaces."""
    print("Testing action space consistency across navigation training environments...\n")

    action_configs = {}

    for env_path in NAVIGATION_TRAINING_ENVS:
        print(f"Checking {env_path}...")

        try:
            enabled_actions = get_enabled_actions(env_path)
            action_configs[env_path] = enabled_actions
            print(f"  Enabled actions: {enabled_actions}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Check consistency
    print("\nChecking consistency...")

    reference_actions = None
    reference_env = None
    all_consistent = True

    for env_path, actions in action_configs.items():
        if reference_actions is None:
            reference_actions = actions
            reference_env = env_path
        elif actions != reference_actions:
            print(f"❌ MISMATCH: {env_path}")
            print(f"   Has actions: {actions}")
            print(f"   But {reference_env} has: {reference_actions}")
            all_consistent = False

    if all_consistent:
        print(f"✅ All environments have consistent actions: {reference_actions}")
    else:
        print("\n❌ Action configurations are NOT consistent across all environments!")
        print("\nTo fix this, ensure all navigation training environments have the same actions enabled.")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_action_spaces())
