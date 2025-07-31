from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv


def test_visitation_counts_configurable():
    """Test that visitation counts can be enabled/disabled via configuration."""

    # Test configuration with visitation counts enabled (default)
    config_with_visitation = {
        "num_agents": 1,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 100,
        "max_steps": 100,
        "track_movement_metrics": True,
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
            "visitation_counts": True,  # Enabled
        },
        "inventory_item_names": ["ore_red", "ore_blue", "ore_green"],
        "recipe_details_obs": False,
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {}, "defense_resources": {}},
            "swap": {"enabled": True},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False},
        },
        "agent": {
            "default_resource_limit": 10,
            "resource_limits": {},
            "freeze_duration": 0,
            "rewards": {"inventory": {}, "stats": {}},
            "action_failure_penalty": 0,
        },
        "groups": {
            "solo": {
                "id": 0,
                "sprite": 0,
                "group_reward_pct": 1.0,
                "props": {
                    "default_resource_limit": 10,
                    "resource_limits": {},
                    "freeze_duration": 0,
                    "rewards": {"inventory": {}, "stats": {}},
                    "action_failure_penalty": 0,
                },
            }
        },
        "objects": {"wall": {"type_id": 1, "swappable": False}},
    }

    # Test configuration with visitation counts disabled
    config_without_visitation = config_with_visitation.copy()
    config_without_visitation["global_obs"]["visitation_counts"] = False

    # Test with visitation counts enabled
    curriculum_with = SingleTaskCurriculum("test_with_visitation", config_with_visitation)
    env_with = MettaGridEnv(curriculum_with, render_mode=None)

    obs, _ = env_with.reset()

    # Count visitation count features (feature ID 14)
    visitation_features_with = []
    for i in range(obs.shape[1]):
        if obs[0, i, 1] == 14:  # feature_id == 14 (VisitationCounts)
            visitation_features_with.append(obs[0, i, 2])  # value

    print(f"With visitation counts enabled: found {len(visitation_features_with)} features")
    assert len(visitation_features_with) == 5, (
        f"Expected 5 visitation count features when enabled, got {len(visitation_features_with)}"
    )

    env_with.close()

    # Test with visitation counts disabled
    curriculum_without = SingleTaskCurriculum("test_without_visitation", config_without_visitation)
    env_without = MettaGridEnv(curriculum_without, render_mode=None)

    obs, _ = env_without.reset()

    # Count visitation count features (feature ID 14)
    visitation_features_without = []
    for i in range(obs.shape[1]):
        if obs[0, i, 1] == 14:  # feature_id == 14 (VisitationCounts)
            visitation_features_without.append(obs[0, i, 2])  # value

    print(f"With visitation counts disabled: found {len(visitation_features_without)} features")
    assert len(visitation_features_without) == 0, (
        f"Expected 0 visitation count features when disabled, got {len(visitation_features_without)}"
    )

    env_without.close()


def test_visitation_counts_default_behavior():
    """Test that visitation counts are enabled by default when not specified."""

    # Test configuration without specifying visitation_counts (should default to True)
    config_default = {
        "num_agents": 1,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 100,
        "max_steps": 100,
        "track_movement_metrics": True,
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
            # visitation_counts not specified - should default to True
        },
        "inventory_item_names": ["ore_red", "ore_blue", "ore_green"],
        "recipe_details_obs": False,
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {}, "defense_resources": {}},
            "swap": {"enabled": True},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False},
        },
        "agent": {
            "default_resource_limit": 10,
            "resource_limits": {},
            "freeze_duration": 0,
            "rewards": {"inventory": {}, "stats": {}},
            "action_failure_penalty": 0,
        },
        "groups": {
            "solo": {
                "id": 0,
                "sprite": 0,
                "group_reward_pct": 1.0,
                "props": {
                    "default_resource_limit": 10,
                    "resource_limits": {},
                    "freeze_duration": 0,
                    "rewards": {"inventory": {}, "stats": {}},
                    "action_failure_penalty": 0,
                },
            }
        },
        "objects": {"wall": {"type_id": 1, "swappable": False}},
    }

    curriculum_default = SingleTaskCurriculum("test_default_visitation", config_default)
    env_default = MettaGridEnv(curriculum_default, render_mode=None)

    obs, _ = env_default.reset()

    # Count visitation count features (feature ID 14)
    visitation_features_default = []
    for i in range(obs.shape[1]):
        if obs[0, i, 1] == 14:  # feature_id == 14 (VisitationCounts)
            visitation_features_default.append(obs[0, i, 2])  # value

    print(f"With default visitation counts: found {len(visitation_features_default)} features")
    assert len(visitation_features_default) == 5, (
        f"Expected 5 visitation count features by default, got {len(visitation_features_default)}"
    )

    env_default.close()


if __name__ == "__main__":
    test_visitation_counts_configurable()
    test_visitation_counts_default_behavior()
    print("All tests passed!")
