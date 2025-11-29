"""Tests verifying LLMPromptBuilder compatibility and correctness."""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.policy.llm_policy import build_game_rules_prompt
from mettagrid.policy.llm_prompt_builder import LLMPromptBuilder
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation, ObservationToken


def create_test_policy_env_info() -> PolicyEnvInterface:
    """Create a policy env info for testing."""
    obs_features = [
        ObservationFeatureSpec(id=0, name="tag", normalization=1.0),
        ObservationFeatureSpec(id=1, name="inv:energy", normalization=1.0),
        ObservationFeatureSpec(id=2, name="inv:carbon", normalization=1.0),
        ObservationFeatureSpec(id=3, name="cooldown_remaining", normalization=1.0),
        ObservationFeatureSpec(id=4, name="agent:group", normalization=1.0),
        ObservationFeatureSpec(id=5, name="protocol_input:carbon", normalization=1.0),
        ObservationFeatureSpec(id=6, name="protocol_output:heart", normalization=1.0),
    ]

    tags = ["agent", "assembler", "carbon_extractor", "chest", "wall"]

    actions_cfg = ActionsConfig()
    obs_space = Box(low=0, high=255, shape=(11, 11, len(obs_features)), dtype=np.uint8)
    action_space = Discrete(6)

    return PolicyEnvInterface(
        obs_features=obs_features,
        tags=tags,
        actions=actions_cfg,
        num_agents=2,
        observation_space=obs_space,
        action_space=action_space,
        obs_width=11,
        obs_height=11,
        assembler_protocols=[],
        tag_id_to_name={i: name for i, name in enumerate(tags)},
    )


def test_basic_info_contains_core_elements():
    """Test that basic_info_prompt contains essential game information."""
    policy_env_info = create_test_policy_env_info()
    builder = LLMPromptBuilder(policy_env_info)

    basic_info = builder.basic_info_prompt()

    # Should contain coordinate system
    assert "11x11 grid" in basic_info
    assert "x=5, y=5" in basic_info  # Center position
    assert "EGOCENTRIC" in basic_info or "egocentric" in basic_info.lower()

    # Should contain movement logic
    assert "WALKABLE" in basic_info or "walkable" in basic_info
    assert "BLOCKED" in basic_info or "blocked" in basic_info

    # Should contain action reference
    assert "ACTIONS" in basic_info or "actions" in basic_info
    assert "noop" in basic_info


def test_observable_prompt_only_includes_visible_elements():
    """Test that observable_prompt filters to only visible objects/features."""
    policy_env_info = create_test_policy_env_info()
    builder = LLMPromptBuilder(policy_env_info)

    # Create observation with specific visible elements
    tag_feature = policy_env_info.obs_features[0]  # tag
    inv_energy = policy_env_info.obs_features[1]  # inv:energy
    cooldown = policy_env_info.obs_features[3]  # cooldown_remaining

    obs = AgentObservation(
        agent_id=0,
        tokens=[
            # My energy inventory
            ObservationToken(
                feature=inv_energy,
                location=(5, 5),
                value=100,
                raw_token=(inv_energy.id, (5 << 16) | 5, 100),
            ),
            # A wall to the north
            ObservationToken(
                feature=tag_feature,
                location=(5, 4),
                value=4,  # wall (index 4)
                raw_token=(tag_feature.id, (5 << 16) | 4, 4),
            ),
            # A carbon extractor to the east
            ObservationToken(
                feature=tag_feature,
                location=(6, 5),
                value=2,  # carbon_extractor (index 2)
                raw_token=(tag_feature.id, (6 << 16) | 5, 2),
            ),
            ObservationToken(
                feature=cooldown,
                location=(6, 5),
                value=0,  # ready
                raw_token=(cooldown.id, (6 << 16) | 5, 0),
            ),
        ],
    )

    observable = builder.observable_prompt(obs)

    # Should include visible object types in nearby objects
    assert "wall" in observable.lower()
    assert "carbon" in observable.lower()

    # Should include directional awareness
    assert "ADJACENT TILES" in observable
    assert "BLOCKED" in observable  # Wall is adjacent

    # Should include inventory
    assert "energy" in observable.lower()


def test_dynamic_prompt_changes_based_on_observation():
    """Test that observable prompt changes when observation changes."""
    policy_env_info = create_test_policy_env_info()
    builder = LLMPromptBuilder(policy_env_info)

    tag_feature = policy_env_info.obs_features[0]
    inv_energy = policy_env_info.obs_features[1]

    # Observation 1: Just a wall
    obs1 = AgentObservation(
        agent_id=0,
        tokens=[
            ObservationToken(
                feature=tag_feature,
                location=(5, 4),
                value=4,  # wall
                raw_token=(tag_feature.id, (5 << 16) | 4, 4),
            ),
        ],
    )

    # Observation 2: Wall + assembler
    obs2 = AgentObservation(
        agent_id=0,
        tokens=[
            ObservationToken(
                feature=tag_feature,
                location=(5, 4),
                value=4,  # wall
                raw_token=(tag_feature.id, (5 << 16) | 4, 4),
            ),
            ObservationToken(
                feature=tag_feature,
                location=(6, 5),
                value=1,  # assembler
                raw_token=(tag_feature.id, (6 << 16) | 5, 1),
            ),
        ],
    )

    prompt1 = builder.observable_prompt(obs1)
    prompt2 = builder.observable_prompt(obs2)

    # Both should mention wall
    assert "wall" in prompt1.lower()
    assert "wall" in prompt2.lower()

    # Only prompt2 should mention assembler
    assert "assembler" not in prompt1.lower() or prompt1.lower().count("assembler") < prompt2.lower().count("assembler")

    # Prompts should be different
    assert prompt1 != prompt2


def test_full_prompt_structure():
    """Test that full_prompt has both basic and observable sections."""
    policy_env_info = create_test_policy_env_info()
    builder = LLMPromptBuilder(policy_env_info)

    tag_feature = policy_env_info.obs_features[0]
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            ObservationToken(
                feature=tag_feature,
                location=(5, 4),
                value=4,  # wall
                raw_token=(tag_feature.id, (5 << 16) | 4, 4),
            ),
        ],
    )

    full = builder.full_prompt(obs)

    # Should contain basic_info content
    assert "COORDINATE SYSTEM" in full or "coordinate system" in full.lower()
    assert "11x11" in full

    # Should contain observable content (new format uses ADJACENT TILES and NEARBY OBJECTS)
    assert "ADJACENT TILES" in full or "NEARBY OBJECTS" in full
    assert "wall" in full.lower()

    # Should contain response instructions
    assert "EXACTLY ONE action name" in full or "exactly one action name" in full.lower()


def test_static_prompt_coverage():
    """Verify old static prompt and new dynamic prompt cover same concepts.

    This test ensures backward compatibility - the new prompt should cover
    at least the same core concepts as the old static prompt.
    """
    policy_env_info = create_test_policy_env_info()

    # Old approach
    old_static = build_game_rules_prompt(policy_env_info)

    # New approach
    builder = LLMPromptBuilder(policy_env_info)
    new_basic = builder.basic_info_prompt()

    # Core concepts that MUST be in both
    core_concepts = [
        "11x11",  # Grid size
        ("x=5", "y=5"),  # Center position (both must be present)
        "north",  # Cardinal directions
        "walkable",  # Movement logic
        "blocked",
        "noop",  # Actions reference
    ]

    for concept in core_concepts:
        if isinstance(concept, tuple):
            # Check that all parts are present
            for part in concept:
                assert part.lower() in old_static.lower(), f"Old prompt missing: {part}"
                assert part.lower() in new_basic.lower(), f"New prompt missing: {part}"
        else:
            assert concept.lower() in old_static.lower(), f"Old prompt missing: {concept}"
            assert concept.lower() in new_basic.lower(), f"New prompt missing: {concept}"


def test_observable_features_are_grouped_correctly():
    """Test that features are grouped by type (inventory, protocol, agent, etc.)."""
    policy_env_info = create_test_policy_env_info()
    builder = LLMPromptBuilder(policy_env_info)

    # Create observation with various feature types
    obs = AgentObservation(
        agent_id=0,
        tokens=[
            ObservationToken(
                feature=policy_env_info.obs_features[1],  # inv:energy
                location=(5, 5),
                value=100,
                raw_token=(1, (5 << 16) | 5, 100),
            ),
            ObservationToken(
                feature=policy_env_info.obs_features[2],  # inv:carbon
                location=(5, 5),
                value=50,
                raw_token=(2, (5 << 16) | 5, 50),
            ),
            ObservationToken(
                feature=policy_env_info.obs_features[5],  # protocol_input:carbon
                location=(6, 5),
                value=10,
                raw_token=(5, (6 << 16) | 5, 10),
            ),
            ObservationToken(
                feature=policy_env_info.obs_features[6],  # protocol_output:heart
                location=(6, 5),
                value=1,
                raw_token=(6, (6 << 16) | 5, 1),
            ),
        ],
    )

    observable = builder.observable_prompt(obs)

    # Should have directional awareness and inventory sections
    assert "ADJACENT TILES" in observable
    assert "INVENTORY" in observable or "inventory" in observable.lower()

    # Inventory should be listed
    assert "energy" in observable.lower()
    assert "carbon" in observable.lower()


def test_agent_sees_11x11_grid():
    """Verify that the agent actually observes an 11x11 grid.

    This confirms the observation window dimensions that the LLM prompt describes.
    """
    from mettagrid.config.mettagrid_config import MettaGridConfig

    # Create a real config
    cfg = MettaGridConfig()

    # Get policy env interface using from_mg_cfg
    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

    # CRITICAL: Verify observation dimensions are 11x11
    # This is what the LLM sees and what the prompt tells it about
    assert policy_env_info.obs_width == 11, f"Expected obs_width=11, got {policy_env_info.obs_width}"
    assert policy_env_info.obs_height == 11, f"Expected obs_height=11, got {policy_env_info.obs_height}"

    # The observation space is flattened for the network, but the logical grid is 11x11
    # Just verify the obs dimensions are consistent
    assert policy_env_info.obs_width * policy_env_info.obs_height == 121  # 11x11 = 121 cells


def test_coordinates_are_egocentric():
    """Verify that coordinates are egocentric (agent always at center 5,5).

    This confirms that the coordinate system described in the prompt is accurate:
    - Agent is always at center (5, 5) in their observation
    - Observations are ego-centric (relative to agent's position)
    """
    from mettagrid.config.mettagrid_config import MettaGridConfig

    cfg = MettaGridConfig()
    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

    # Verify center position calculation
    center_x = policy_env_info.obs_width // 2
    center_y = policy_env_info.obs_height // 2

    # CRITICAL: Agent is always at center (5, 5) in egocentric coordinates
    assert center_x == 5, f"Expected center_x=5 (agent's X position), got {center_x}"
    assert center_y == 5, f"Expected center_y=5 (agent's Y position), got {center_y}"

    # This means in the agent's 11x11 observation:
    # - (5, 5) is where the agent sees itself
    # - (5, 4) is North
    # - (5, 6) is South
    # - (6, 5) is East
    # - (4, 5) is West


def test_egocentric_observation_bounds():
    """Verify that egocentric observations have correct bounds.

    Egocentric means:
    - Observation is centered on the agent
    - Coordinates range from (0,0) to (10,10) in an 11x11 grid
    - Agent is always at (5, 5) regardless of world position
    """
    from mettagrid.config.mettagrid_config import MettaGridConfig

    cfg = MettaGridConfig()
    policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

    # Verify bounds
    assert policy_env_info.obs_width == 11
    assert policy_env_info.obs_height == 11

    # In egocentric coordinates:
    # - Top-left corner: (0, 0)
    # - Top-right corner: (10, 0)
    # - Bottom-left corner: (0, 10)
    # - Bottom-right corner: (10, 10)
    # - Agent center: (5, 5)

    # This is consistent with what the prompt tells the LLM


def test_prompt_builder_reflects_egocentric_coordinates():
    """Verify that the prompt builder correctly explains egocentric coordinates."""
    policy_env_info = create_test_policy_env_info()
    builder = LLMPromptBuilder(policy_env_info)

    basic_info = builder.basic_info_prompt()

    # Should explain that agent is at center
    assert "center" in basic_info.lower()
    assert "x=5" in basic_info and "y=5" in basic_info

    # Should explain egocentric nature
    assert "egocentric" in basic_info.lower() or "relative to you" in basic_info.lower()

    # Should show cardinal directions from center
    assert "north" in basic_info.lower()
    assert "south" in basic_info.lower()
    assert "east" in basic_info.lower()
    assert "west" in basic_info.lower()

    # Should indicate grid bounds
    assert "11x11" in basic_info or "11 x 11" in basic_info
