"""Tests for LLM prompt builder with context window management."""

from __future__ import annotations

import json

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig, MettaGridConfig
from mettagrid.policy.llm_prompt_builder import LLMPromptBuilder, VisibleElements
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation, ObservationToken, Simulation


@pytest.fixture
def mock_policy_env_info():
    """Create a mock PolicyEnvInterface for testing."""
    # Create realistic feature specs
    obs_features = [
        ObservationFeatureSpec(id=0, name="tag", normalization=1.0),
        ObservationFeatureSpec(id=1, name="inv:energy", normalization=1.0),
        ObservationFeatureSpec(id=2, name="cooldown_remaining", normalization=1.0),
        ObservationFeatureSpec(id=3, name="agent:group", normalization=1.0),
        ObservationFeatureSpec(id=4, name="last_action", normalization=1.0),
        ObservationFeatureSpec(id=5, name="last_reward", normalization=1.0),
    ]

    # Create tag names matching common MettaGrid objects
    tags = ["agent", "assembler", "carbon_extractor", "charger", "chest",
            "germanium_extractor", "oxygen_extractor", "silicon_extractor", "wall"]

    # Simple action names for testing
    action_names = ["noop", "move_north", "move_south", "move_east", "move_west", "use"]

    actions_cfg = ActionsConfig()
    obs_space = Box(low=0, high=255, shape=(11, 11, len(obs_features)), dtype=np.uint8)
    action_space = Discrete(len(action_names))

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


@pytest.fixture
def prompt_builder(mock_policy_env_info):
    """Create a prompt builder with default settings."""
    return LLMPromptBuilder(
        policy_env_info=mock_policy_env_info,
        context_window_size=20,
    )


@pytest.fixture
def sample_observation(mock_policy_env_info):
    """Create a sample observation with various tokens."""
    features = mock_policy_env_info.obs_features

    # Find features by name
    tag_feature = next(f for f in features if f.name == "tag")
    inv_energy_feature = next(f for f in features if f.name == "inv:energy")
    cooldown_feature = next(f for f in features if f.name == "cooldown_remaining")

    tokens = [
        # Agent's own state (at center 5,5)
        ObservationToken(
            feature=inv_energy_feature,
            location=(5, 5),
            value=100,
            raw_token=(inv_energy_feature.id, (5 << 16) | 5, 100),
        ),
        # A wall to the north
        ObservationToken(
            feature=tag_feature,
            location=(5, 4),
            value=8,  # tag 8 = wall
            raw_token=(tag_feature.id, (5 << 16) | 4, 8),
        ),
        # An extractor to the east
        ObservationToken(
            feature=tag_feature,
            location=(6, 5),
            value=2,  # tag 2 = carbon_extractor
            raw_token=(tag_feature.id, (6 << 16) | 5, 2),
        ),
        ObservationToken(
            feature=cooldown_feature,
            location=(6, 5),
            value=0,  # ready to use
            raw_token=(cooldown_feature.id, (6 << 16) | 5, 0),
        ),
    ]

    return AgentObservation(agent_id=0, tokens=tokens)


class TestLLMPromptBuilder:
    """Test suite for LLMPromptBuilder."""

    def test_initialization(self, mock_policy_env_info):
        """Test that prompt builder initializes correctly."""
        builder = LLMPromptBuilder(
            policy_env_info=mock_policy_env_info,
            context_window_size=20,
        )

        assert builder.step_count == 0
        assert builder.context_window_size == 20
        assert builder._last_visible is None

    def test_basic_info_prompt_structure(self, prompt_builder):
        """Test that basic_info_prompt contains essential game information."""
        basic_info = prompt_builder.basic_info_prompt()

        # Check for key sections in minimal prompt
        assert "GOAL" in basic_info
        assert "HOW TO PLAY" in basic_info
        assert "HEART RECIPE" in basic_info
        assert "VIBES" in basic_info

        # Check heart recipe is present
        assert "HEART" in basic_info
        assert "carbon" in basic_info

        # Check key gameplay instructions
        assert "ADJACENT" in basic_info
        assert "heart_a" in basic_info
        assert "heart_b" in basic_info

    def test_observable_prompt_extracts_visible_elements(self, prompt_builder, sample_observation):
        """Test that observable_prompt only describes visible elements."""
        observable = prompt_builder.observable_prompt(sample_observation)

        # Should include spatial map section
        assert "MAP" in observable
        assert "@" in observable  # Agent position marker

        # Should include directional awareness section
        assert "ADJACENT TILES" in observable

        # Should include inventory section
        assert "INVENTORY" in observable

    def test_observable_prompt_filters_to_visible_only(self, prompt_builder, sample_observation):
        """Test that observable_prompt doesn't include invisible elements."""
        observable = prompt_builder.observable_prompt(sample_observation)

        # Should mention visible objects with directions (wall, carbon_extractor)
        assert "wall" in observable.lower() or "carbon_extractor" in observable.lower()

        # Should have directional info
        assert "North" in observable or "South" in observable or "East" in observable or "West" in observable

        # Should show blocked status for adjacent walls
        assert "BLOCKED" in observable

    def test_full_prompt_combines_basic_and_observable(self, prompt_builder, sample_observation):
        """Test that full_prompt contains both basic_info and observable."""
        full = prompt_builder.full_prompt(sample_observation)

        # Full prompt should contain key parts of both
        assert "GOAL" in full  # From basic_info
        assert "MAP" in full  # From observable (spatial grid)
        assert "ADJACENT TILES" in full  # From observable

        # Should also include the response instructions
        assert "EXACTLY ONE action name" in full

    def test_context_prompt_first_step_includes_basic_info(self, prompt_builder, sample_observation):
        """Test that first step includes basic info."""
        prompt, includes_basic = prompt_builder.context_prompt(sample_observation)

        assert includes_basic is True
        assert "GOAL" in prompt
        assert "HEART RECIPE" in prompt
        assert prompt_builder.step_count == 1

    def test_context_prompt_second_step_excludes_basic_info(self, prompt_builder, sample_observation):
        """Test that second step only includes observable."""
        # First step
        prompt_builder.context_prompt(sample_observation)

        # Second step
        prompt, includes_basic = prompt_builder.context_prompt(sample_observation)

        assert includes_basic is False
        assert "GOAL" not in prompt  # Basic info not included
        assert "ADJACENT TILES" in prompt  # Observable is still there
        assert prompt_builder.step_count == 2

    def test_context_prompt_resets_at_window_boundary(self, prompt_builder, sample_observation):
        """Test that basic_info is resent at context window boundary."""
        # Steps 1-19: basic info only at step 1
        for i in range(19):
            _, includes_basic = prompt_builder.context_prompt(sample_observation)
            expected_basic = (i == 0)  # Only first step
            assert includes_basic == expected_basic

        # Step 20: should NOT reset yet (20 % 20 = 0, but we want reset at 21)
        _, includes_basic = prompt_builder.context_prompt(sample_observation)
        assert includes_basic is False
        assert prompt_builder.step_count == 20

        # Step 21: should reset (21 % 20 = 1, first of new window)
        _, includes_basic = prompt_builder.context_prompt(sample_observation)
        assert includes_basic is True
        assert prompt_builder.step_count == 21

    def test_context_prompt_force_basic_info(self, prompt_builder, sample_observation):
        """Test that force_basic_info parameter works."""
        # First step
        prompt_builder.context_prompt(sample_observation)

        # Second step with force_basic_info=True
        prompt, includes_basic = prompt_builder.context_prompt(
            sample_observation,
            force_basic_info=True,
        )

        assert includes_basic is True
        assert "GOAL" in prompt

    def test_reset_context(self, prompt_builder, sample_observation):
        """Test that reset_context resets the step counter."""
        # Take several steps
        for _ in range(5):
            prompt_builder.context_prompt(sample_observation)

        assert prompt_builder.step_count == 5

        # Reset
        prompt_builder.reset_context()

        assert prompt_builder.step_count == 0
        assert prompt_builder._last_visible is None

    def test_extract_visible_elements(self, prompt_builder, sample_observation):
        """Test that visible elements are correctly extracted."""
        visible = prompt_builder._extract_visible_elements(sample_observation)

        assert isinstance(visible, VisibleElements)
        assert 8 in visible.tags  # wall tag
        assert 2 in visible.tags  # carbon_extractor tag
        assert "tag" in visible.features
        assert "inv:energy" in visible.features
        assert "cooldown_remaining" in visible.features

    def test_visible_elements_equality(self):
        """Test VisibleElements equality comparison."""
        ve1 = VisibleElements(tags={1, 2, 3}, features={"a", "b", "c"})
        ve2 = VisibleElements(tags={1, 2, 3}, features={"a", "b", "c"})
        ve3 = VisibleElements(tags={1, 2}, features={"a", "b", "c"})

        assert ve1 == ve2
        assert ve1 != ve3
        assert ve1 != "not a VisibleElements"

    def test_observation_to_json_structure(self, prompt_builder, sample_observation):
        """Test that observation is correctly converted to JSON."""
        # Test without actions (default behavior for non-reset steps)
        obs_json = prompt_builder._observation_to_json(sample_observation, include_actions=False)
        assert obs_json["agent_id"] == 0
        assert "visible_objects" in obs_json
        assert "available_actions" not in obs_json  # Should not be included
        assert obs_json["num_visible_objects"] == len(sample_observation.tokens)

        # Test with actions (for first step or reset)
        obs_json_with_actions = prompt_builder._observation_to_json(sample_observation, include_actions=True)
        assert obs_json_with_actions["agent_id"] == 0
        assert "visible_objects" in obs_json_with_actions
        assert "available_actions" in obs_json_with_actions  # Should be included
        assert obs_json_with_actions["num_visible_objects"] == len(sample_observation.tokens)

        # Check token structure
        for token_dict in obs_json["visible_objects"]:
            assert "feature" in token_dict
            assert "location" in token_dict
            assert "value" in token_dict
            assert "x" in token_dict["location"]
            assert "y" in token_dict["location"]

    def test_custom_context_window_size(self, mock_policy_env_info, sample_observation):
        """Test that custom context window sizes work correctly."""
        builder = LLMPromptBuilder(
            policy_env_info=mock_policy_env_info,
            context_window_size=5,
        )

        # Step 1: includes basic
        _, includes_basic = builder.context_prompt(sample_observation)
        assert includes_basic is True

        # Steps 2-5: no basic
        for _ in range(4):
            _, includes_basic = builder.context_prompt(sample_observation)
            assert includes_basic is False

        # Step 6: reset (6 % 5 = 1)
        _, includes_basic = builder.context_prompt(sample_observation)
        assert includes_basic is True

    def test_tag_descriptions_comprehensive(self, prompt_builder):
        """Test that tag descriptions cover common object types."""
        # Test descriptions for common tags
        assert "agent" in prompt_builder._get_tag_description("agent").lower() or "cog" in prompt_builder._get_tag_description("agent").lower()
        assert "wall" in prompt_builder._get_tag_description("wall").lower()
        assert "carbon" in prompt_builder._get_tag_description("carbon_extractor").lower()
        assert "storage" in prompt_builder._get_tag_description("chest").lower() or "deposit" in prompt_builder._get_tag_description("chest").lower()

        # Unknown tag should have fallback
        unknown_desc = prompt_builder._get_tag_description("unknown_object_type")
        assert "unknown" in unknown_desc.lower() or "station" in unknown_desc.lower()

    def test_feature_descriptions_comprehensive(self, prompt_builder):
        """Test that feature descriptions cover common features."""
        # Test descriptions for common features
        assert "steps" in prompt_builder._get_feature_description("cooldown_remaining").lower()  # "Steps until..."
        assert "team" in prompt_builder._get_feature_description("agent:group").lower()
        assert "frozen" in prompt_builder._get_feature_description("agent:frozen").lower()
        assert "reward" in prompt_builder._get_feature_description("last_reward").lower()

    def test_prompt_builder_handles_empty_observation(self, prompt_builder, mock_policy_env_info):
        """Test that builder handles observations with no tokens."""
        empty_obs = AgentObservation(agent_id=0, tokens=[])

        # Should not crash
        observable = prompt_builder.observable_prompt(empty_obs)
        assert isinstance(observable, str)

        # Should still have directional awareness structure
        assert "ADJACENT TILES" in observable

    def test_json_serialization_of_observation(self, prompt_builder, sample_observation):
        """Test that observation JSON is valid and serializable."""
        obs_json = prompt_builder._observation_to_json(sample_observation)

        # Should be serializable to JSON string
        json_str = json.dumps(obs_json, indent=2)
        assert isinstance(json_str, str)

        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed["agent_id"] == obs_json["agent_id"]

    # --- Tests merged from test_llm_prompt_compatibility.py ---

    def test_dynamic_prompt_changes_based_on_observation(self, mock_policy_env_info):
        """Test that observable prompt changes when observation changes."""
        builder = LLMPromptBuilder(policy_env_info=mock_policy_env_info)

        tag_feature = mock_policy_env_info.obs_features[0]  # tag

        # Observation 1: Just a wall
        obs1 = AgentObservation(
            agent_id=0,
            tokens=[
                ObservationToken(
                    feature=tag_feature,
                    location=(5, 4),
                    value=8,  # wall (index 8)
                    raw_token=(tag_feature.id, (5 << 16) | 4, 8),
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
                    value=8,  # wall
                    raw_token=(tag_feature.id, (5 << 16) | 4, 8),
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
        assert "wall" in prompt1.lower() or "W" in prompt1
        assert "wall" in prompt2.lower() or "W" in prompt2

        # Prompts should be different (obs2 has assembler)
        assert prompt1 != prompt2

    def test_agent_sees_11x11_grid(self):
        """Verify that the agent observes an 11x11 grid."""
        cfg = MettaGridConfig()
        policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

        assert policy_env_info.obs_width == 11, f"Expected obs_width=11, got {policy_env_info.obs_width}"
        assert policy_env_info.obs_height == 11, f"Expected obs_height=11, got {policy_env_info.obs_height}"
        assert policy_env_info.obs_width * policy_env_info.obs_height == 121  # 11x11 = 121 cells

    def test_coordinates_are_egocentric(self):
        """Verify that coordinates are egocentric (agent always at center 5,5)."""
        cfg = MettaGridConfig()
        policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

        center_x = policy_env_info.obs_width // 2
        center_y = policy_env_info.obs_height // 2

        assert center_x == 5, f"Expected center_x=5, got {center_x}"
        assert center_y == 5, f"Expected center_y=5, got {center_y}"

    def test_egocentric_observation_bounds(self):
        """Verify that egocentric observations have correct bounds."""
        cfg = MettaGridConfig()
        policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

        assert policy_env_info.obs_width == 11
        assert policy_env_info.obs_height == 11

    def test_observable_features_are_grouped_correctly(self, mock_policy_env_info):
        """Test that features are grouped by type (inventory, protocol, agent, etc.)."""
        # Add protocol features to the mock
        obs_features = list(mock_policy_env_info.obs_features) + [
            ObservationFeatureSpec(id=6, name="inv:carbon", normalization=1.0),
            ObservationFeatureSpec(id=7, name="protocol_input:carbon", normalization=1.0),
            ObservationFeatureSpec(id=8, name="protocol_output:heart", normalization=1.0),
        ]

        extended_info = PolicyEnvInterface(
            obs_features=obs_features,
            tags=mock_policy_env_info.tags,
            actions=mock_policy_env_info.actions,
            num_agents=2,
            observation_space=mock_policy_env_info.observation_space,
            action_space=mock_policy_env_info.action_space,
            obs_width=11,
            obs_height=11,
            assembler_protocols=[],
            tag_id_to_name=mock_policy_env_info.tag_id_to_name,
        )

        builder = LLMPromptBuilder(policy_env_info=extended_info)

        inv_energy = next(f for f in obs_features if f.name == "inv:energy")
        inv_carbon = next(f for f in obs_features if f.name == "inv:carbon")

        obs = AgentObservation(
            agent_id=0,
            tokens=[
                ObservationToken(
                    feature=inv_energy,
                    location=(5, 5),
                    value=100,
                    raw_token=(inv_energy.id, (5 << 16) | 5, 100),
                ),
                ObservationToken(
                    feature=inv_carbon,
                    location=(5, 5),
                    value=50,
                    raw_token=(inv_carbon.id, (5 << 16) | 5, 50),
                ),
            ],
        )

        observable = builder.observable_prompt(obs)

        # Should have inventory section
        assert "INVENTORY" in observable or "inventory" in observable.lower()
        assert "energy" in observable.lower()
        assert "carbon" in observable.lower()

    def test_prompt_sequence_pattern(self, mock_policy_env_info, sample_observation):
        """Test that prompt sequence follows FULL/SHORT pattern correctly.

        This simulates what happens in llm_policy.py step() method:
        - Step 1: FULL (basic + obs)
        - Steps 2-N: SHORT (obs only)
        - Step N+1: FULL again (context window reset)
        """
        context_window_size = 5
        builder = LLMPromptBuilder(
            policy_env_info=mock_policy_env_info,
            context_window_size=context_window_size,
        )

        results = []

        # Simulate 12 steps (more than 2 full context windows)
        for _ in range(12):
            basic_info = builder.basic_info_prompt()
            observable = builder.observable_prompt(sample_observation)
            builder._step_counter += 1
            step = builder.step_count
            is_first_step = step == 1
            is_window_reset = step % builder.context_window_size == 1
            includes_basic_info = is_first_step or is_window_reset

            if includes_basic_info:
                results.append(f"[Step {step}] FULL: basic, obs")
            else:
                results.append(f"[Step {step}] SHORT: obs")

        # Verify the pattern
        # With context_window_size=5, FULL should be at steps 1, 6, 11
        assert "FULL" in results[0]  # Step 1
        assert "SHORT" in results[1]  # Step 2
        assert "SHORT" in results[2]  # Step 3
        assert "SHORT" in results[3]  # Step 4
        assert "SHORT" in results[4]  # Step 5
        assert "FULL" in results[5]  # Step 6 (6 % 5 == 1)
        assert "SHORT" in results[6]  # Step 7
        assert "SHORT" in results[7]  # Step 8
        assert "SHORT" in results[8]  # Step 9
        assert "SHORT" in results[9]  # Step 10
        assert "FULL" in results[10]  # Step 11 (11 % 5 == 1)
        assert "SHORT" in results[11]  # Step 12

        # Count totals
        full_count = sum(1 for r in results if "FULL" in r)
        short_count = sum(1 for r in results if "SHORT" in r)
        assert full_count == 3  # Steps 1, 6, 11
        assert short_count == 9  # All other steps
