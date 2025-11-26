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

        # Check for key sections
        assert "COORDINATE SYSTEM" in basic_info
        assert "OBSERVATION FORMAT" in basic_info
        assert "CORE GAME MECHANICS" in basic_info
        assert "AVAILABLE ACTIONS" in basic_info

        # Check coordinate system details
        assert "11x11 grid" in basic_info
        assert "x=5, y=5" in basic_info  # Center position

        # Check that it explains walkability
        assert "WALKABLE" in basic_info or "walkable" in basic_info

    def test_observable_prompt_extracts_visible_elements(self, prompt_builder, sample_observation):
        """Test that observable_prompt only describes visible elements."""
        observable = prompt_builder.observable_prompt(sample_observation)

        # Should include visible tags section
        assert "OBJECTS YOU CAN SEE" in observable

        # Should include visible features section
        assert "FEATURES YOU CAN SEE" in observable

        # Should include current observation JSON
        assert "CURRENT OBSERVATION" in observable
        assert "visible_objects" in observable

    def test_observable_prompt_filters_to_visible_only(self, prompt_builder, sample_observation):
        """Test that observable_prompt doesn't include invisible elements."""
        observable = prompt_builder.observable_prompt(sample_observation)

        # Should mention visible tags (wall=8, carbon_extractor=2)
        assert "wall" in observable.lower() or "tag 8" in observable.lower()

        # Should NOT mention tags that aren't visible (e.g., chest, assembler)
        # This is harder to test definitively, but we can check structure
        visible_section = observable.split("OBJECTS YOU CAN SEE")[1].split("===")[0]
        # Count how many tag descriptions are present
        tag_count = visible_section.count("Tag ")
        # Should be small (only visible tags)
        assert tag_count <= 5  # Sample has 2 tags, but this is a loose check

    def test_full_prompt_combines_basic_and_observable(self, prompt_builder, sample_observation):
        """Test that full_prompt contains both basic_info and observable."""
        full = prompt_builder.full_prompt(sample_observation)
        basic = prompt_builder.basic_info_prompt()
        observable = prompt_builder.observable_prompt(sample_observation)

        # Full prompt should contain key parts of both
        assert "COORDINATE SYSTEM" in full  # From basic_info
        assert "OBJECTS YOU CAN SEE" in full  # From observable
        assert "CURRENT OBSERVATION" in full  # From observable

        # Should also include the response instructions
        assert "EXACTLY ONE action name" in full

    def test_context_prompt_first_step_includes_basic_info(self, prompt_builder, sample_observation):
        """Test that first step includes basic info."""
        prompt, includes_basic = prompt_builder.context_prompt(sample_observation)

        assert includes_basic is True
        assert "COORDINATE SYSTEM" in prompt
        assert "AVAILABLE ACTIONS" in prompt
        assert prompt_builder.step_count == 1

    def test_context_prompt_second_step_excludes_basic_info(self, prompt_builder, sample_observation):
        """Test that second step only includes observable."""
        # First step
        prompt_builder.context_prompt(sample_observation)

        # Second step
        prompt, includes_basic = prompt_builder.context_prompt(sample_observation)

        assert includes_basic is False
        assert "COORDINATE SYSTEM" not in prompt
        assert "AVAILABLE ACTIONS" not in prompt
        assert "OBJECTS YOU CAN SEE" in prompt  # Observable is still there
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
        assert "COORDINATE SYSTEM" in prompt

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
        obs_json = prompt_builder._observation_to_json(sample_observation)

        assert obs_json["agent_id"] == 0
        assert "visible_objects" in obs_json
        assert "available_actions" in obs_json
        assert obs_json["num_visible_objects"] == len(sample_observation.tokens)

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
        assert "agent" in prompt_builder._get_tag_description("agent").lower()
        assert "wall" in prompt_builder._get_tag_description("wall").lower()
        assert "carbon" in prompt_builder._get_tag_description("carbon_extractor").lower()
        assert "stores" in prompt_builder._get_tag_description("chest").lower()  # "Stores resources"

        # Unknown tag should have fallback
        unknown_desc = prompt_builder._get_tag_description("unknown_object_type")
        assert "unknown" in unknown_desc.lower()

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

        # Should still have structure
        assert "CURRENT OBSERVATION" in observable

    def test_json_serialization_of_observation(self, prompt_builder, sample_observation):
        """Test that observation JSON is valid and serializable."""
        obs_json = prompt_builder._observation_to_json(sample_observation)

        # Should be serializable to JSON string
        json_str = json.dumps(obs_json, indent=2)
        assert isinstance(json_str, str)

        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed["agent_id"] == obs_json["agent_id"]
