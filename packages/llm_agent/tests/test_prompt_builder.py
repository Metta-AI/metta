"""Tests for LLM prompt builder with context window management."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig, MettaGridConfig
from llm_agent.policy.prompt_builder import LLMPromptBuilder, VisibleElements
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
        assert "VIBES" in basic_info

        # Check heart deposit goal is present
        assert "HEART" in basic_info

        # Check key gameplay instructions
        assert "heart_a" in basic_info
        assert "heart_b" in basic_info

    def test_observable_prompt_extracts_visible_elements(self, prompt_builder, sample_observation):
        """Test that observable_prompt only describes visible elements."""
        observable = prompt_builder.observable_prompt(sample_observation)

        # Should include visible objects with coordinates section
        assert "VISIBLE OBJECTS" in observable

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
        assert "VISIBLE OBJECTS" in full  # From observable (spatial coordinates)
        assert "ADJACENT TILES" in full  # From observable

        # Should also include the JSON response format instructions
        assert "reasoning" in full
        assert "action" in full

    def test_context_prompt_first_step_includes_basic_info(self, prompt_builder, sample_observation):
        """Test that first step includes basic info."""
        prompt, includes_basic = prompt_builder.context_prompt(sample_observation)

        assert includes_basic is True
        assert "GOAL" in prompt
        assert "HOW TO PLAY" in prompt
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

    def test_prompt_builder_handles_empty_observation(self, prompt_builder, mock_policy_env_info):
        """Test that builder handles observations with no tokens."""
        empty_obs = AgentObservation(agent_id=0, tokens=[])

        # Should not crash
        observable = prompt_builder.observable_prompt(empty_obs)
        assert isinstance(observable, str)

        # Should still have directional awareness structure
        assert "ADJACENT TILES" in observable

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

    def test_coordinate_system_matches_movement_directions(self, mock_policy_env_info):
        """Verify that coordinate system matches movement directions.

        In MettaGrid (SWAPPED from standard):
        - col() = X axis (horizontal, East/West)
        - row() = Y axis (vertical, North/South)

        Based on grid conventions:
        - North: row decreases (moving up, y decreases)
        - South: row increases (moving down, y increases)
        - East: col increases (moving right, x increases)
        - West: col decreases (moving left, x decreases)
        """
        builder = LLMPromptBuilder(policy_env_info=mock_policy_env_info)

        tag_feature = mock_policy_env_info.obs_features[0]  # tag

        # Agent is at center (5, 5)
        # In the code: x, y = token.row(), token.col()
        # So x=row, y=col
        agent_x, agent_y = 5, 5

        # Object to the NORTH: same col, row decreases
        # location format: (col, row) where token.col()=location[0], token.row()=location[1]
        # With x=row, y=col: object at row=4, col=5 appears at (4, 5)
        north_obs = AgentObservation(
            agent_id=0,
            tokens=[
                ObservationToken(
                    feature=tag_feature,
                    location=(agent_x, agent_y - 1),  # (col=5, row=4) -> North
                    value=1,  # assembler
                    raw_token=(tag_feature.id, 0, 1),
                ),
            ],
        )

        # Object to the EAST: col increases, same row
        # With x=row, y=col: object at row=5, col=6 appears at (5, 6)
        east_obs = AgentObservation(
            agent_id=0,
            tokens=[
                ObservationToken(
                    feature=tag_feature,
                    location=(agent_x + 1, agent_y),  # (col=6, row=5) -> East
                    value=2,  # carbon_extractor
                    raw_token=(tag_feature.id, 0, 2),
                ),
            ],
        )

        # Test North object - with x=row, y=col, object at row=4, col=5 shows as (4,5)
        north_prompt = builder.observable_prompt(north_obs)
        assert "West" in north_prompt, f"Object should show West direction (due to x=row convention), got: {north_prompt}"
        assert "(4,5)" in north_prompt, f"Object should be at (4,5), got: {north_prompt}"

        # Test East object - with x=row, y=col, object at row=5, col=6 shows as (5,6)
        east_prompt = builder.observable_prompt(east_obs)
        assert "South" in east_prompt, f"Object should show South direction (due to y=col convention), got: {east_prompt}"
        assert "(5,6)" in east_prompt, f"Object should be at (5,6), got: {east_prompt}"

    def test_context_window_ab_pattern(self, mock_policy_env_info, sample_observation):
        """Test that context window produces correct A/B pattern.

        With context_window_size=20:
        - Step 1: A (full prompt with basic info)
        - Steps 2-20: B (dynamic prompt, observable only)
        - Step 21: A (new window, basic info again)
        - Steps 22-40: B
        - etc.

        So in 20 steps we should see: 1 A, 19 Bs
        In 40 steps we should see: 2 As, 38 Bs
        """
        context_window_size = 20
        builder = LLMPromptBuilder(
            policy_env_info=mock_policy_env_info,
            context_window_size=context_window_size,
        )

        results = []

        steps = 20
        # Run for exactly 20 steps (one full context window)
        for _ in range(steps):
            prompt, includes_basic_info = builder.context_prompt(sample_observation)
            if includes_basic_info:
                results.append("A")
            else:
                results.append("B")

        # Count As and Bs
        a_count = results.count("A")
        b_count = results.count("B")

        # In 20 steps: 1 A (step 1), 19 Bs (steps 2-20)
        assert a_count == 1, f"Expected 1 A in 20 steps, got {a_count}. Pattern: {''.join(results)}"
        assert b_count == 19, f"Expected 19 Bs in 20 steps, got {b_count}. Pattern: {''.join(results)}"

        # Verify the pattern starts with A
        assert results[0] == "A", f"First step should be A, got {results[0]}"

        # All subsequent steps should be B
        assert all(r == "B" for r in results[1:]), f"Steps 2-20 should all be B. Pattern: {''.join(results)}"

        # Continue for another 20 steps (second context window)
        for _ in range(20):
            prompt, includes_basic_info = builder.context_prompt(sample_observation)
            if includes_basic_info:
                results.append("A")
            else:
                results.append("B")

        # In 40 steps: 2 As (steps 1 and 21), 38 Bs
        a_count = results.count("A")
        b_count = results.count("B")

        assert a_count == 2, f"Expected 2 As in 40 steps, got {a_count}. Pattern: {''.join(results)}"
        assert b_count == 38, f"Expected 38 Bs in 40 steps, got {b_count}. Pattern: {''.join(results)}"

        # Verify A appears at positions 0 and 20 (steps 1 and 21)
        assert results[0] == "A", f"Step 1 should be A"
        assert results[20] == "A", f"Step 21 should be A (new context window)"

