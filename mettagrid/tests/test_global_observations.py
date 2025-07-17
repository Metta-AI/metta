from metta.mettagrid.mettagrid_c import AgentConfig, GameConfig, GlobalObsConfig, MettaGrid


class TestGlobalObservations:
    """Test global observation features in MettaGrid."""

    def test_game_rewards_observation(self):
        """Test that game_rewards global observation is correctly packed and included."""
        # Define inventory items with specific resources
        inventory_items = ["ore", "battery", "laser", "armor", "other_item"]

        # Create agent configs with different reward values
        agent_config = AgentConfig(
            1,  # type_id
            "agent",  # type_name
            0,  # group_id
            "test_group",  # group_name
            0,  # freeze_duration
            0.0,  # action_failure_penalty
            {i: 100 for i in range(len(inventory_items))},  # resource_limits
            {
                0: 0.25,  # ore: low reward (should quantize to 1)
                1: 0.75,  # battery: medium reward (should quantize to 2)
                2: 1.5,  # laser: high reward (should quantize to 3)
                3: 0.0,  # armor: no reward (should quantize to 0)
                4: 2.0,  # other_item: not included in packed rewards
            },  # resource_rewards
            {i: 100 for i in range(len(inventory_items))},  # resource_reward_max
            0.0,  # group_reward_pct
        )

        # Create global obs config with game_rewards enabled
        global_obs_config = GlobalObsConfig(
            episode_completion_pct=True,
            last_action=True,
            last_reward=True,
            game_rewards=True,
        )

        # Create game config with game_rewards enabled in global_obs
        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            global_obs=global_obs_config,
            actions={},
            objects={"agent.test_group": agent_config},
        )

        # Create simple map
        game_map = [["agent.test_group"]]

        # Create environment
        env = MettaGrid(game_config, game_map, 42)

        # Reset environment
        observations, _ = env.reset()

        # Check observations shape (1 agent, 50 tokens, 3 values per token)
        assert observations.shape == (1, 50, 3), f"Unexpected shape: {observations.shape}"

        # Extract observation tokens for the first agent
        agent_obs = observations[0]

        # Find game rewards token
        # Feature ID 13 is GameRewards, should be at center (2,2) in 5x5 obs
        game_rewards_token = None
        for token in agent_obs:
            if token[0] != 0xFF and token[1] == 13:  # Feature ID 13 is GameRewards
                game_rewards_token = token
                break

        assert game_rewards_token is not None, "Game rewards token not found in observation"

        # Expected packed value:
        # ore (0.25) -> 1 (bits 7-6: 01)
        # battery (0.75) -> 2 (bits 5-4: 10)
        # laser (1.5) -> 3 (bits 3-2: 11)
        # armor (0.0) -> 0 (bits 1-0: 00)
        # Binary: 01 10 11 00 = 0x6C = 108
        expected_packed = (1 << 6) | (2 << 4) | (3 << 2) | 0
        assert expected_packed == 108, f"Expected packed calculation wrong: {expected_packed}"

        assert game_rewards_token[2] == expected_packed, (
            f"Incorrect packed game rewards value. Expected: {expected_packed}, Got: {game_rewards_token[2]}"
        )

    def test_game_rewards_disabled(self):
        """Test that game_rewards observation is not included when disabled."""
        inventory_items = ["ore", "battery", "laser", "armor"]

        agent_config = AgentConfig(
            1,  # type_id
            "agent",  # type_name
            0,  # group_id
            "test_group",  # group_name
            0,  # freeze_duration
            0.0,  # action_failure_penalty
            {i: 100 for i in range(len(inventory_items))},  # resource_limits
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},  # resource_rewards
            {i: 100 for i in range(len(inventory_items))},  # resource_reward_max
            0.0,  # group_reward_pct
        )

        # Create global obs config with game_rewards disabled
        global_obs_config = GlobalObsConfig(
            episode_completion_pct=True,
            last_action=True,
            last_reward=True,
            game_rewards=False,
        )

        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            global_obs=global_obs_config,
            actions={},
            objects={"agent.test_group": agent_config},
        )

        game_map = [["agent.test_group"]]
        env = MettaGrid(game_config, game_map, 42)

        observations, _ = env.reset()
        agent_obs = observations[0]

        # Verify no game rewards token present
        for token in agent_obs:
            if token[0] != 0xFF:  # Not empty
                assert token[1] != 13, "Game rewards token found when disabled"

    def test_game_rewards_with_missing_resources(self):
        """Test game rewards with missing resource types."""
        # Only include ore and laser, missing battery and armor
        inventory_items = ["ore_red", "laser", "other_item"]

        agent_config = AgentConfig(
            1,  # type_id
            "agent",  # type_name
            0,  # group_id
            "test_group",  # group_name
            0,  # freeze_duration
            0.0,  # action_failure_penalty
            {i: 100 for i in range(len(inventory_items))},  # resource_limits
            {
                0: 2.0,  # ore_red: high reward (should quantize to 3)
                1: 0.3,  # laser: low reward (should quantize to 1)
            },  # resource_rewards
            {i: 100 for i in range(len(inventory_items))},  # resource_reward_max
            0.0,  # group_reward_pct
        )

        # Create global obs config with game_rewards enabled
        global_obs_config = GlobalObsConfig(
            episode_completion_pct=True,
            last_action=True,
            last_reward=True,
            game_rewards=True,
        )

        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            global_obs=global_obs_config,
            actions={},
            objects={"agent.test_group": agent_config},
        )

        game_map = [["agent.test_group"]]
        env = MettaGrid(game_config, game_map, 42)

        observations, _ = env.reset()
        agent_obs = observations[0]

        # Find game rewards token
        game_rewards_token = None
        for token in agent_obs:
            if token[0] != 0xFF and token[1] == 13:
                game_rewards_token = token
                break

        assert game_rewards_token is not None

        # Expected packed value:
        # ore (2.0) -> 3 (bits 7-6: 11)
        # battery (missing) -> 0 (bits 5-4: 00)
        # laser (0.3) -> 1 (bits 3-2: 01)
        # armor (missing) -> 0 (bits 1-0: 00)
        # Binary: 11 00 01 00 = 0xC4 = 196
        expected_packed = (3 << 6) | (0 << 4) | (1 << 2) | 0
        assert game_rewards_token[2] == expected_packed

    def test_game_rewards_quantization(self):
        """Test the quantization of reward values."""
        inventory_items = ["ore", "battery", "laser", "armor"]

        # Test edge cases of quantization
        agent_config = AgentConfig(
            1,  # type_id
            "agent",  # type_name
            0,  # group_id
            "test_group",  # group_name
            0,  # freeze_duration
            0.0,  # action_failure_penalty
            {i: 100 for i in range(len(inventory_items))},  # resource_limits
            {
                0: 0.0,  # ore: exactly 0 -> 0
                1: 0.5,  # battery: exactly 0.5 -> 1
                2: 1.0,  # laser: exactly 1.0 -> 2
                3: 10.0,  # armor: very high -> 3
            },  # resource_rewards
            {i: 100 for i in range(len(inventory_items))},  # resource_reward_max
            0.0,  # group_reward_pct
        )

        # Create global obs config with game_rewards enabled
        global_obs_config = GlobalObsConfig(
            episode_completion_pct=True,
            last_action=True,
            last_reward=True,
            game_rewards=True,
        )

        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            global_obs=global_obs_config,
            actions={},
            objects={"agent.test_group": agent_config},
        )

        game_map = [["agent.test_group"]]
        env = MettaGrid(game_config, game_map, 42)

        observations, _ = env.reset()
        agent_obs = observations[0]

        # Find game rewards token
        game_rewards_token = None
        for token in agent_obs:
            if token[0] != 0xFF and token[1] == 13:
                game_rewards_token = token
                break

        assert game_rewards_token is not None

        # Expected packed value:
        # ore (0.0) -> 0 (bits 7-6: 00)
        # battery (0.5) -> 1 (bits 5-4: 01)
        # laser (1.0) -> 2 (bits 3-2: 10)
        # armor (10.0) -> 3 (bits 1-0: 11)
        # Binary: 00 01 10 11 = 0x1B = 27
        expected_packed = (0 << 6) | (1 << 4) | (2 << 2) | 3
        assert game_rewards_token[2] == expected_packed
