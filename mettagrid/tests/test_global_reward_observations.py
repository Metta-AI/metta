from metta.mettagrid.mettagrid_c import AgentConfig, GameConfig, GlobalObsConfig, MettaGrid


class TestGlobalRewardObservations:
    """Test global reward observation features in MettaGrid."""

    def test_resource_rewards_observation(self):
        """Test that inventory rewards global observation is correctly packed and included."""
        # Define inventory items - first 8 will be included in packed rewards
        inventory_items = ["item0", "item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9"]

        # Create agent configs with different reward values
        agent_config = AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="test_group",
            freeze_duration=0,
            action_failure_penalty=0.0,
            resource_limits={i: 100 for i in range(len(inventory_items))},
            resource_rewards={
                0: 0.25,  # item0: has reward (bit = 1)
                1: 0.0,  # item1: no reward (bit = 0)
                2: 1.5,  # item2: has reward (bit = 1)
                3: 0.0,  # item3: no reward (bit = 0)
                4: 2.0,  # item4: has reward (bit = 1)
                5: -0.5,  # item5: negative reward (bit = 0)
                6: 0.1,  # item6: has reward (bit = 1)
                7: 0.0,  # item7: no reward (bit = 0)
                8: 5.0,  # item8: has reward but not included (beyond 8 items)
                9: 1.0,  # item9: has reward but not included (beyond 8 items)
            },
            resource_reward_max={i: 100 for i in range(len(inventory_items))},
            group_reward_pct=0.0,
        )

        # Create global obs config - uses defaults
        global_obs_config = GlobalObsConfig(resource_rewards=True)

        # Create game config
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
            track_movement_metrics=False,  # Explicitly set for clarity
            no_agent_interference=False,
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

        # Find inventory rewards token
        # Feature ID 13 is ResourceRewards, should be at center as global token
        resource_rewards_token = None
        for token in agent_obs:
            if token[0] != 0xFF and token[1] == 13:  # Feature ID 13 is ResourceRewards
                resource_rewards_token = token
                break

        assert resource_rewards_token is not None, "Inventory rewards token not found in observation"

        # Expected packed value with 1-bit per item:
        # item0 (0.25) -> 1 (bit 7)
        # item1 (0.0) -> 0 (bit 6)
        # item2 (1.5) -> 1 (bit 5)
        # item3 (0.0) -> 0 (bit 4)
        # item4 (2.0) -> 1 (bit 3)
        # item5 (-0.5) -> 0 (bit 2)
        # item6 (0.1) -> 1 (bit 1)
        # item7 (0.0) -> 0 (bit 0)
        # Binary: 10101010 = 0xAA = 170
        expected_packed = 0b10101010
        assert expected_packed == 170, f"Expected packed calculation wrong: {expected_packed}"

        assert resource_rewards_token[2] == expected_packed, (
            f"Incorrect packed inventory rewards value. Expected: {expected_packed}, Got: {resource_rewards_token[2]}"
        )

    def test_resource_rewards_only_for_observing_agent(self):
        """Test that inventory rewards observation is included as global token for each agent."""
        inventory_items = ["item0", "item1", "item2", "item3"]

        agent_config = AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="test_group",
            freeze_duration=0,
            action_failure_penalty=0.0,
            resource_limits={i: 100 for i in range(len(inventory_items))},
            resource_rewards={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            resource_reward_max={i: 100 for i in range(len(inventory_items))},
            group_reward_pct=0.0,
        )

        # Create global obs config - uses defaults
        global_obs_config = GlobalObsConfig(resource_rewards=True)

        game_config = GameConfig(
            num_agents=2,  # Two agents to test multi-agent case
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            global_obs=global_obs_config,
            actions={},
            objects={"agent.test_group": agent_config},
            track_movement_metrics=False,
            no_agent_interference=False,
        )

        game_map = [["agent.test_group", "agent.test_group"]]  # Two agents
        env = MettaGrid(game_config, game_map, 42)

        observations, _ = env.reset()

        # Check that each agent sees their own game rewards at center
        for agent_idx in range(2):
            agent_obs = observations[agent_idx]

            # Find game rewards token at center (should be at position 2,2 for 5x5 observation)
            # The center is packed as position (2,2) for a 5x5 observation window
            center_packed = (2 << 4) | 2  # Row 2, Col 2

            resource_rewards_found_at_center = False
            for token in agent_obs:
                if token[0] == center_packed and token[1] == 13:  # Feature ID 13 is ResourceRewards
                    resource_rewards_found_at_center = True
                    # All 4 items have rewards, so packed value should be 0b11110000 = 240
                    assert token[2] == 0b11110000, f"Wrong packed value for agent {agent_idx}: {token[2]}"
                    break

            assert resource_rewards_found_at_center, f"Game rewards not found at center for agent {agent_idx}"

    def test_resource_rewards_with_partial_items(self):
        """Test inventory rewards with fewer than 8 items."""
        # Only 3 items, testing that we handle fewer than 8 items
        inventory_items = ["item0", "item1", "item2"]

        agent_config = AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="test_group",
            freeze_duration=0,
            action_failure_penalty=0.0,
            resource_limits={i: 100 for i in range(len(inventory_items))},
            resource_rewards={
                0: 2.0,  # item0: has reward (bit = 1)
                1: 0.0,  # item1: no reward (bit = 0)
                2: 0.3,  # item2: has reward (bit = 1)
            },
            resource_reward_max={i: 100 for i in range(len(inventory_items))},
            group_reward_pct=0.0,
        )

        # Create global obs config - uses defaults
        global_obs_config = GlobalObsConfig(resource_rewards=True)

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
            track_movement_metrics=False,
            no_agent_interference=False,
        )

        game_map = [["agent.test_group"]]
        env = MettaGrid(game_config, game_map, 42)

        observations, _ = env.reset()
        agent_obs = observations[0]

        # Find game rewards token
        resource_rewards_token = None
        for token in agent_obs:
            if token[0] != 0xFF and token[1] == 13:
                resource_rewards_token = token
                break

        assert resource_rewards_token is not None

        # Expected packed value with 1-bit per item:
        # item0 (2.0) -> 1 (bit 7)
        # item1 (0.0) -> 0 (bit 6)
        # item2 (0.3) -> 1 (bit 5)
        # Remaining 5 bits are 0 (no more items)
        # Binary: 10100000 = 0xA0 = 160
        expected_packed = 0b10100000
        assert resource_rewards_token[2] == expected_packed

    def test_resource_rewards_binary_quantization(self):
        """Test the binary quantization of reward values."""
        inventory_items = ["item0", "item1", "item2", "item3", "item4", "item5", "item6", "item7"]

        # Test edge cases of quantization
        agent_config = AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="test_group",
            freeze_duration=0,
            action_failure_penalty=0.0,
            resource_limits={i: 100 for i in range(len(inventory_items))},
            resource_rewards={
                0: 0.0,  # item0: exactly 0 -> 0
                1: 0.01,  # item1: small positive -> 1
                2: 1.0,  # item2: positive -> 1
                3: -1.0,  # item3: negative -> 0
                4: 100.0,  # item4: large positive -> 1
                5: -0.1,  # item5: small negative -> 0
                6: 0.5,  # item6: positive -> 1
                7: 0.0,  # item7: zero -> 0
            },
            resource_reward_max={i: 100 for i in range(len(inventory_items))},
            group_reward_pct=0.0,
        )

        # Create global obs config - uses defaults
        global_obs_config = GlobalObsConfig(resource_rewards=True)

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
            track_movement_metrics=False,
            no_agent_interference=False,
        )

        game_map = [["agent.test_group"]]
        env = MettaGrid(game_config, game_map, 42)

        observations, _ = env.reset()
        agent_obs = observations[0]

        # Find game rewards token
        resource_rewards_token = None
        for token in agent_obs:
            if token[0] != 0xFF and token[1] == 13:
                resource_rewards_token = token
                break

        assert resource_rewards_token is not None

        # Expected packed value with 1-bit per item:
        # item0 (0.0) -> 0 (bit 7)
        # item1 (0.01) -> 1 (bit 6)
        # item2 (1.0) -> 1 (bit 5)
        # item3 (-1.0) -> 0 (bit 4)
        # item4 (100.0) -> 1 (bit 3)
        # item5 (-0.1) -> 0 (bit 2)
        # item6 (0.5) -> 1 (bit 1)
        # item7 (0.0) -> 0 (bit 0)
        # Binary: 01101010 = 0x6A = 106
        expected_packed = 0b01101010
        assert resource_rewards_token[2] == expected_packed
