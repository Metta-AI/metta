from metta.mettagrid.mettagrid_c import AgentConfig, GameConfig, MettaGrid


class TestGlobalObservations:
    """Test global observation features in MettaGrid."""

    def test_game_rewards_observation(self):
        """Test that game_rewards global observation is correctly packed and included."""
        # Define inventory items with specific resources
        inventory_items = ["ore", "battery", "laser", "armor", "other_item"]

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
                0: 0.25,  # ore: low reward (should quantize to 1)
                1: 0.75,  # battery: medium reward (should quantize to 2)
                2: 1.5,  # laser: high reward (should quantize to 3)
                3: 0.0,  # armor: no reward (should quantize to 0)
                4: 2.0,  # other_item: not included in packed rewards
            },
            resource_reward_max={i: 100 for i in range(len(inventory_items))},
            group_reward_pct=0.0,
        )

        # Create game config with global_obs_game_rewards enabled
        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            actions={},
            objects={"agent.test_group": agent_config},
            global_obs_game_rewards=True,
        )

        # Create simple map
        game_map = [["agent.test_group"]]

        # Create environment
        env = MettaGrid(game_config, game_map, 42)

        # Reset environment
        observations, _ = env.reset()

        # Find the game_rewards token in observations
        # Global tokens are placed at the center of observation (2, 2) for 5x5 window
        # We need to look for feature_id = 13 (GameRewards)
        game_rewards_token = None

        for token_idx in range(observations.shape[1]):
            feature_id = observations[0, token_idx, 1]  # feature_id is at index 1
            if feature_id == 13:  # ObservationFeature::GameRewards
                game_rewards_token = observations[0, token_idx, :]
                break

        assert game_rewards_token is not None, "Game rewards token not found in observations"

        # Extract the packed value
        packed_value = game_rewards_token[2]  # value is at index 2

        # Unpack and verify
        # Expected packing: ore(1) << 6 | battery(2) << 4 | laser(3) << 2 | armor(0)
        # = 01 10 11 00 = 0x6C = 108
        expected_ore = 1
        expected_battery = 2
        expected_laser = 3
        expected_armor = 0

        extracted_ore = (packed_value >> 6) & 0x3
        extracted_battery = (packed_value >> 4) & 0x3
        extracted_laser = (packed_value >> 2) & 0x3
        extracted_armor = packed_value & 0x3

        assert extracted_ore == expected_ore, f"Expected ore={expected_ore}, got {extracted_ore}"
        assert extracted_battery == expected_battery, f"Expected battery={expected_battery}, got {extracted_battery}"
        assert extracted_laser == expected_laser, f"Expected laser={expected_laser}, got {extracted_laser}"
        assert extracted_armor == expected_armor, f"Expected armor={expected_armor}, got {extracted_armor}"

    def test_game_rewards_disabled(self):
        """Test that game_rewards token is not included when disabled."""
        inventory_items = ["ore", "battery", "laser", "armor"]

        agent_config = AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="test_group",
            freeze_duration=0,
            action_failure_penalty=0.0,
            resource_limits={i: 100 for i in range(len(inventory_items))},
            resource_rewards={i: 1.0 for i in range(len(inventory_items))},
            resource_reward_max={i: 100 for i in range(len(inventory_items))},
            group_reward_pct=0.0,
        )

        # Create game config with global_obs_game_rewards disabled
        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            actions={},
            objects={"agent.test_group": agent_config},
            global_obs_game_rewards=False,  # Disabled
        )

        game_map = [["agent.test_group"]]
        env = MettaGrid(game_config, game_map, 42)
        observations, _ = env.reset()

        # Check that game_rewards token is NOT present
        game_rewards_found = False
        for token_idx in range(observations.shape[1]):
            feature_id = observations[0, token_idx, 1]
            if feature_id == 13:  # ObservationFeature::GameRewards
                game_rewards_found = True
                break

        assert not game_rewards_found, "Game rewards token found when it should be disabled"

    def test_game_rewards_with_missing_resources(self):
        """Test game_rewards packing when some resources are missing."""
        # Only include some of the expected resources
        inventory_items = ["battery", "armor", "gold"]  # Missing ore and laser

        agent_config = AgentConfig(
            type_id=1,
            type_name="agent",
            group_id=0,
            group_name="test_group",
            freeze_duration=0,
            action_failure_penalty=0.0,
            resource_limits={i: 100 for i in range(len(inventory_items))},
            resource_rewards={
                0: 0.5,  # battery
                1: 1.0,  # armor
                2: 2.0,  # gold (not included in packed rewards)
            },
            resource_reward_max={i: 100 for i in range(len(inventory_items))},
            group_reward_pct=0.0,
        )

        game_config = GameConfig(
            num_agents=1,
            max_steps=100,
            episode_truncates=False,
            obs_width=5,
            obs_height=5,
            inventory_item_names=inventory_items,
            num_observation_tokens=50,
            actions={},
            objects={"agent.test_group": agent_config},
            global_obs_game_rewards=True,
        )

        game_map = [["agent.test_group"]]
        env = MettaGrid(game_config, game_map, 42)
        observations, _ = env.reset()

        # Find game_rewards token
        game_rewards_value = None
        for token_idx in range(observations.shape[1]):
            if observations[0, token_idx, 1] == 13:  # GameRewards
                game_rewards_value = observations[0, token_idx, 2]
                break

        assert game_rewards_value is not None

        # With missing ore and laser, expected packing:
        # ore(0) << 6 | battery(1) << 4 | laser(0) << 2 | armor(2)
        # = 00 01 00 10 = 0x12 = 18
        extracted_ore = (game_rewards_value >> 6) & 0x3
        extracted_battery = (game_rewards_value >> 4) & 0x3
        extracted_laser = (game_rewards_value >> 2) & 0x3
        extracted_armor = game_rewards_value & 0x3

        assert extracted_ore == 0, "Missing ore should be 0"
        assert extracted_battery == 1, f"Battery should be 1, got {extracted_battery}"
        assert extracted_laser == 0, "Missing laser should be 0"
        assert extracted_armor == 2, f"Armor should be 2, got {extracted_armor}"

    def test_game_rewards_quantization(self):
        """Test the quantization of reward values into 2-bit values."""
        inventory_items = ["ore", "battery", "laser", "armor"]

        # Test different reward values to verify quantization
        test_cases = [
            (0.0, 0),  # No reward -> 0
            (0.25, 1),  # Low reward -> 1
            (0.5, 1),  # Still low -> 1
            (0.75, 2),  # Medium reward -> 2
            (1.0, 2),  # Still medium -> 2
            (1.5, 3),  # High reward -> 3
            (10.0, 3),  # Very high -> 3 (capped)
        ]

        for ore_reward, expected_quantized in test_cases:
            agent_config = AgentConfig(
                type_id=1,
                type_name="agent",
                group_id=0,
                group_name="test_group",
                freeze_duration=0,
                action_failure_penalty=0.0,
                resource_limits={i: 100 for i in range(len(inventory_items))},
                resource_rewards={
                    0: ore_reward,  # Test various ore reward values
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
                resource_reward_max={i: 100 for i in range(len(inventory_items))},
                group_reward_pct=0.0,
            )

            game_config = GameConfig(
                num_agents=1,
                max_steps=100,
                episode_truncates=False,
                obs_width=5,
                obs_height=5,
                inventory_item_names=inventory_items,
                num_observation_tokens=50,
                actions={},
                objects={"agent.test_group": agent_config},
                global_obs_game_rewards=True,
            )

            game_map = [["agent.test_group"]]
            env = MettaGrid(game_config, game_map, 42)
            observations, _ = env.reset()

            # Find and check game_rewards value
            for token_idx in range(observations.shape[1]):
                if observations[0, token_idx, 1] == 13:  # GameRewards
                    packed_value = observations[0, token_idx, 2]
                    extracted_ore = (packed_value >> 6) & 0x3
                    assert extracted_ore == expected_quantized, (
                        f"Reward {ore_reward} should quantize to {expected_quantized}, got {extracted_ore}"
                    )
                    break
