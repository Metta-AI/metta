"""
Tests for LSTM state reset behavior between episodes and trials.

This module tests the reset_lstm_state_between_episodes functionality to ensure that:
1. LSTM states persist across trial boundaries within the same episode
2. LSTM states reset at episode boundaries when reset_lstm_state_between_episodes=True
3. LSTM states persist across episode boundaries when reset_lstm_state_between_episodes=False
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from metta.rl.experience import Experience


class TestLSTMStateReset:
    """Test LSTM state reset behavior between episodes and trials."""

    @pytest.fixture
    def mock_obs_space(self):
        """Create a mock observation space."""
        mock_space = Mock()
        mock_space.shape = (11, 11, 3)  # Typical MettaGrid observation shape
        mock_space.dtype = np.float32  # Changed from uint8 to float32
        return mock_space

    @pytest.fixture
    def mock_action_space(self):
        """Create a mock action space."""
        mock_space = Mock()
        mock_space.shape = (2,)  # Typical MettaGrid action shape
        mock_space.dtype = np.int32
        return mock_space

    @pytest.fixture
    def experience_with_reset(self, mock_obs_space, mock_action_space):
        """Create Experience instance with reset_lstm_state_between_episodes=True."""
        return Experience(
            total_agents=2,
            batch_size=128,
            bptt_horizon=32,
            minibatch_size=64,
            max_minibatch_size=64,
            obs_space=mock_obs_space,
            atn_space=mock_action_space,
            device="cpu",
            hidden_size=128,
            num_lstm_layers=2,
            reset_lstm_state_between_episodes=True,
        )

    @pytest.fixture
    def experience_without_reset(self, mock_obs_space, mock_action_space):
        """Create Experience instance with reset_lstm_state_between_episodes=False."""
        return Experience(
            total_agents=2,
            batch_size=128,
            bptt_horizon=32,
            minibatch_size=64,
            max_minibatch_size=64,
            obs_space=mock_obs_space,
            atn_space=mock_action_space,
            device="cpu",
            hidden_size=128,
            num_lstm_layers=2,
            reset_lstm_state_between_episodes=False,
        )

    def test_lstm_state_initialization(self, experience_with_reset):
        """Test that LSTM states are properly initialized."""
        # Check that LSTM states are created for each agent
        # The Experience class creates LSTM states for each batch, not per agent
        assert len(experience_with_reset.lstm_h) == 1  # Only one batch for 2 agents
        assert len(experience_with_reset.lstm_c) == 1

        # Check that states are properly shaped
        # Shape should be (num_layers, batch_size, hidden_size)
        assert experience_with_reset.lstm_h[0].shape == (2, 2, 128)  # (layers, batch, hidden)
        assert experience_with_reset.lstm_c[0].shape == (2, 2, 128)

        # Check that states are initialized to zero
        assert torch.all(experience_with_reset.lstm_h[0] == 0)
        assert torch.all(experience_with_reset.lstm_c[0] == 0)

    def test_lstm_state_persistence_across_trials(self, experience_with_reset):
        """Test that LSTM states persist across trial boundaries within the same episode."""
        # Simulate storing experience for multiple steps within the same episode
        # (trial boundaries don't trigger episode resets)

        # Create mock data
        obs = torch.randn(2, 11, 11, 3)
        actions = torch.randint(0, 10, (2, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(2)
        rewards = torch.randn(2)
        dones = torch.zeros(2, dtype=torch.bool)  # No episode done
        truncations = torch.zeros(2, dtype=torch.bool)
        values = torch.randn(2)
        mask = torch.ones(2, dtype=torch.bool)

        # Create non-zero LSTM state
        lstm_state = {"lstm_h": torch.randn(2, 2, 128), "lstm_c": torch.randn(2, 2, 128)}

        # Store experience for agent 0
        env_id = slice(0, 1)
        experience_with_reset.store(
            obs=obs[0:1],
            actions=actions[0:1],
            logprobs=logprobs[0:1],
            rewards=rewards[0:1],
            dones=dones[0:1],
            truncations=truncations[0:1],
            values=values[0:1],
            env_id=env_id,
            mask=mask[0:1],
            lstm_state=lstm_state,
        )

        # Verify LSTM state was stored
        stored_h, stored_c = experience_with_reset.get_lstm_state(0)
        assert stored_h is not None
        assert stored_c is not None
        assert torch.allclose(stored_h, lstm_state["lstm_h"])
        assert torch.allclose(stored_c, lstm_state["lstm_c"])

        # Simulate trial boundary (environment reset but episode continues)
        # This should NOT reset the LSTM state
        new_lstm_state = {"lstm_h": torch.randn(2, 2, 128), "lstm_c": torch.randn(2, 2, 128)}

        # Store more experience (simulating trial boundary)
        experience_with_reset.store(
            obs=obs[0:1],
            actions=actions[0:1],
            logprobs=logprobs[0:1],
            rewards=rewards[0:1],
            dones=dones[0:1],  # Still no episode done
            truncations=truncations[0:1],
            values=values[0:1],
            env_id=env_id,
            mask=mask[0:1],
            lstm_state=new_lstm_state,
        )

        # LSTM state should be updated, not reset
        updated_h, updated_c = experience_with_reset.get_lstm_state(0)
        assert updated_h is not None
        assert updated_c is not None
        assert torch.allclose(updated_h, new_lstm_state["lstm_h"])
        assert torch.allclose(updated_c, new_lstm_state["lstm_c"])

    def test_lstm_state_reset_at_episode_boundary(self, experience_with_reset):
        """Test that LSTM states reset at episode boundaries when reset_lstm_state_between_episodes=True."""
        # Create mock data
        obs = torch.randn(2, 11, 11, 3)
        actions = torch.randint(0, 10, (2, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(2)
        rewards = torch.randn(2)
        values = torch.randn(2)
        mask = torch.ones(2, dtype=torch.bool)

        # Create non-zero LSTM state
        lstm_state = {"lstm_h": torch.randn(2, 2, 128), "lstm_c": torch.randn(2, 2, 128)}

        # Store experience for agent 0
        env_id = slice(0, 1)
        experience_with_reset.store(
            obs=obs[0:1],
            actions=actions[0:1],
            logprobs=logprobs[0:1],
            rewards=rewards[0:1],
            dones=torch.zeros(1, dtype=torch.bool),  # Episode not done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values[0:1],
            env_id=env_id,
            mask=mask[0:1],
            lstm_state=lstm_state,
        )

        # Verify LSTM state was stored
        stored_h, stored_c = experience_with_reset.get_lstm_state(0)
        assert stored_h is not None
        assert stored_c is not None
        assert not torch.allclose(stored_h, torch.zeros_like(stored_h))
        assert not torch.allclose(stored_c, torch.zeros_like(stored_c))

        # Simulate episode completion (dones=True)
        experience_with_reset.store(
            obs=obs[0:1],
            actions=actions[0:1],
            logprobs=logprobs[0:1],
            rewards=rewards[0:1],
            dones=torch.ones(1, dtype=torch.bool),  # Episode done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values[0:1],
            env_id=env_id,
            mask=mask[0:1],
            lstm_state=lstm_state,
        )

        # LSTM state should be reset to zero
        reset_h, reset_c = experience_with_reset.get_lstm_state(0)
        assert reset_h is not None
        assert reset_c is not None
        assert torch.allclose(reset_h, torch.zeros_like(reset_h))
        assert torch.allclose(reset_c, torch.zeros_like(reset_c))

    def test_lstm_state_persistence_across_episodes(self, experience_without_reset):
        """Test that LSTM states persist across episode boundaries when reset_lstm_state_between_episodes=False."""
        # Create mock data
        obs = torch.randn(2, 11, 11, 3)
        actions = torch.randint(0, 10, (2, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(2)
        rewards = torch.randn(2)
        values = torch.randn(2)
        mask = torch.ones(2, dtype=torch.bool)

        # Create non-zero LSTM state
        lstm_state = {"lstm_h": torch.randn(2, 2, 128), "lstm_c": torch.randn(2, 2, 128)}

        # Store experience for agent 0
        env_id = slice(0, 1)
        experience_without_reset.store(
            obs=obs[0:1],
            actions=actions[0:1],
            logprobs=logprobs[0:1],
            rewards=rewards[0:1],
            dones=torch.zeros(1, dtype=torch.bool),
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values[0:1],
            env_id=env_id,
            mask=mask[0:1],
            lstm_state=lstm_state,
        )

        # Verify LSTM state was stored
        stored_h, stored_c = experience_without_reset.get_lstm_state(0)
        assert stored_h is not None
        assert stored_c is not None
        assert torch.allclose(stored_h, lstm_state["lstm_h"])
        assert torch.allclose(stored_c, lstm_state["lstm_c"])

        # Simulate episode completion (dones=True) but with reset_lstm_state_between_episodes=False
        experience_without_reset.store(
            obs=obs[0:1],
            actions=actions[0:1],
            logprobs=logprobs[0:1],
            rewards=rewards[0:1],
            dones=torch.ones(1, dtype=torch.bool),  # Episode done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values[0:1],
            env_id=env_id,
            mask=mask[0:1],
            lstm_state=lstm_state,
        )

        # LSTM state should NOT be reset (should persist)
        persistent_h, persistent_c = experience_without_reset.get_lstm_state(0)
        assert persistent_h is not None
        assert persistent_c is not None
        assert torch.allclose(persistent_h, lstm_state["lstm_h"])
        assert torch.allclose(persistent_c, lstm_state["lstm_c"])

    def test_multiple_agents_lstm_state_management(self, experience_with_reset):
        """Test LSTM state management for multiple agents."""
        # Create mock data for 2 agents
        obs = torch.randn(2, 11, 11, 3)
        actions = torch.randint(0, 10, (2, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(2)
        rewards = torch.randn(2)
        values = torch.randn(2)
        mask = torch.ones(2, dtype=torch.bool)

        # Create LSTM state for the batch (both agents share the same batch)
        lstm_state = {"lstm_h": torch.randn(2, 2, 128), "lstm_c": torch.randn(2, 2, 128)}

        # Store experience for both agents in the same batch
        env_id = slice(0, 2)  # Both agents together
        experience_with_reset.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.zeros(2, dtype=torch.bool),
            truncations=torch.zeros(2, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=lstm_state,
        )

        # Verify LSTM states are stored (only agent 0 has an entry since it's the batch starting index)
        h_0, c_0 = experience_with_reset.get_lstm_state(0)
        h_1, c_1 = experience_with_reset.get_lstm_state(1)

        # Agent 0 should have the LSTM state (batch starting index)
        assert torch.allclose(h_0, lstm_state["lstm_h"])
        assert torch.allclose(c_0, lstm_state["lstm_c"])

        # Agent 1 should return None since it doesn't have its own LSTM state entry
        # (it shares the same batch as agent 0)
        assert h_1 is None
        assert c_1 is None

        # Now test episode completion - only one agent completes
        experience_with_reset.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.tensor([True, False], dtype=torch.bool),  # Only agent 0 completes
            truncations=torch.zeros(2, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=lstm_state,
        )

        # Since reset_lstm_state_between_episodes=True, the entire batch LSTM state should be reset
        h_after, c_after = experience_with_reset.get_lstm_state(0)
        assert torch.allclose(h_after, torch.zeros_like(h_after))
        assert torch.allclose(c_after, torch.zeros_like(c_after))

    def test_bptt_horizon_episode_completion(self, experience_with_reset):
        """Test that episodes complete when BPTT horizon is reached."""
        # Create mock data
        obs = torch.randn(2, 11, 11, 3)
        actions = torch.randint(0, 10, (2, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(2)
        rewards = torch.randn(2)
        values = torch.randn(2)
        mask = torch.ones(2, dtype=torch.bool)

        # Create non-zero LSTM state
        lstm_state = {"lstm_h": torch.randn(2, 2, 128), "lstm_c": torch.randn(2, 2, 128)}

        env_id = slice(0, 2)  # Use both agents to avoid bounds issues

        # Fill up the BPTT horizon (30 steps, not 31 to avoid bounds)
        for _step in range(30):
            experience_with_reset.store(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                rewards=rewards,
                dones=torch.zeros(2, dtype=torch.bool),
                truncations=torch.zeros(2, dtype=torch.bool),
                values=values,
                env_id=env_id,
                mask=mask,
                lstm_state=lstm_state,
            )

        # The next step should trigger episode completion and LSTM reset
        experience_with_reset.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.zeros(2, dtype=torch.bool),  # No episode done, but BPTT horizon reached
            truncations=torch.zeros(2, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=lstm_state,
        )

        # Since reset_lstm_state_between_episodes=True, LSTM should NOT be reset at BPTT horizon
        # LSTM reset only happens on dones=True, not on BPTT horizon
        h_after, c_after = experience_with_reset.get_lstm_state(0)
        assert torch.allclose(h_after, lstm_state["lstm_h"])  # Should NOT be reset
        assert torch.allclose(c_after, lstm_state["lstm_c"])  # Should NOT be reset

        # Now test actual episode completion
        experience_with_reset.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.ones(2, dtype=torch.bool),  # Episode done
            truncations=torch.zeros(2, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=lstm_state,
        )

        # Now LSTM should be reset because dones=True
        h_final, c_final = experience_with_reset.get_lstm_state(0)
        assert torch.allclose(h_final, torch.zeros_like(h_final))
        assert torch.allclose(c_final, torch.zeros_like(c_final))

    def test_set_lstm_state(self, experience_with_reset):
        """Test the set_lstm_state method."""
        # Create new LSTM state
        new_lstm_h = torch.randn(2, 2, 128)
        new_lstm_c = torch.randn(2, 2, 128)

        # Set LSTM state for agent 0
        experience_with_reset.set_lstm_state(0, new_lstm_h, new_lstm_c)

        # Verify the state was set correctly
        stored_h, stored_c = experience_with_reset.get_lstm_state(0)
        assert stored_h is not None
        assert stored_c is not None
        assert torch.allclose(stored_h, new_lstm_h)
        assert torch.allclose(stored_c, new_lstm_c)

    def test_get_lstm_state_nonexistent_agent(self, experience_with_reset):
        """Test get_lstm_state for nonexistent agent."""
        # Try to get LSTM state for agent that doesn't exist
        h, c = experience_with_reset.get_lstm_state(999)
        assert h is None
        assert c is None


class TestLSTMStateResetWithNumTrials:
    """Test LSTM state reset behavior specifically with num_trials > 1 scenarios."""

    @pytest.fixture
    def mock_obs_space(self):
        """Create a mock observation space."""
        mock_space = Mock()
        mock_space.shape = (11, 11, 3)
        mock_space.dtype = np.float32  # Changed from uint8 to float32
        return mock_space

    @pytest.fixture
    def mock_action_space(self):
        """Create a mock action space."""
        mock_space = Mock()
        mock_space.shape = (2,)
        mock_space.dtype = np.int32
        return mock_space

    def test_num_trials_gt_one_lstm_persistence_across_trials(self, mock_obs_space, mock_action_space):
        """Test that LSTM states persist across trial boundaries when num_trials > 1."""
        # Create experience buffer with reset enabled
        experience = Experience(
            total_agents=1,
            batch_size=64,
            bptt_horizon=16,
            minibatch_size=32,
            max_minibatch_size=32,
            obs_space=mock_obs_space,
            atn_space=mock_action_space,
            device="cpu",
            hidden_size=64,
            num_lstm_layers=2,
            reset_lstm_state_between_episodes=True,
        )

        # Create mock data
        obs = torch.randn(1, 11, 11, 3)
        actions = torch.randint(0, 10, (1, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(1)
        rewards = torch.randn(1)
        values = torch.randn(1)
        mask = torch.ones(1, dtype=torch.bool)
        env_id = slice(0, 1)

        # Initial LSTM state
        initial_lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}

        # Store initial experience (trial 1)
        experience.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.zeros(1, dtype=torch.bool),  # Episode not done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=initial_lstm_state,
        )

        # Verify LSTM state was stored
        stored_h, stored_c = experience.get_lstm_state(0)
        assert stored_h is not None
        assert stored_c is not None
        assert torch.allclose(stored_h, initial_lstm_state["lstm_h"])
        assert torch.allclose(stored_c, initial_lstm_state["lstm_c"])

        # Simulate trial boundary (trial 1 -> trial 2, but same episode)
        # In a real environment with num_trials > 1, the environment would reset
        # but the episode would continue, so LSTM state should persist
        trial2_lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}

        # Store experience for trial 2 (still same episode)
        experience.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.zeros(1, dtype=torch.bool),  # Episode still not done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=trial2_lstm_state,
        )

        # LSTM state should be updated (not reset) because episode continues
        updated_h, updated_c = experience.get_lstm_state(0)
        assert updated_h is not None
        assert updated_c is not None
        assert torch.allclose(updated_h, trial2_lstm_state["lstm_h"])
        assert torch.allclose(updated_c, trial2_lstm_state["lstm_c"])

        # Simulate trial boundary (trial 2 -> trial 3, but same episode)
        trial3_lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}

        # Store experience for trial 3 (still same episode)
        experience.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.zeros(1, dtype=torch.bool),  # Episode still not done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=trial3_lstm_state,
        )

        # LSTM state should continue to persist across trial boundaries
        final_h, final_c = experience.get_lstm_state(0)
        assert final_h is not None
        assert final_c is not None
        assert torch.allclose(final_h, trial3_lstm_state["lstm_h"])
        assert torch.allclose(final_c, trial3_lstm_state["lstm_c"])

        # Now simulate episode completion (all trials done)
        experience.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.ones(1, dtype=torch.bool),  # Episode done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=trial3_lstm_state,
        )

        # LSTM state should be reset at episode boundary
        reset_h, reset_c = experience.get_lstm_state(0)
        assert reset_h is not None
        assert reset_c is not None
        assert torch.allclose(reset_h, torch.zeros_like(reset_h))
        assert torch.allclose(reset_c, torch.zeros_like(reset_c))

    def test_num_trials_gt_one_without_reset(self, mock_obs_space, mock_action_space):
        """Test that LSTM states persist across both trial and episode boundaries when
        reset_lstm_state_between_episodes=False."""
        # Create experience buffer with reset disabled
        experience = Experience(
            total_agents=1,
            batch_size=64,
            bptt_horizon=16,
            minibatch_size=32,
            max_minibatch_size=32,
            obs_space=mock_obs_space,
            atn_space=mock_action_space,
            device="cpu",
            hidden_size=64,
            num_lstm_layers=2,
            reset_lstm_state_between_episodes=False,
        )

        # Create mock data
        obs = torch.randn(1, 11, 11, 3)
        actions = torch.randint(0, 10, (1, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(1)
        rewards = torch.randn(1)
        values = torch.randn(1)
        mask = torch.ones(1, dtype=torch.bool)
        env_id = slice(0, 1)

        # Initial LSTM state
        initial_lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}

        # Store initial experience (trial 1)
        experience.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.zeros(1, dtype=torch.bool),
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=initial_lstm_state,
        )

        # Simulate multiple trial boundaries within the same episode
        for _trial in range(2, 5):  # Trials 2, 3, 4
            trial_lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}

            experience.store(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                rewards=rewards,
                dones=torch.zeros(1, dtype=torch.bool),  # Episode not done
                truncations=torch.zeros(1, dtype=torch.bool),
                values=values,
                env_id=env_id,
                mask=mask,
                lstm_state=trial_lstm_state,
            )

        # Simulate episode completion
        final_lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}

        experience.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.ones(1, dtype=torch.bool),  # Episode done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=final_lstm_state,
        )

        # LSTM state should persist even across episode boundaries
        persistent_h, persistent_c = experience.get_lstm_state(0)
        assert persistent_h is not None
        assert persistent_c is not None
        assert torch.allclose(persistent_h, final_lstm_state["lstm_h"])
        assert torch.allclose(persistent_c, final_lstm_state["lstm_c"])

    def test_trial_boundary_detection(self, mock_obs_space, mock_action_space):
        """Test that trial boundaries are properly detected and handled."""
        # This test simulates the key insight: trial boundaries (environment resets)
        # should not trigger LSTM resets, only episode boundaries should

        experience = Experience(
            total_agents=1,
            batch_size=64,
            bptt_horizon=16,
            minibatch_size=32,
            max_minibatch_size=32,
            obs_space=mock_obs_space,
            atn_space=mock_action_space,
            device="cpu",
            hidden_size=64,
            num_lstm_layers=2,
            reset_lstm_state_between_episodes=True,
        )

        # Create mock data
        obs = torch.randn(1, 11, 11, 3)
        actions = torch.randint(0, 10, (1, 2), dtype=torch.int32)  # Fixed dtype
        logprobs = torch.randn(1)
        rewards = torch.randn(1)
        values = torch.randn(1)
        mask = torch.ones(1, dtype=torch.bool)
        env_id = slice(0, 1)

        # Track LSTM state changes
        lstm_states = []

        # Simulate a sequence of trials within the same episode
        for _trial in range(3):  # 3 trials in the same episode
            lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}
            lstm_states.append(lstm_state)

            # Store experience for this trial
            experience.store(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                rewards=rewards,
                dones=torch.zeros(1, dtype=torch.bool),  # Episode not done
                truncations=torch.zeros(1, dtype=torch.bool),
                values=values,
                env_id=env_id,
                mask=mask,
                lstm_state=lstm_state,
            )

            # Verify LSTM state persists across trial boundaries
            current_h, current_c = experience.get_lstm_state(0)
            assert current_h is not None
            assert current_c is not None
            assert torch.allclose(current_h, lstm_state["lstm_h"])
            assert torch.allclose(current_c, lstm_state["lstm_c"])

        # Now simulate episode completion
        final_lstm_state = {"lstm_h": torch.randn(2, 1, 64), "lstm_c": torch.randn(2, 1, 64)}

        experience.store(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=torch.ones(1, dtype=torch.bool),  # Episode done
            truncations=torch.zeros(1, dtype=torch.bool),
            values=values,
            env_id=env_id,
            mask=mask,
            lstm_state=final_lstm_state,
        )

        # LSTM state should be reset at episode boundary
        reset_h, reset_c = experience.get_lstm_state(0)
        assert reset_h is not None
        assert reset_c is not None
        assert torch.allclose(reset_h, torch.zeros_like(reset_h))
        assert torch.allclose(reset_c, torch.zeros_like(reset_c))
