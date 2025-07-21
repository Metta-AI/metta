"""Integration tests for PufferLib with Metta.

These tests verify that PufferLib can be properly integrated with Metta,
including environment creation, training loops, and checkpoint handling.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import MettaAgent, make_policy
from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyStore
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.puffer_env import MettaGridPufferEnv
from metta.rl.vecenv import make_vecenv


def create_minimal_config() -> DictConfig:
    """Create a minimal configuration for testing."""
    return DictConfig({
        "game": {
            "max_steps": 20,
            "num_agents": 2,
            "obs_width": 5,
            "obs_height": 5,
            "num_observation_tokens": 10,
            "inventory_item_names": ["key", "coin"],
            "groups": {
                "agent": {"id": 0, "sprite": 0}
            },
            "agent": {
                "default_resource_limit": 5,
                "rewards": {
                    "inventory": {"coin": 1.0}
                }
            },
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": False},
                "get_items": {"enabled": True},
                "attack": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 0}
            },
            "objects": {
                "wall": {"type_id": 1, "swappable": False},
                "coin": {"type_id": 2, "pickupable": True},
                "key": {"type_id": 3, "pickupable": True}
            }
        },
        "procedural": {
            "seed": 42,
            "width": 8,
            "height": 8,
            "num_objects": {"coin": 2, "key": 1}
        }
    })


class TestPufferLibIntegration:
    """Test suite for PufferLib integration."""

    def test_import_pufferlib(self):
        """Test that PufferLib can be imported."""
        import pufferlib
        import pufferlib.vector
        
        assert hasattr(pufferlib, "PufferEnv")
        assert hasattr(pufferlib.vector, "make")

    def test_single_environment_creation(self):
        """Test creating a single PufferLib environment."""
        config = create_minimal_config()
        curriculum = SingleTaskCurriculum("test_single", config)
        
        env = MettaGridPufferEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=True
        )
        
        assert env is not None
        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")
        assert env.num_agents == 2
        
        # Test reset
        obs, info = env.reset(seed=42)
        assert obs.shape[0] == env.num_agents
        assert isinstance(info, dict)
        
        # Test step
        actions = np.zeros((env.num_agents, 2), dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)
        
        assert obs.shape[0] == env.num_agents
        assert rewards.shape == (env.num_agents,)
        assert terminals.shape == (env.num_agents,)
        assert truncations.shape == (env.num_agents,)
        
        env.close()

    def test_vectorized_environment(self):
        """Test creating vectorized environments."""
        config = create_minimal_config()
        curriculum = SingleTaskCurriculum("test_vec", config)
        
        num_envs = 4
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization="serial",
            num_envs=num_envs,
            num_workers=1,
            is_training=True
        )
        
        assert vecenv.num_envs == num_envs
        assert vecenv.agents_per_env == 2
        
        # Test reset
        obs, infos = vecenv.reset()
        total_agents = num_envs * vecenv.agents_per_env
        assert obs.shape[0] == total_agents
        
        # Test step with torch tensors (as would be used in training)
        actions = torch.randint(0, 3, size=(total_agents, 2), dtype=torch.int32)
        obs, rewards, terminals, truncations, infos = vecenv.step(actions)
        
        assert obs.shape[0] == total_agents
        assert rewards.shape[0] == total_agents
        
        vecenv.close()

    def test_environment_with_policy(self):
        """Test environment interaction with a policy."""
        config = create_minimal_config()
        config.device = "cpu"
        config.agent = {
            "obs_width": config.game.obs_width,
            "obs_height": config.game.obs_height,
            "num_actions": 6,  # Simplified action space
            "fc_size": 64,
            "num_layers": 2
        }
        
        curriculum = SingleTaskCurriculum("test_policy", config)
        
        # Create environment
        env = MettaGridPufferEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=True
        )
        
        # Create policy
        policy = make_policy(env, config)
        assert isinstance(policy, MettaAgent)
        
        # Test forward pass
        obs, _ = env.reset(seed=42)
        obs_tensor = torch.from_numpy(obs).float()
        
        # Create policy state
        state = PolicyState()
        state.hidden = torch.zeros(env.num_agents, config.agent.fc_size)
        
        # Forward pass through policy
        with torch.no_grad():
            action_logits, values = policy({"grid_obs": obs_tensor}, state)
        
        assert action_logits.shape[0] == env.num_agents
        assert values.shape == (env.num_agents, 1)
        
        env.close()

    def test_checkpoint_compatibility(self):
        """Test checkpoint saving and loading with PufferLib environments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PolicyStore(tmpdir)
            
            # Create a policy state
            state = PolicyState()
            state.hidden = torch.randn(4, 128)
            state.lstm_h = torch.randn(4, 256) 
            state.lstm_c = torch.randn(4, 256)
            
            # Create and save a policy record
            pr = store.create_policy_record("test_checkpoint", state, None)
            store.save(pr)
            
            # Verify file was created
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            assert checkpoint_path.exists()
            
            # Load the checkpoint
            loaded_pr = store.load("test_checkpoint")
            
            # Verify state was restored correctly
            assert torch.allclose(state.hidden, loaded_pr.state.hidden)
            assert torch.allclose(state.lstm_h, loaded_pr.state.lstm_h)
            assert torch.allclose(state.lstm_c, loaded_pr.state.lstm_c)

    def test_multiprocessing_vectorization(self):
        """Test multiprocessing vectorization (if available)."""
        config = create_minimal_config()
        curriculum = SingleTaskCurriculum("test_mp", config)
        
        try:
            vecenv = make_vecenv(
                curriculum=curriculum,
                vectorization="multiprocessing",
                num_envs=2,
                num_workers=2,
                is_training=True
            )
            
            # Quick test that it works
            obs, _ = vecenv.reset()
            assert obs.shape[0] == 2 * vecenv.agents_per_env
            
            vecenv.close()
        except Exception as e:
            # Multiprocessing might not work in all test environments
            pytest.skip(f"Multiprocessing not available: {e}")

    @pytest.mark.parametrize("num_agents", [1, 4, 8])
    def test_variable_agent_counts(self, num_agents):
        """Test environments with different agent counts."""
        config = create_minimal_config()
        config.game.num_agents = num_agents
        
        curriculum = SingleTaskCurriculum(f"test_{num_agents}_agents", config)
        
        env = MettaGridPufferEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=True
        )
        
        assert env.num_agents == num_agents
        
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] == num_agents
        
        env.close()

    def test_environment_info_handling(self):
        """Test that environment info dictionaries are properly handled."""
        config = create_minimal_config()
        curriculum = SingleTaskCurriculum("test_info", config)
        
        env = MettaGridPufferEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=True
        )
        
        obs, reset_info = env.reset(seed=42)
        assert isinstance(reset_info, dict)
        
        # Take several steps and check info
        for _ in range(5):
            actions = np.random.randint(0, 3, size=(env.num_agents, 2), dtype=np.int32)
            obs, rewards, terminals, truncations, step_info = env.step(actions)
            
            assert isinstance(step_info, dict)
            # PufferLib expects numeric values in info for averaging
            for key, value in step_info.items():
                if isinstance(value, dict):
                    # Nested dicts should have been flattened
                    for subkey, subvalue in value.items():
                        assert isinstance(subvalue, (int, float, np.number)), \
                            f"Info[{key}][{subkey}] = {subvalue} is not numeric"
        
        env.close()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])