#!/usr/bin/env python3
"""
Test script for the Basal Ganglia model.

This script tests that the basal ganglia model can be instantiated
and configured correctly.
"""

import os
import sys
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue, OmegaConfBaseException

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from metta.agent.metta_agent import MettaAgent


def test_basal_ganglia_model():
    """Test the basal ganglia model instantiation and forward pass."""
    print("🧪 Testing Basal Ganglia Model Instantiation")

    try:
        # Load the full run config
        with hydra.initialize(version_base=None, config_path="configs"):
            run_cfg = hydra.compose(config_name="user/train_basal_ganglia")
        print(f"✅ Run config loaded. Keys: {list(run_cfg.keys())}")

        # Print the structure of the run config
        for key in run_cfg.keys():
            try:
                value = run_cfg[key]
                print(f"   {key}: {type(value).__name__}")
                if hasattr(value, 'keys'):
                    print(f"     subkeys: {list(value.keys())}")
            except Exception as e:
                print(f"   {key}: Error accessing - {e}")

        # Use the agent config from the run config
        agent_cfg = run_cfg['user']['agent']
        print(f"✅ Agent config from run config. Keys: {list(agent_cfg.keys())}")

        print(f"   components: {agent_cfg['components']}")
        print(f"   components type: {type(agent_cfg['components'])}")

        # Create a mock environment for testing
        class MockEnv:
            def __init__(self):
                self.single_observation_space = type('obj', (object,), {
                    'shape': (23, 15, 15),  # Grid observation shape
                    'dtype': 'float32'
                })()
                self.single_action_space = type('obj', (object,), {
                    'nvec': [5, 10]  # Mock action space
                })()

        mock_env = MockEnv()

        # Mock feature normalizations
        feature_normalizations = {i: 1.0 for i in range(23)}

        try:
            # Instantiate the agent
            if 'device' in agent_cfg:
                del agent_cfg['device']
            from omegaconf import OmegaConf
            minimal_cfg = OmegaConf.create({'components': agent_cfg['components']})
            print(f"   minimal_cfg keys: {list(minimal_cfg.keys())}")
            agent = MettaAgent(
                obs_space=mock_env.single_observation_space,
                obs_width=15,
                obs_height=15,
                action_space=mock_env.single_action_space,
                feature_normalizations=feature_normalizations,
                device="cpu",
                cfg=agent_cfg
            )

            print("✅ Agent instantiated successfully")

            # Initialize the agent with action space
            action_names = ["move", "interact"]
            action_max_params = [4, 9]  # move: [0,1,2,3,4], interact: [0,1,2,3,4,5,6,7,8,9]

            # Create simple test features
            features = {
                "type_id": {"id": 0, "type": "categorical"},
                "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
            }

            agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

            # Test that the agent has the expected components
            print("✅ Agent components:")
            for name, component in agent.components.items():
                print(f"   - {name}: {type(component).__name__}")

            # Test forward pass with mock data
            mock_obs = torch.randn(1, 23, 15, 15)  # [batch, channels, width, height]
            mock_state = PolicyState()

            print("🧪 Testing forward pass...")

            try:
                with torch.no_grad():
                    actions, log_probs, entropy, value, full_log_probs = agent(mock_obs, mock_state)
                print("✅ Forward pass successful")
                print(f"   Actions shape: {actions.shape}")
                print(f"   Value shape: {value.shape}")
                print(f"   Log probs shape: {log_probs.shape}")

                # Test basal ganglia stats collection
                print("🧪 Testing basal ganglia stats collection...")
                basal_ganglia_stats = agent.get_basal_ganglia_stats()
                print(f"✅ Basal ganglia stats: {basal_ganglia_stats}")

            except Exception as e:
                import traceback
                print(f"❌ Forward pass failed: {e}")
                traceback.print_exc()
                return False

            print("🎉 All tests passed! Basal Ganglia model is working correctly.")
            return True

        except Exception as e:
            print(f"❌ Agent instantiation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basal_ganglia_model()
    sys.exit(0 if success else 1)
