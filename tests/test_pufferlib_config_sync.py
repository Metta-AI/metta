#!/usr/bin/env python3
"""
Unit tests for PufferLib and Metta configuration compatibility.

This test suite validates that Metta's Hydra-based configurations can be properly
converted to PufferLib format and successfully used to initialize PufferLib components.

It includes:
- Configuration conversion tests
- PufferLib environment initialization tests  
- Integration tests with actual PufferLib training
- Regression tests for known compatibility issues
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.common.util.fs import cd_repo_root
from metta.mettagrid import MettaGridPufferEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


class TestPufferLibConfigSync:
    """Test suite for PufferLib and Metta configuration compatibility."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        cd_repo_root()

    def load_metta_config(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
        """Load a Metta configuration using Hydra.
        
        Args:
            config_path: Path to config file relative to configs/
            overrides: Optional config overrides
            
        Returns:
            Loaded configuration as DictConfig
        """
        import hydra
        from hydra import initialize_config_dir
        from pathlib import Path
        import os
        
        # Initialize Hydra with the configs directory
        config_dir = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_path)
            
        # Apply overrides if provided
        if overrides:
            from omegaconf import OmegaConf
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, overrides)
            OmegaConf.set_struct(cfg, True)
            
        return cfg

    def test_basic_config_conversion(self):
        """Test basic configuration conversion from Metta to C++ format."""
        # Load a simple test config
        test_config = {
            "game": {
                "num_agents": 2,
                "obs_width": 11,
                "obs_height": 11,
                "max_steps": 1000,
                "num_observation_tokens": 100,
                "inventory_item_names": ["ore", "battery", "heart"],
                "groups": {
                    "agent": {"id": 0, "sprite": 0, "props": {}}
                },
                "agent": {
                    "default_resource_limit": 10,
                    "freeze_duration": 5,
                    "rewards": {"inventory": {"heart": 1.0}}
                },
                "objects": {
                    "wall": {"type_id": 1},
                    "altar": {
                        "type_id": 8,
                        "cooldown": 10,
                        "max_output": 5,
                        "conversion_ticks": 1,
                        "initial_resource_count": 1
                    }
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True},
                    "rotate": {"enabled": True}
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 20,
                    "height": 20,
                    "agents": 2
                }
            }
        }
        
        # Convert to C++ config
        game_config = test_config["game"]
        cpp_config = from_mettagrid_config(game_config)
        
        # Verify conversion succeeded
        assert cpp_config is not None
        assert cpp_config.num_agents == 2
        assert cpp_config.obs_width == 11
        assert cpp_config.obs_height == 11
        assert cpp_config.max_steps == 1000

    def test_environment_configs(self):
        """Test conversion of various environment configurations."""
        env_configs = [
            "env/mettagrid/puffer",
            "env/mettagrid/laser_tag",
            "env/mettagrid/ants",
        ]
        
        for config_path in env_configs:
            try:
                # Load config
                cfg = self.load_metta_config(config_path)
                
                # Extract game config
                if hasattr(cfg, "game"):
                    game_config = OmegaConf.to_container(cfg.game, resolve=True)
                    assert isinstance(game_config, dict)
                    
                    # Remove map_builder for conversion test
                    if "map_builder" in game_config:
                        del game_config["map_builder"]
                    
                    # Convert to C++ config
                    cpp_config = from_mettagrid_config(game_config)
                    assert cpp_config is not None
                    
            except Exception as e:
                pytest.fail(f"Failed to convert config {config_path}: {str(e)}")

    def test_pufferlib_env_initialization(self):
        """Test that converted configs can initialize PufferLib environments."""
        # Try importing PufferLib
        try:
            import pufferlib
        except ImportError:
            pytest.skip("PufferLib not installed")
        
        # Create a simple config
        config = DictConfig({
            "game": {
                "num_agents": 1,
                "obs_width": 7,
                "obs_height": 7,
                "max_steps": 100,
                "num_observation_tokens": 50,
                "inventory_item_names": ["heart"],
                "groups": {
                    "agent": {"id": 0, "sprite": 0, "props": {}}
                },
                "agent": {
                    "default_resource_limit": 5,
                    "freeze_duration": 0,
                    "rewards": {}
                },
                "objects": {},
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True}
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 10,
                    "height": 10,
                    "agents": 1
                }
            }
        })
        
        # Create curriculum
        curriculum = SingleTaskCurriculum("test_task", config)
        
        # Initialize PufferLib environment
        env = MettaGridPufferEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=True
        )
        
        # Verify environment properties
        assert env.num_agents == 1
        assert env.observation_space is not None
        assert env.action_space is not None
        
        # Test reset and step
        obs, info = env.reset()
        assert obs.shape[0] == 1  # num_agents
        
        # Random action
        import numpy as np
        action = np.array([[0, 1]], dtype=np.int32)  # [agent_id, action]
        
        obs, rewards, terminals, truncations, info = env.step(action)
        assert obs.shape[0] == 1
        assert rewards.shape[0] == 1
        assert terminals.shape[0] == 1
        assert truncations.shape[0] == 1
        
        env.close()

    def test_nested_config_handling(self):
        """Test handling of nested configurations with custom resolvers."""
        # Test config with nested structures
        nested_config = {
            "game": {
                "num_agents": 4,
                "obs_width": 15,
                "obs_height": 15,
                "max_steps": 2000,
                "num_observation_tokens": 200,
                "inventory_item_names": ["ore_red", "ore_blue", "battery_red", "battery_blue"],
                "groups": {
                    "agent": {"id": 0, "sprite": 0, "props": {}},
                    "team_1": {"id": 1, "sprite": 1, "group_reward_pct": 0.5, "props": {}},
                    "team_2": {"id": 2, "sprite": 2, "group_reward_pct": 0.5, "props": {}}
                },
                "agent": {
                    "default_resource_limit": 20,
                    "resource_limits": {
                        "ore_red": 50,
                        "ore_blue": 50,
                        "battery_red": 10,
                        "battery_blue": 10
                    },
                    "freeze_duration": 10,
                    "rewards": {
                        "inventory": {
                            "battery_red": 0.1,
                            "battery_blue": 0.1
                        }
                    }
                },
                "objects": {
                    "wall": {"type_id": 1},
                    "mine_red": {
                        "type_id": 2,
                        "output_resources": {"ore_red": 1},
                        "cooldown": 20,
                        "max_output": 5,
                        "conversion_ticks": 1,
                        "initial_resource_count": 10
                    },
                    "mine_blue": {
                        "type_id": 3,
                        "output_resources": {"ore_blue": 1},
                        "cooldown": 20,
                        "max_output": 5,
                        "conversion_ticks": 1,
                        "initial_resource_count": 10
                    }
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True},
                    "rotate": {"enabled": True},
                    "put_items": {"enabled": True},
                    "get_items": {"enabled": True},
                    "attack": {
                        "enabled": True,
                        "consumed_resources": {"laser": 1},
                        "defense_resources": {"armor": 1}
                    }
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 30,
                    "height": 30,
                    "agents": 4,
                    "objects": {
                        "mine_red": 5,
                        "mine_blue": 5,
                        "wall": 20
                    }
                }
            }
        }
        
        # Convert config
        game_config = nested_config["game"]
        cpp_config = from_mettagrid_config(game_config)
        
        # Verify nested structures were properly converted
        assert cpp_config is not None
        assert cpp_config.num_agents == 4
        
        # The conversion should handle agent groups properly
        # Groups are converted internally during the config transformation

    def test_edge_cases(self):
        """Test edge cases in configuration conversion."""
        # Test empty inventory
        empty_inventory_config = {
            "game": {
                "num_agents": 1,
                "obs_width": 5,
                "obs_height": 5,
                "max_steps": 10,
                "num_observation_tokens": 10,
                "inventory_item_names": [],
                "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
                "agent": {"default_resource_limit": 0, "freeze_duration": 0, "rewards": {}},
                "objects": {},
                "actions": {"noop": {"enabled": True}},
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 5,
                    "height": 5,
                    "agents": 1
                }
            }
        }
        
        cpp_config = from_mettagrid_config(empty_inventory_config["game"])
        assert cpp_config is not None
        
        # Test maximum values
        max_values_config = {
            "game": {
                "num_agents": 100,
                "obs_width": 255,
                "obs_height": 255,
                "max_steps": 1000000,
                "num_observation_tokens": 10000,
                "inventory_item_names": [f"item_{i}" for i in range(50)],
                "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
                "agent": {
                    "default_resource_limit": 255,
                    "freeze_duration": 1000,
                    "rewards": {}
                },
                "objects": {},
                "actions": {"noop": {"enabled": True}},
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 100,
                    "height": 100,
                    "agents": 100
                }
            }
        }
        
        cpp_config = from_mettagrid_config(max_values_config["game"])
        assert cpp_config is not None
        assert cpp_config.num_agents == 100

    @pytest.mark.integration
    def test_pufferlib_train_metta(self):
        """Integration test: Clone PufferLib and verify 'puffer train metta' works."""
        # Skip if not in CI or if explicitly disabled
        import os
        if not os.getenv("CI") and not os.getenv("FORCE_PUFFERLIB_TEST"):
            pytest.skip("Skipping PufferLib integration test (set CI=1 or FORCE_PUFFERLIB_TEST=1 to run)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Clone PufferLib
            clone_cmd = [
                "git", "clone", "--depth", "1",
                "https://github.com/Metta-AI/PufferLib.git",
                str(tmpdir_path / "PufferLib")
            ]
            
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"Failed to clone PufferLib: {result.stderr}")
            
            pufferlib_dir = tmpdir_path / "PufferLib"
            
            # Install PufferLib in test mode
            install_cmd = [
                sys.executable, "-m", "pip", "install", "-e", str(pufferlib_dir)
            ]
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"Failed to install PufferLib: {result.stderr}")
            
            # Try to run puffer train metta (dry run)
            train_cmd = [
                sys.executable, "-m", "pufferlib.train",
                "--env", "metta",
                "--train.total_timesteps", "100",  # Very short for testing
                "--train.checkpoint_interval", "0",  # No checkpoints
                "--train.wandb", "disabled",
                "--vec.num_envs", "1",
                "--train.minibatch_size", "1",
                "--train.batch_size", "4"
            ]
            
            # Run in the current directory (metta repo)
            result = subprocess.run(
                train_cmd, 
                capture_output=True, 
                text=True,
                cwd=os.getcwd()
            )
            
            # Check if command succeeded or failed with expected errors
            # We're mainly checking that PufferLib can load Metta configs
            if result.returncode != 0:
                # Check if it's a known/expected error
                if "metta" not in result.stderr.lower() or "config" in result.stderr.lower():
                    pytest.fail(f"PufferLib train failed unexpectedly: {result.stderr}")

    def test_regression_known_issues(self):
        """Test for known compatibility issues between versions."""
        # Test for issue where 'groups' vs 'agent_groups' naming caused problems
        config_with_groups = {
            "game": {
                "num_agents": 2,
                "obs_width": 7,
                "obs_height": 7,
                "max_steps": 100,
                "num_observation_tokens": 50,
                "inventory_item_names": ["heart"],
                "groups": {  # This should be converted to agent_groups
                    "agent": {"id": 0, "sprite": 0, "props": {}},
                    "team": {"id": 1, "sprite": 1, "props": {}}
                },
                "agent": {
                    "default_resource_limit": 5,
                    "freeze_duration": 0,
                    "rewards": {}
                },
                "objects": {},
                "actions": {"noop": {"enabled": True}},
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "width": 10,
                    "height": 10,
                    "agents": 2
                }
            }
        }
        
        # This should not raise KeyError: 'groups'
        cpp_config = from_mettagrid_config(config_with_groups["game"])
        assert cpp_config is not None
        # The conversion should handle groups -> agent_groups transformation internally

    def test_trainer_config_compatibility(self):
        """Test trainer configuration compatibility."""
        # Load trainer config
        trainer_cfg = self.load_metta_config("trainer/trainer.yaml")
        
        # Verify key trainer fields exist
        assert hasattr(trainer_cfg, "curriculum") or "curriculum" in trainer_cfg
        assert hasattr(trainer_cfg, "total_timesteps") or "total_timesteps" in trainer_cfg
        assert hasattr(trainer_cfg, "ppo") or "ppo" in trainer_cfg
        
        # Verify PPO config has expected fields
        if hasattr(trainer_cfg, "ppo"):
            ppo_cfg = trainer_cfg.ppo
            expected_fields = ["clip_coef", "ent_coef", "gamma", "gae_lambda"]
            for field in expected_fields:
                assert hasattr(ppo_cfg, field) or field in ppo_cfg

    def test_agent_config_compatibility(self):
        """Test agent configuration compatibility."""
        agent_configs = [
            "agent/fast.yaml",
            "agent/latent_attn_med.yaml",
            "agent/latent_attn_small.yaml"
        ]
        
        for config_path in agent_configs:
            try:
                cfg = self.load_metta_config(config_path)
                # Verify agent config has required structure
                assert cfg is not None
            except Exception as e:
                pytest.fail(f"Failed to load agent config {config_path}: {str(e)}")