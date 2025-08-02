import glob

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from metta.agent.agent_config import AgentConfig, create_agent_config


class TestAgentConfig:
    def _load_config(self, config_file: str) -> AgentConfig:
        cfg: DictConfig = OmegaConf.load(config_file)  # type: ignore
        return create_agent_config(cfg)

    def test_load_all_agent_configs(self):
        """Test that all agent configs in configs/agent can be loaded and parsed."""
        config_files = glob.glob("configs/agent/*.yaml")
        assert len(config_files) > 0, "No agent config files found"

        loaded_configs = []

        for config_file in config_files:
            config_name = config_file.split("/")[-1].replace(".yaml", "")

            agent_config = self._load_config(config_file)
            # Basic validation
            assert agent_config.target_ == "metta.agent.metta_agent.MettaAgent"
            assert agent_config.observations.obs_key == "grid_obs"
            assert agent_config.clip_range >= 0
            assert agent_config.analyze_weights_interval >= 0
            assert agent_config.l2_init_weight_update_interval >= 0

            # Validate required components exist
            assert "_obs_" in agent_config.components
            assert "_core_" in agent_config.components
            assert "_action_embeds_" in agent_config.components
            assert "_action_" in agent_config.components
            assert "_value_" in agent_config.components

            # Validate _core_ has required fields
            assert agent_config.components["_core_"]["output_size"] is not None
            assert agent_config.components["_core_"]["nn_params"]["num_layers"] is not None

            # Check that _target_ is set for all components
            for comp_name in ["_obs_", "_core_", "_action_embeds_", "_action_", "_value_"]:
                comp = agent_config.components[comp_name]
                assert "_target_" in comp, f"{comp_name} missing _target_"

            loaded_configs.append((config_name, agent_config))

        # Print summary
        print(f"Successfully loaded {len(loaded_configs)} agent configs:")
        for name, config in loaded_configs:
            print(f"  - {name}: {config.components['_core_']['_target_']}")

    def test_latent_attn_tiny_config(self):
        """Test specific validation for latent_attn_tiny config."""
        agent_config = self._load_config("configs/agent/latent_attn_tiny.yaml")

        # Check specific components for this architecture
        assert "obs_normalizer" in agent_config.components
        assert "obs_fourier" in agent_config.components
        assert "obs_cross_attn" in agent_config.components

        # Check obs_cross_attn specific fields
        obs_cross_attn = agent_config.components["obs_cross_attn"]
        assert obs_cross_attn["out_dim"] == 128
        assert obs_cross_attn["use_mask"] is True
        assert obs_cross_attn["num_layers"] == 2
        assert obs_cross_attn["num_heads"] == 4

    def test_reference_design_config(self):
        """Test specific validation for reference_design config."""
        agent_config = self._load_config("configs/agent/reference_design.yaml")

        # Check policy selector
        assert agent_config.policy_selector is not None
        assert agent_config.policy_selector.type == "top"
        assert agent_config.policy_selector.range == 0
        assert agent_config.policy_selector.metric == "final.score"

        # Check specific components
        assert "channel_selector_0-11" in agent_config.components
        assert "cnn_merger" in agent_config.components

        # Check merge layer source configuration
        cnn_merger = agent_config.components["cnn_merger"]
        assert len(cnn_merger["sources"]) == 2
        assert cnn_merger["sources"][0]["name"] == "cnn2_channels_0-11"
        assert cnn_merger["sources"][0]["slice"] == [0, 64]
        assert cnn_merger["sources"][0]["dim"] == 1

    def test_fast_config(self):
        """Test specific validation for fast config."""
        agent_config = self._load_config("configs/agent/fast.yaml")

        # Check it has CNN layers
        assert "cnn1" in agent_config.components
        assert "cnn2" in agent_config.components
        assert agent_config.components["cnn1"]["_target_"] == "metta.agent.lib.nn_layer_library.Conv2d"

    def test_invalid_config_missing_core(self):
        """Test that config without _core_ component fails validation."""
        cfg = {
            "_target_": "metta.agent.metta_agent.MettaAgent",
            "observations": {"obs_key": "grid_obs"},
            "clip_range": 0,
            "analyze_weights_interval": 300,
            "l2_init_weight_update_interval": 0,
            "components": {
                "_obs_": {"_target_": "test.Obs", "sources": None},
                "_action_embeds_": {"_target_": "test.ActionEmbed", "sources": None},
                "_action_": {"_target_": "test.Action", "sources": None},
                "_value_": {"_target_": "test.Value", "sources": None},
                # Missing _core_
            },
        }

        with pytest.raises(ValidationError):
            create_agent_config(OmegaConf.create(cfg))

    def test_invalid_config_core_missing_output_size(self):
        """Test that _core_ without output_size fails validation."""
        cfg = {
            "_target_": "metta.agent.metta_agent.MettaAgent",
            "observations": {"obs_key": "grid_obs"},
            "clip_range": 0,
            "analyze_weights_interval": 300,
            "l2_init_weight_update_interval": 0,
            "components": {
                "_obs_": {"_target_": "test.Obs", "sources": None},
                "_core_": {"_target_": "test.Core", "sources": None, "nn_params": {"num_layers": 2}},
                "_action_embeds_": {"_target_": "test.ActionEmbed", "sources": None},
                "_action_": {"_target_": "test.Action", "sources": None},
                "_value_": {"_target_": "test.Value", "sources": None},
            },
        }

        with pytest.raises(ValueError, match="_core_ component must have 'output_size' defined"):
            create_agent_config(OmegaConf.create(cfg))
