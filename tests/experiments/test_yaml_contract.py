"""Test the YAML API contract between experiments and tools/train.py.

These tests ensure that the YAML we generate matches what tools/train.py expects.
They serve as a contract test - if train.py changes its expectations, these should fail.
"""

import yaml

from experiments.training_run_config import TrainingRunConfig
from metta.rl.trainer_config import OptimizerConfig, PPOConfig, TrainerConfig


class TestYAMLContract:
    """Test that generated YAML matches the contract expected by tools/train.py."""

    def test_minimal_yaml_has_required_fields(self):
        """Test that even minimal configs produce all required fields."""
        config = TrainingRunConfig()
        yaml_dict = config.serialize_to_yaml()

        # Required top-level fields for Hydra config
        assert "defaults" in yaml_dict
        assert isinstance(yaml_dict["defaults"], list)

        # Should have agent config in defaults
        agent_defaults = [d for d in yaml_dict["defaults"] if "agent:" in d]
        assert len(agent_defaults) == 1

        # Required trainer section
        assert "trainer" in yaml_dict
        trainer = yaml_dict["trainer"]

        # Essential trainer fields
        assert "total_timesteps" in trainer
        assert "num_workers" in trainer
        assert "batch_size" in trainer
        assert "curriculum" in trainer

        # Optimizer config
        assert "optimizer" in trainer
        assert "type" in trainer["optimizer"]
        assert "learning_rate" in trainer["optimizer"]

        # PPO config
        assert "ppo" in trainer
        assert "clip_coef" in trainer["ppo"]
        assert "ent_coef" in trainer["ppo"]

        # WandB config - only included when not using defaults
        # Using metta-research entity uses the default wandb config
        # so it won't have explicit wandb section

        # Other required fields
        assert "seed" in yaml_dict
        assert isinstance(yaml_dict["seed"], int)

    def test_yaml_structure_matches_hydra_expectations(self):
        """Test that YAML structure matches Hydra's config composition."""
        config = TrainingRunConfig(
            agent_config="latent_attn_tiny",
            curriculum="test/curriculum",
            wandb_tags=["exp1", "test"],
        )

        yaml_dict = config.serialize_to_yaml()

        # Check defaults structure for Hydra
        defaults = yaml_dict["defaults"]
        assert defaults[0] == "common"

        # Agent config should be in specific format
        agent_line = next(d for d in defaults if "agent:" in d)
        assert agent_line == "agent: latent_attn_tiny"

        # Tags should be a list
        assert isinstance(yaml_dict["wandb"]["tags"], list)
        assert yaml_dict["wandb"]["tags"] == ["exp1", "test"]

    def test_trainer_overrides_are_properly_nested(self):
        """Test that trainer config overrides maintain correct nesting."""
        from metta.rl.trainer_config import (
            CheckpointConfig,
            SimulationConfig,
            TorchProfilerConfig,
        )

        trainer = TrainerConfig(
            total_timesteps=5000,
            batch_size=128,
            minibatch_size=32,
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.0005,
                eps=1e-8,
            ),
            ppo=PPOConfig(
                clip_coef=0.25,
                ent_coef=0.02,
                vf_coef=0.6,
            ),
            curriculum="override/curriculum",
            num_workers=2,
            checkpoint=CheckpointConfig(checkpoint_dir="${run_dir}/checkpoints"),
            simulation=SimulationConfig(replay_dir="${run_dir}/replays"),
            profiler=TorchProfilerConfig(profile_dir="${run_dir}/torch_traces"),
        )

        config = TrainingRunConfig(
            curriculum="original/curriculum",  # Will be overridden
            trainer=trainer,
        )

        yaml_dict = config.serialize_to_yaml()

        # Check nesting structure
        t = yaml_dict["trainer"]
        assert t["total_timesteps"] == 5000
        assert t["batch_size"] == 128
        assert t["minibatch_size"] == 32

        # Optimizer nesting
        assert t["optimizer"]["type"] == "adam"
        assert t["optimizer"]["learning_rate"] == 0.0005
        assert t["optimizer"]["eps"] == 1e-8

        # PPO nesting
        assert t["ppo"]["clip_coef"] == 0.25
        assert t["ppo"]["ent_coef"] == 0.02
        assert t["ppo"]["vf_coef"] == 0.6

        # Curriculum from TrainingRunConfig wins (by design)
        assert t["curriculum"] == "original/curriculum"

    def test_yaml_file_is_loadable_by_hydra(self):
        """Test that generated YAML files can be loaded as Hydra configs."""
        config = TrainingRunConfig(
            agent_config="fast",
            curriculum="env/mettagrid/curriculum/test",
            wandb_entity="test-org",
            wandb_project="test-project",
        )

        yaml_path, yaml_dict = config.serialize_to_yaml_file()

        try:
            # Verify file exists and is valid YAML
            assert yaml_path.exists()
            assert yaml_path.suffix == ".yaml"

            # Load as Hydra would
            with open(yaml_path) as f:
                loaded = yaml.safe_load(f)

            # Should match what we serialized
            assert loaded == yaml_dict

            # Verify it has the package directive for Hydra
            assert "# @package _global_" in open(yaml_path).read()

        finally:
            # Clean up
            if yaml_path.exists():
                yaml_path.unlink()

    def test_additional_args_handling(self):
        """Test that additional_args were removed."""
        # additional_args functionality was removed per user request
        config = TrainingRunConfig()

        # Check that additional_args is not in the config
        assert not hasattr(config, "additional_args")

        yaml_dict = config.serialize_to_yaml()
        assert "additional_args" not in yaml_dict

    def test_special_fields_handling(self):
        """Test handling of special fields like py_agent and run_name_pattern."""
        config = TrainingRunConfig(
            py_agent="custom_agent",
            run_name_pattern="{experiment}_{timestamp}",
            bypass_mac_overrides=True,
        )

        yaml_dict = config.serialize_to_yaml()

        # These should be included at top level
        assert yaml_dict["py_agent"] == "custom_agent"
        assert yaml_dict["run_name_pattern"] == "{experiment}_{timestamp}"
        assert yaml_dict["bypass_mac_overrides"] is True

    def test_checkpoint_and_simulation_paths(self):
        """Test that checkpoint and simulation paths use proper variables."""
        config = TrainingRunConfig()
        yaml_dict = config.serialize_to_yaml()

        trainer = yaml_dict["trainer"]

        # Should use ${run_dir} variable for paths
        if "checkpoint" in trainer:
            assert "${run_dir}" in trainer["checkpoint"].get("checkpoint_dir", "")

        if "simulation" in trainer:
            assert "${run_dir}" in trainer["simulation"].get("replay_dir", "")

        if "profiler" in trainer:
            assert "${run_dir}" in trainer["profiler"].get("profile_dir", "")

    def test_empty_optional_fields_are_excluded(self):
        """Test that None/empty optional fields are excluded from YAML."""
        config = TrainingRunConfig(
            wandb_tags=None,
            wandb_group=None,
            wandb_notes=None,
            py_agent=None,
            run_name_pattern=None,
        )

        yaml_dict = config.serialize_to_yaml()

        # None values should appear as null in YAML (not omitted)
        # py_agent and run_name_pattern should be included as null
        assert yaml_dict["py_agent"] is None
        assert yaml_dict["run_name_pattern"] is None

        # wandb section may not exist if using defaults
        if "wandb" in yaml_dict:
            wandb = yaml_dict["wandb"]
            assert "tags" not in wandb
            assert "group" not in wandb
            assert "notes" not in wandb
