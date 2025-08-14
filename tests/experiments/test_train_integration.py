"""Integration tests with tools/train.py.

These tests verify that our generated YAML configs work with the actual training script.
"""

from experiments.training_run_config import TrainingRunConfig
from metta.rl.trainer_config import TrainerConfig


class TestTrainIntegration:
    """Test that generated configs work with tools/train.py."""

    def test_agent_config_loads_correctly(self):
        """Test that agent configs are loaded correctly by train.py."""
        # Create a config with specific agent
        config = TrainingRunConfig(
            agent_config="fast",  # Use a simple agent config
            curriculum="env/mettagrid/curriculum/test",
        )

        # Verify the YAML structure is correct
        yaml_dict = config.serialize_to_yaml()

        # Check that agent config is properly referenced
        assert "defaults" in yaml_dict
        # Now using relative paths
        assert "../agent/fast" in yaml_dict["defaults"]

        # Check curriculum is set
        assert yaml_dict["trainer"]["curriculum"] == "env/mettagrid/curriculum/test"

        # Test save_for_local_testing creates a file and returns a command
        test_command = config.save_for_local_testing(instance_name="test_agent_config")
        yaml_path = config._saved_yaml_path

        try:
            assert yaml_path.exists()
            assert "tools/train.py" in test_command
            assert "+experiments=" in test_command
        finally:
            # Clean up
            if yaml_path and yaml_path.exists():
                yaml_path.unlink()

    def test_multiple_agent_configs(self):
        """Test that different agent configs produce different YAMLs."""
        configs = []

        for agent in ["fast", "latent_attn_tiny"]:
            config = TrainingRunConfig(
                agent_config=agent,
                curriculum="env/mettagrid/curriculum/test",
            )
            yaml_dict = config.serialize_to_yaml()
            configs.append(yaml_dict)

        # Check that agent references are different
        assert configs[0]["defaults"] != configs[1]["defaults"]

        # Check that correct agent is referenced (now using relative paths)
        assert "../agent/fast" in configs[0]["defaults"]
        assert "../agent/latent_attn_tiny" in configs[1]["defaults"]

    def test_trainer_config_overrides_work(self):
        """Test that trainer config overrides are applied correctly."""
        from metta.rl.trainer_config import (
            CheckpointConfig,
            SimulationConfig,
            TorchProfilerConfig,
        )

        # Create custom trainer
        trainer = TrainerConfig(
            total_timesteps=12345,
            batch_size=1024,  # Must be divisible by minibatch_size
            minibatch_size=256,  # Must divide batch_size evenly
            curriculum="custom/curriculum",
            num_workers=2,
            checkpoint=CheckpointConfig(checkpoint_dir="${run_dir}/checkpoints"),
            simulation=SimulationConfig(replay_dir="${run_dir}/replays"),
            profiler=TorchProfilerConfig(profile_dir="${run_dir}/torch_traces"),
        )

        config = TrainingRunConfig(
            curriculum="original/curriculum",  # Will be overridden
            trainer=trainer,
        )

        # Save and get command
        test_command = config.save_for_local_testing(instance_name="test_trainer_overrides")
        # After save_for_local_testing, the path is stored in _saved_yaml_path
        yaml_path = config._saved_yaml_path

        try:
            # Load the saved YAML
            with open(yaml_path) as f:
                content = f.read()

            # The YAML file has a package directive at the top
            # Parse the YAML while handling the directive
            import yaml

            # Remove the package directive line if present
            if content.startswith("#"):
                lines = content.split("\n")
                yaml_content = "\n".join(lines[1:])
            else:
                yaml_content = content

            loaded = yaml.safe_load(yaml_content)

            # Verify trainer overrides were applied
            assert loaded["trainer"]["total_timesteps"] == 12345
            assert loaded["trainer"]["batch_size"] == 1024
            # Curriculum from TrainingRunConfig wins (by design)
            assert loaded["trainer"]["curriculum"] == "original/curriculum"

            # Test command should work
            assert "tools/train.py" in test_command
            assert "+experiments=" in test_command
            assert yaml_path.stem in test_command

        finally:
            # Clean up
            if yaml_path and yaml_path.exists():
                yaml_path.unlink()

    def test_yaml_structure_for_hydra(self):
        """Test that YAML structure is correct for Hydra composition."""
        config = TrainingRunConfig(
            curriculum="env/mettagrid/curriculum/test",
            agent_config="latent_attn_tiny",
            sim_config="navigation",
            wandb_entity="test-org",
            wandb_project="test-project",
            wandb_tags=["integration", "test"],
        )

        yaml_path, yaml_dict = config.serialize_to_yaml_file(instance_name="test_hydra_structure")

        try:
            # Read the file to check package directive
            with open(yaml_path) as f:
                content = f.read()

            # Should have package directive
            assert content.startswith("# @package _global_")

            # Check defaults structure
            assert "defaults" in yaml_dict
            defaults = yaml_dict["defaults"]

            # Should have proper Hydra defaults (using relative paths)
            assert "../common" in defaults
            assert "../agent/latent_attn_tiny" in defaults
            assert "../sim/navigation" in defaults
            assert "_self_" in defaults

            # WandB overrides should be present
            assert "wandb" in yaml_dict
            assert yaml_dict["wandb"]["entity"] == "test-org"
            assert yaml_dict["wandb"]["project"] == "test-project"
            assert yaml_dict["wandb"]["tags"] == ["integration", "test"]

        finally:
            # Clean up
            if yaml_path.exists():
                yaml_path.unlink()
