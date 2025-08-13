"""Integration tests with tools/train.py.

These tests verify that our generated YAML configs work with the actual training script.
"""

import subprocess

import yaml

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

        # Save to temp location
        _ = config.save_for_local_testing()  # Creates the file
        yaml_path = config._saved_yaml_path

        try:
            # Try to load the config with hydra (dry run)
            # This tests that the YAML structure is correct for hydra
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "./tools/train.py",
                    f"--config-path={yaml_path.parent}",
                    f"--config-name={yaml_path.stem}",
                    "--cfg",
                    "job",  # Just show config, don't run
                    "hydra.mode=MULTIRUN",
                    "hydra.dry=true",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Check that command succeeded
            assert result.returncode == 0, f"Config validation failed: {result.stderr}"

            # Check that agent config was loaded
            assert "agent:" in result.stdout or "fast" in result.stdout

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

        # Check that correct agent is referenced
        assert any("agent: fast" in d for d in configs[0]["defaults"])
        assert any("agent: latent_attn_tiny" in d for d in configs[1]["defaults"])

    def test_trainer_config_overrides_work(self):
        """Test that trainer config overrides are applied correctly."""
        # Create custom trainer
        trainer = TrainerConfig(
            total_timesteps=12345,
            batch_size=999,
            curriculum="custom/curriculum",
        )

        config = TrainingRunConfig(
            curriculum="original/curriculum",  # Will be overridden
            trainer=trainer,
        )

        # Save and get command
        test_command = config.save_for_local_testing()
        yaml_path = config._saved_yaml_path

        try:
            # Load the saved YAML
            with open(yaml_path) as f:
                # Skip the package directive
                lines = f.readlines()
                yaml_content = "".join(lines[1:])  # Skip first line
                loaded = yaml.safe_load(yaml_content)

            # Verify trainer overrides were applied
            assert loaded["trainer"]["total_timesteps"] == 12345
            assert loaded["trainer"]["batch_size"] == 999
            assert loaded["trainer"]["curriculum"] == "custom/curriculum"

            # Test command should work
            assert "tools/train.py" in test_command
            assert f"--config-path={yaml_path.parent}" in test_command
            assert f"--config-name={yaml_path.stem}" in test_command

        finally:
            # Clean up
            if yaml_path and yaml_path.exists():
                yaml_path.unlink()

    def test_yaml_structure_for_hydra(self):
        """Test that YAML structure is correct for Hydra composition."""
        config = TrainingRunConfig(
            agent_config="latent_attn_tiny",
            sim_config="navigation",
            wandb_entity="test-org",
            wandb_project="test-project",
            wandb_tags=["integration", "test"],
        )

        yaml_path, yaml_dict = config.serialize_to_yaml_file()

        try:
            # Read the file to check package directive
            with open(yaml_path) as f:
                content = f.read()

            # Should have package directive
            assert content.startswith("# @package _global_")

            # Check defaults structure
            assert "defaults" in yaml_dict
            defaults = yaml_dict["defaults"]

            # Should have proper Hydra defaults
            assert "common" in defaults
            assert any("agent: latent_attn_tiny" in d for d in defaults)
            assert any("sim: navigation" in d for d in defaults)
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
