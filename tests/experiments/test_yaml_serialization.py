"""Tests for YAML serialization and config transfer."""

from unittest.mock import Mock, patch

import yaml

from experiments.experiment import SingleJobExperimentConfig
from experiments.skypilot_job_config import SkypilotJobConfig
from experiments.training_job import TrainingJob, TrainingJobConfig
from experiments.training_run_config import TrainingRunConfig


def test_training_run_config_serialization():
    """Test that TrainingRunConfig serializes to valid YAML."""
    config = TrainingRunConfig(
        curriculum="test/curriculum",
        agent_config="fast",
        wandb_entity="test-entity",
        wandb_tags=["test", "yaml"],
    )

    # Serialize to dict
    config_dict = config.serialize_to_yaml()

    # Check structure
    assert "defaults" in config_dict
    assert "trainer" in config_dict
    assert config_dict["seed"] == 1
    assert "agent: fast" in config_dict["defaults"]

    # Check wandb overrides
    assert config_dict["wandb"]["entity"] == "test-entity"
    assert config_dict["wandb"]["tags"] == ["test", "yaml"]


def test_training_run_config_with_trainer():
    """Test serialization with trainer config overrides."""
    from metta.rl.trainer_config import OptimizerConfig, TrainerConfig

    trainer = TrainerConfig(
        total_timesteps=999,
        batch_size=256,
        num_workers=2,
        optimizer=OptimizerConfig(learning_rate=0.001),
        curriculum="test/curriculum",
    )

    config = TrainingRunConfig(
        curriculum="test/curriculum",
        trainer=trainer,
    )

    config_dict = config.serialize_to_yaml()

    # Check trainer overrides are included
    assert config_dict["trainer"]["total_timesteps"] == 999
    assert config_dict["trainer"]["batch_size"] == 256
    assert config_dict["trainer"]["optimizer"]["learning_rate"] == 0.001


def test_yaml_file_creation():
    """Test that YAML file is created correctly."""
    config = TrainingRunConfig(
        curriculum="test/curriculum",
        agent_config="latent_attn_tiny",
    )

    yaml_path, full_config = config.serialize_to_yaml_file()

    try:
        # Check file exists
        assert yaml_path.exists()
        assert yaml_path.suffix == ".yaml"

        # Check it's valid YAML
        with open(yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded == full_config
        assert loaded["defaults"][1] == "agent: latent_attn_tiny"

    finally:
        # Clean up
        if yaml_path.exists():
            yaml_path.unlink()


def test_single_job_experiment_config_to_training_job():
    """Test conversion from experiment config to training job config."""
    exp_config = SingleJobExperimentConfig(
        name="test_exp",
        total_timesteps=5000,
        batch_size=1024,
        learning_rate=0.0003,
    )

    job_config = exp_config.to_training_job_config()

    # Check structure
    assert isinstance(job_config, TrainingJobConfig)
    assert isinstance(job_config.skypilot, SkypilotJobConfig)
    assert isinstance(job_config.training, TrainingRunConfig)

    # Check trainer overrides were applied
    assert job_config.training.trainer is not None
    assert job_config.training.trainer.total_timesteps == 5000
    assert job_config.training.trainer.batch_size == 1024
    assert job_config.training.trainer.optimizer.learning_rate == 0.0003


@patch("experiments.skypilot_service.subprocess.Popen")
def test_skypilot_service_yaml_transfer(mock_popen):
    """Test that SkypilotService correctly sets up YAML file transfer."""
    from experiments.skypilot_service import SkypilotService

    # Mock the subprocess
    mock_process = Mock()
    mock_process.poll.return_value = 0
    mock_process.returncode = 0
    mock_process.stdout.readline.side_effect = [
        b"Launching job...\n",
        b"Job ID: sky-2024-01-01-12-00-00-abc123\n",
        b"",  # EOF
    ]
    mock_process.stderr.read.return_value = b""
    mock_popen.return_value = mock_process

    service = SkypilotService()

    # Create a training job with config
    job_config = TrainingJobConfig()
    job = TrainingJob(name="test_job", config=job_config)

    with patch.object(service, "run_preflight_checks", return_value=None):
        service.launch_training("test_run", job)

    # Check that launch was called with correct arguments
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args[0][0]  # Get the command list

    # Should have --config-file argument
    assert any("--config-file" in arg for arg in call_args)

    # Find the config file path
    config_file_idx = call_args.index("--config-file")
    config_file_path = call_args[config_file_idx + 1]

    # Should be a valid path to a YAML file
    assert config_file_path.endswith(".yaml")


def test_launch_false_shows_preview():
    """Test that launch=False shows preview without launching."""
    from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig

    config = SingleJobExperimentConfig(
        name="test_preview",
        launch=False,  # Don't actually launch
        total_timesteps=999,
    )

    experiment = SingleJobExperiment(config)

    # This should create configs but not launch
    experiment.load_or_launch_training_jobs()

    # Should have configs but no launched jobs
    assert len(experiment._training_job_configs) == 1
    assert len(experiment.launched_training_jobs) == 0

    # Verify the config was created correctly
    job_config = experiment._training_job_configs[0]
    assert job_config.training.trainer.total_timesteps == 999
