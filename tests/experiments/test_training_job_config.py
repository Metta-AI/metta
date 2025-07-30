"""Tests for TrainingJobConfig functionality."""

from experiments import TrainingJob, TrainingJobConfig


class TestTrainingJobConfig:
    """Test TrainingJobConfig integration."""

    def test_training_job_config_creation(self):
        """Test creating a TrainingJobConfig."""
        config = TrainingJobConfig(
            curriculum="env/mettagrid/curriculum/test",
            gpus=4,
            nodes=2,
            spot=False,  # equivalent to no_spot=True
            wandb_tags=["test", "experiment"],
            additional_args=[
                "trainer.optimizer.learning_rate=0.001",
                "trainer.optimizer.type=adam",
                "trainer.batch_size=32",
            ],
        )

        assert config.curriculum == "env/mettagrid/curriculum/test"
        assert config.gpus == 4
        assert config.nodes == 2
        assert config.spot is False
        assert config.wandb_tags == ["test", "experiment"]
        assert len(config.additional_args) == 3
        assert "trainer.optimizer.learning_rate=0.001" in config.additional_args

    def test_training_job_with_config(self):
        """Test TrainingJob with TrainingJobConfig."""
        config = TrainingJobConfig(
            curriculum="test/curriculum",
            gpus=2,
            additional_args=["trainer.optimizer.learning_rate=0.0045", "trainer.optimizer.type=muon"],
        )

        job = TrainingJob(name="test.run.1", config=config)
        job.job_id = "sky-test-123"

        assert job.name == "test.run.1"
        assert job.job_id == "sky-test-123"
        assert isinstance(job.config, TrainingJobConfig)
        assert job.config.curriculum == "test/curriculum"
        assert job.config.get_arg_value("trainer.optimizer.learning_rate") == "0.0045"
        assert job.config.get_arg_value("trainer.optimizer.type") == "muon"

    def test_get_arg_value(self):
        """Test get_arg_value helper method."""
        config = TrainingJobConfig(
            curriculum="test",
            additional_args=[
                "trainer.optimizer.learning_rate=0.001",
                "trainer.optimizer.type=adam",
                "trainer.batch_size=32",
                "some_flag",  # Arg without value
            ],
        )

        assert config.get_arg_value("trainer.optimizer.learning_rate") == "0.001"
        assert config.get_arg_value("trainer.optimizer.type") == "adam"
        assert config.get_arg_value("trainer.batch_size") == "32"
        assert config.get_arg_value("nonexistent") is None
        assert config.get_arg_value("some_flag") is None
