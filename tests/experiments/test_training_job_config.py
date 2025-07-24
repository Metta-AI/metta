"""Tests for TrainingJobConfig functionality."""

import pytest
from experiments import TrainingJob, TrainingJobConfig


class TestTrainingJobConfig:
    """Test TrainingJobConfig integration."""
    
    def test_training_job_config_creation(self):
        """Test creating a TrainingJobConfig."""
        config = TrainingJobConfig(
            curriculum="env/mettagrid/curriculum/test",
            gpus=4,
            nodes=2,
            no_spot=True,
            wandb_tags=["test", "experiment"],
            additional_args=[
                "trainer.optimizer.learning_rate=0.001",
                "trainer.optimizer.type=adam",
                "trainer.batch_size=32"
            ]
        )
        
        assert config.curriculum == "env/mettagrid/curriculum/test"
        assert config.gpus == 4
        assert config.nodes == 2
        assert config.no_spot is True
        assert config.wandb_tags == ["test", "experiment"]
        assert len(config.additional_args) == 3
        assert "trainer.optimizer.learning_rate=0.001" in config.additional_args
    
    def test_training_job_with_config(self):
        """Test TrainingJob with TrainingJobConfig."""
        config = TrainingJobConfig(
            curriculum="test/curriculum",
            gpus=2,
            additional_args=[
                "trainer.optimizer.learning_rate=0.0045",
                "trainer.optimizer.type=muon"
            ]
        )
        
        job = TrainingJob(
            wandb_run_name="test.run.1",
            skypilot_job_id="sky-test-123",
            config=config,
            notes="Test job with config"
        )
        
        assert job.wandb_run_name == "test.run.1"
        assert job.skypilot_job_id == "sky-test-123"
        assert isinstance(job.config, TrainingJobConfig)
        assert job.config.curriculum == "test/curriculum"
        assert job.config.get_arg_value("trainer.optimizer.learning_rate") == "0.0045"
        assert job.config.get_arg_value("trainer.optimizer.type") == "muon"
    
    def test_training_job_serialization_with_config(self):
        """Test serializing and deserializing TrainingJob with config."""
        config = TrainingJobConfig(
            curriculum="test/curriculum",
            gpus=8,
            additional_args=["trainer.optimizer.learning_rate=0.001"]
        )
        
        original_job = TrainingJob(
            wandb_run_name="test.serialize.1",
            skypilot_job_id="sky-serialize-123",
            config=config
        )
        
        # Serialize to dict
        job_dict = original_job.to_dict()
        
        # Check dict structure
        assert job_dict["wandb_run_name"] == "test.serialize.1"
        assert job_dict["config"]["curriculum"] == "test/curriculum"
        assert job_dict["config"]["gpus"] == 8
        assert job_dict["config"]["additional_args"] == ["trainer.optimizer.learning_rate=0.001"]
        
        # Deserialize back
        restored_job = TrainingJob.from_dict(job_dict)
        
        assert restored_job.wandb_run_name == original_job.wandb_run_name
        assert restored_job.skypilot_job_id == original_job.skypilot_job_id
        assert isinstance(restored_job.config, TrainingJobConfig)
        assert restored_job.config.curriculum == "test/curriculum"
        assert restored_job.config.gpus == 8
        assert restored_job.config.additional_args == ["trainer.optimizer.learning_rate=0.001"]
    
    def test_get_arg_value(self):
        """Test get_arg_value helper method."""
        config = TrainingJobConfig(
            curriculum="test",
            additional_args=[
                "trainer.optimizer.learning_rate=0.001",
                "trainer.optimizer.type=adam",
                "trainer.batch_size=32",
                "some_flag"  # Arg without value
            ]
        )
        
        assert config.get_arg_value("trainer.optimizer.learning_rate") == "0.001"
        assert config.get_arg_value("trainer.optimizer.type") == "adam"
        assert config.get_arg_value("trainer.batch_size") == "32"
        assert config.get_arg_value("nonexistent") is None
        assert config.get_arg_value("some_flag") is None