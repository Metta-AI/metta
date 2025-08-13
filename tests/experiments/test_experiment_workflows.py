"""Test experiment workflows end-to-end.

Focus on testing the actual workflows users will use, not internal implementation details.
"""

from unittest.mock import Mock, patch

import yaml

from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig
from experiments.skypilot_job_config import SkypilotJobConfig
from experiments.training_run_config import TrainingRunConfig


class TestExperimentWorkflows:
    """Test the main workflows users will perform with experiments."""

    def test_preview_workflow_generates_yaml_without_launching(self):
        """Test that --no-launch shows what will be sent without launching.

        User workflow: Preview what will be sent to Skypilot before launching.
        """
        # User creates config with launch=False to preview
        config = SingleJobExperimentConfig(
            name="test_preview",
            launch=False,
            total_timesteps=50000,
            learning_rate=0.001,
        )

        experiment = SingleJobExperiment(config)

        # Load or launch should create configs but not launch
        experiment.load_or_launch_training_jobs()

        # Should have generated configs
        assert len(experiment._training_job_configs) == 1
        job_config = experiment._training_job_configs[0]

        # Verify YAML can be generated and contains expected values
        yaml_path, yaml_dict = job_config.training.serialize_to_yaml_file()

        try:
            # Verify file exists and is valid YAML
            assert yaml_path.exists()
            with open(yaml_path) as f:
                loaded = yaml.safe_load(f)

            # Check that overrides were applied
            assert loaded["trainer"]["total_timesteps"] == 50000
            assert loaded["trainer"]["optimizer"]["learning_rate"] == 0.001

            # Check required fields for tools/train.py
            assert "defaults" in loaded
            assert "trainer" in loaded
            assert "wandb" in loaded
            assert "seed" in loaded

        finally:
            # Clean up
            if yaml_path.exists():
                yaml_path.unlink()

        # Should not have launched anything
        assert len(experiment.launched_training_jobs) == 0

    @patch("experiments.skypilot_service.subprocess.Popen")
    def test_launch_workflow_sends_yaml_to_skypilot(self, mock_popen):
        """Test that launch workflow correctly sends YAML via file_mounts.

        User workflow: Launch experiment to Skypilot with config.
        """
        from experiments.skypilot_service import get_skypilot_service

        # Mock subprocess for launch
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [
            b"Job ID: sky-test-123\n",
            b"",
        ]
        mock_process.stderr.read.return_value = b""
        mock_popen.return_value = mock_process

        # User creates and launches experiment
        config = SingleJobExperimentConfig(
            name="test_launch",
            launch=True,
            skypilot=SkypilotJobConfig(
                gpus=2,
                nodes=1,
                spot=False,
                git_check=False,  # Skip for test
            ),
            training=TrainingRunConfig(
                curriculum="test/curriculum",
                wandb_tags=["test"],
            ),
        )

        experiment = SingleJobExperiment(config)

        with patch.object(get_skypilot_service(), "run_preflight_checks", return_value=None):
            jobs = experiment.launch_training_jobs()

        # Should have launched one job
        assert len(jobs) == 1
        assert jobs[0].launched is True
        assert jobs[0].job_id == "sky-test-123"

        # Verify launch command includes config file
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]

        # Should have --config-file in the command
        assert "--config-file" in call_args
        config_file_idx = call_args.index("--config-file")
        config_file_path = call_args[config_file_idx + 1]

        # Config file should be a YAML file
        assert config_file_path.endswith(".yaml")

    def test_experiment_with_trainer_overrides_workflow(self):
        """Test creating experiment with specific trainer settings.

        User workflow: Create experiment with custom training hyperparameters.
        """
        from metta.rl.trainer_config import OptimizerConfig, PPOConfig, TrainerConfig

        # User provides custom trainer config
        trainer = TrainerConfig(
            total_timesteps=100000,
            batch_size=512,
            optimizer=OptimizerConfig(
                type="muon",
                learning_rate=0.0001,
            ),
            ppo=PPOConfig(
                clip_coef=0.3,
                ent_coef=0.05,
            ),
            curriculum="custom/curriculum",
            torch_profiler=None,  # Disable profiler
        )

        config = SingleJobExperimentConfig(
            name="custom_trainer",
            launch=False,
            training=TrainingRunConfig(
                curriculum="override/curriculum",  # Will be overridden by trainer
                trainer=trainer,
            ),
        )

        experiment = SingleJobExperiment(config)
        experiment.load_or_launch_training_jobs()

        # Get the generated YAML
        job_config = experiment._training_job_configs[0]
        yaml_dict = job_config.training.serialize_to_yaml()

        # Verify trainer settings are preserved
        assert yaml_dict["trainer"]["total_timesteps"] == 100000
        assert yaml_dict["trainer"]["batch_size"] == 512
        assert yaml_dict["trainer"]["optimizer"]["type"] == "muon"
        assert yaml_dict["trainer"]["optimizer"]["learning_rate"] == 0.0001
        assert yaml_dict["trainer"]["ppo"]["clip_coef"] == 0.3

        # Curriculum should come from trainer, not training config
        assert yaml_dict["trainer"]["curriculum"] == "custom/curriculum"

    @patch("experiments.experiment.get_skypilot_service")
    def test_load_previous_jobs_workflow(self, mock_get_service):
        """Test loading existing Skypilot jobs for analysis.

        User workflow: Load previously launched experiments for monitoring.
        """
        # Mock service to return job info
        mock_service = Mock()
        mock_service.get_wandb_run_name_from_sky_job = Mock(return_value="previous.run.123")
        mock_get_service.return_value = mock_service

        # User loads previous jobs by ID
        config = SingleJobExperimentConfig(
            name="load_test",
            launch=False,
            previous_job_ids=["sky-old-job-1", "sky-old-job-2"],
        )

        experiment = SingleJobExperiment(config)
        jobs = experiment.load_training_jobs()

        # Should have loaded both jobs
        assert len(jobs) == 2
        assert all(job.launched for job in jobs)
        assert all(job.success for job in jobs)
        assert jobs[0].job_id == "sky-old-job-1"
        assert jobs[1].job_id == "sky-old-job-2"

        # Service should have been called to get run names
        assert mock_service.get_wandb_run_name_from_sky_job.call_count == 2


class TestMultipleJobWorkflow:
    """Test workflows involving multiple training jobs."""

    def test_sweep_experiment_creates_multiple_configs(self):
        """Test that sweep experiments generate multiple job configs.

        User workflow: Launch hyperparameter sweep with multiple settings.
        """
        # This would be implemented when SweepExperiment is added
        # For now, test that the infrastructure supports multiple jobs

        config = SingleJobExperimentConfig(
            name="sweep_base",
            launch=False,
        )

        experiment = SingleJobExperiment(config)

        # Manually add multiple configs to simulate sweep
        from experiments.training_job import TrainingJobConfig

        # Clear and add multiple
        experiment._training_job_configs = []
        for lr in [0.0001, 0.0003, 0.001]:
            training = TrainingRunConfig(
                curriculum="sweep/curriculum",
            )
            training.trainer = training.get_trainer_config()
            training.trainer.optimizer.learning_rate = lr

            job_config = TrainingJobConfig(
                skypilot=config.skypilot,
                training=training,
            )
            experiment._training_job_configs.append(job_config)

        # Should support multiple configs
        assert len(experiment._training_job_configs) == 3

        # Each should have different learning rate
        lrs = [cfg.training.trainer.optimizer.learning_rate for cfg in experiment._training_job_configs]
        assert lrs == [0.0001, 0.0003, 0.001]
