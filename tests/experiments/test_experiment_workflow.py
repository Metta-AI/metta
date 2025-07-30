"""Tests for the experiment workflow with the new API."""

from unittest.mock import Mock, patch

from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig
from experiments.notebooks.notebook import NotebookConfig, AnalysisConfig
from experiments.training_job import TrainingJob, TrainingJobConfig


class TestExperimentWorkflow:
    """Test the experiment workflow including job launching."""

    @patch("experiments.skypilot_service.get_skypilot_service")
    def test_single_job_experiment_launch(self, mock_get_service):
        """Test launching a single job experiment."""
        # Mock the skypilot service with required attributes
        from experiments.skypilot_service import LaunchResult, SkypilotService

        mock_service = Mock(spec=SkypilotService)
        mock_service._tracked_jobs = {}
        mock_service._jobs_by_name = {}

        # Create a side effect that mimics what the real service does
        def mock_launch(**kwargs):
            result = LaunchResult(success=True, job_id="sky-test-123")
            # Update the job if provided (mimicking real service behavior)
            if "training_job" in kwargs and kwargs["training_job"]:
                job = kwargs["training_job"]
                job.job_id = result.job_id
                job.launched = True
                job.success = result.success
            return result

        mock_service.launch_training.side_effect = mock_launch
        mock_get_service.return_value = mock_service

        # Create experiment config
        config = SingleJobExperimentConfig(
            name="test_experiment",
            curriculum="env/test/curriculum",
            gpus=2,
            nodes=1,
            spot=False,
            git_check=False,  # Changed from skip_git_check=True
            wandb_tags=["test", "unit-test"],
        )

        # Create and run experiment
        experiment = SingleJobExperiment(config)
        jobs = experiment.launch_training_jobs()

        # Verify results
        assert len(jobs) == 1
        job = jobs[0]
        assert job.name == "test_experiment_job_0"
        assert job.launched is True
        assert job.success is True
        assert job.job_id == "sky-test-123"

        # Verify service was called correctly
        mock_service.launch_training.assert_called_once()
        call_kwargs = mock_service.launch_training.call_args.kwargs
        assert call_kwargs["run_name"] == "test_experiment_job_0"
        assert call_kwargs["curriculum"] == "env/test/curriculum"
        assert call_kwargs["gpus"] == 2
        assert call_kwargs["nodes"] == 1
        assert call_kwargs["spot"] is False
        assert call_kwargs["skip_git_check"] is True  # Service still expects skip_git_check
        assert call_kwargs["wandb_tags"] == ["test", "unit-test"]
        assert call_kwargs["additional_args"] is None
        assert call_kwargs["training_job"] == jobs[0]

    @patch("experiments.skypilot_service.get_skypilot_service")
    def test_failed_job_launch(self, mock_get_service):
        """Test handling of failed job launch."""
        # Mock failed launch
        from experiments.skypilot_service import LaunchResult, SkypilotService

        mock_service = Mock(spec=SkypilotService)
        mock_service._tracked_jobs = {}
        mock_service._jobs_by_name = {}

        # Create a side effect for failed launch
        def mock_failed_launch(**kwargs):
            result = LaunchResult(success=False, job_id=None)
            # Update the job even on failure
            if "training_job" in kwargs and kwargs["training_job"]:
                job = kwargs["training_job"]
                job.job_id = result.job_id
                job.launched = True
                job.success = result.success
            return result

        mock_service.launch_training.side_effect = mock_failed_launch
        mock_get_service.return_value = mock_service

        config = SingleJobExperimentConfig(
            name="failed_experiment", curriculum="env/test/curriculum", git_check=False
        )

        experiment = SingleJobExperiment(config)
        jobs = experiment.launch_training_jobs()

        # Job should be marked as launched but not successful
        assert len(jobs) == 1
        job = jobs[0]
        assert job.launched is True
        assert job.success is False
        assert job.job_id is None

    @patch("experiments.experiment.get_skypilot_service")
    def test_experiment_with_previous_jobs(self, mock_get_service):
        """Test loading experiment with previous job IDs."""
        # Mock the skypilot service
        mock_service = Mock()
        mock_service.get_wandb_run_name_from_sky_job = Mock(return_value="user.loaded.run")
        mock_get_service.return_value = mock_service

        config = SingleJobExperimentConfig(
            name="loaded_experiment", launch=False, previous_job_ids=["sky-previous-123"]
        )

        experiment = SingleJobExperiment(config)
        jobs = experiment.load_training_jobs()

        assert len(jobs) == 1
        job = jobs[0]
        assert job.launched is True
        assert job.success is True
        assert job.job_id == "sky-previous-123"
        assert job.name == "user.loaded.run"

    @patch("experiments.experiment.write_notebook")
    def test_notebook_generation(self, mock_write_notebook):
        """Test that notebook generation is called with correct parameters."""
        mock_write_notebook.return_value = "/tmp/test_notebook.ipynb"

        config = SingleJobExperimentConfig(
            name="notebook_test", launch=False, previous_job_ids=["sky-123"], output_dir="/tmp/notebooks"
        )

        experiment = SingleJobExperiment(config)
        experiment.unlaunched_training_jobs = []
        experiment.launched_training_jobs = [TrainingJob(name="test.run", config=TrainingJobConfig())]

        experiment.generate_notebook()

        # Just verify the key parameters without being too specific about section order
        mock_write_notebook.assert_called_once()
        call_args = mock_write_notebook.call_args[1]
        assert call_args["user"] == experiment.user
        assert call_args["name"] == "notebook_test"
        assert call_args["launched_jobs"] == experiment.launched_training_jobs
        assert call_args["training_job_configs"] == []
        assert call_args["output_dir"] == "/tmp/notebooks"
        # Don't test exact section order - that's an implementation detail

    def test_training_job_config_additional_args(self):
        """Test that additional args are properly passed to launch command."""
        job = TrainingJob(
            name="test_job",
            config=TrainingJobConfig(
                curriculum="test", additional_args=["trainer.optimizer.learning_rate=0.001", "trainer.batch_size=32"]
            ),
        )

        # Test the get_arg_value helper
        assert job.config.get_arg_value("trainer.optimizer.learning_rate") == "0.001"
        assert job.config.get_arg_value("trainer.batch_size") == "32"
        assert job.config.get_arg_value("nonexistent") is None

    @patch("experiments.experiment.write_notebook")
    def test_notebook_config_sections(self, mock_write_notebook):
        """Test that NotebookConfig correctly controls which sections are included."""
        mock_write_notebook.return_value = "/tmp/test_notebook.ipynb"

        # Create config with specific notebook sections enabled
        notebook_config = NotebookConfig(
            setup=True,
            state=False,
            launch=True,
            monitor=False,
            analysis=True,
            analysis_config=AnalysisConfig(sps=True),
            replays=False,
            scratch=False,
            export=True,
        )

        config = SingleJobExperimentConfig(
            name="notebook_config_test", launch=False, previous_job_ids=["sky-123"], notebook=notebook_config
        )

        experiment = SingleJobExperiment(config)
        experiment.unlaunched_training_jobs = []
        experiment.launched_training_jobs = []

        experiment.generate_notebook()

        # Verify write_notebook was called with correct sections
        mock_write_notebook.assert_called_once()
        call_args = mock_write_notebook.call_args
        sections = call_args.kwargs["sections"]

        assert sections == ["setup", "launch", "analysis", "export"]
        assert "state" not in sections
        assert "monitor" not in sections
        assert "replays" not in sections
        assert "scratch" not in sections
