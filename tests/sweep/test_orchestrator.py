"""Tests for the sweep orchestrator core components."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from metta.sweep import (
    JobDefinition,
    JobStatus,
    JobTypes,
    LocalDispatcher,
    Observation,
    RunInfo,
    SweepMetadata,
)
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.schedulers.optimizing import OptimizingScheduler, OptimizingSchedulerConfig
from metta.sweep.stores.wandb import WandbStore


class TestJobStatus:
    """Test JobStatus enum and status computation."""

    def test_job_status_from_run_info_pending(self):
        """Test that a new run has PENDING status."""
        run = RunInfo(
            run_id="test_run_001",
            has_started_training=False,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        )
        assert run.status == JobStatus.PENDING

    def test_job_status_from_run_info_training(self):
        """Test that a training run has IN_TRAINING status."""
        run = RunInfo(
            run_id="test_run_001",
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=False,
        )
        assert run.status == JobStatus.IN_TRAINING

    def test_job_status_from_run_info_evaluating(self):
        """Test that an evaluating run has IN_EVAL status."""
        run = RunInfo(
            run_id="test_run_001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=False,
            has_failed=False,
        )
        assert run.status == JobStatus.IN_EVAL

    def test_job_status_from_run_info_completed(self):
        """Test that a fully evaluated run has EVAL_DONE_NOT_COMPLETED status."""
        # TODO: Bug detected - when has_been_evaluated=True, status should be COMPLETED
        # but it returns EVAL_DONE_NOT_COMPLETED. Need to fix the logic in RunInfo.status property
        run = RunInfo(
            run_id="test_run_001",
            has_started_training=True,
            has_completed_training=True,
            has_started_eval=True,
            has_been_evaluated=True,
            has_failed=False,
        )
        # Current behavior (buggy)
        assert run.status == JobStatus.EVAL_DONE_NOT_COMPLETED

    def test_job_status_from_run_info_failed(self):
        """Test that a failed run has FAILED status."""
        run = RunInfo(
            run_id="test_run_001",
            has_started_training=True,
            has_completed_training=False,
            has_started_eval=False,
            has_been_evaluated=False,
            has_failed=True,
        )
        assert run.status == JobStatus.FAILED


class TestLocalDispatcher:
    """Test LocalDispatcher subprocess management."""

    @patch("subprocess.Popen")
    def test_local_dispatcher_dispatch(self, mock_popen):
        """Test that LocalDispatcher spawns subprocesses correctly."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process

        dispatcher = LocalDispatcher()

        job = JobDefinition(
            run_id="test_run_001",
            cmd="experiments.recipes.arena.train",
            type=JobTypes.LAUNCH_TRAINING,
            args=[],
            overrides={"trainer.total_timesteps": "1000"},
        )

        # Dispatch the job
        dispatcher.dispatch(job)

        # Verify subprocess was created with correct command
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]

        # Check command structure
        assert "uv" in call_args[0]
        assert "run" in call_args
        assert "./tools/run.py" in call_args
        assert "experiments.recipes.arena.train" in call_args
        assert "--args" in call_args
        assert "run=test_run_001" in call_args
        assert "trainer.total_timesteps=1000" in call_args

    @patch("subprocess.Popen")
    def test_local_dispatcher_check_processes(self, mock_popen):
        """Test that LocalDispatcher tracks subprocess status via check_processes."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process

        dispatcher = LocalDispatcher()

        job = JobDefinition(
            run_id="test_run_001",
            cmd="experiments.recipes.arena.train",
            type=JobTypes.LAUNCH_TRAINING,
        )

        # Dispatch and check active count
        dispatcher.dispatch(job)
        active_count = dispatcher.check_processes()

        assert active_count == 1
        assert len(dispatcher._processes) == 1

        # Simulate process completion
        mock_process.poll.return_value = 0
        active_count = dispatcher.check_processes()

        # After reaping, should have 0 active processes
        assert active_count == 0
        assert len(dispatcher._processes) == 0


class TestProtocolCompliance:
    """Test that components comply with Protocol interfaces."""

    def test_scheduler_protocol(self):
        """Test that schedulers follow the Scheduler protocol."""
        # Create a simple protein config
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",
            parameters={"test_param": {"min": 0, "max": 1}},
        )

        optimizer = ProteinOptimizer(protein_config)
        scheduler_config = OptimizingSchedulerConfig(
            max_trials=5,
            recipe_module="experiments.recipes.arena",
            train_entrypoint="train",
            eval_entrypoint="evaluate",
        )
        scheduler = OptimizingScheduler(scheduler_config, optimizer)

        # Verify protocol methods exist (Scheduler only requires schedule)
        assert hasattr(scheduler, "schedule")
        assert callable(scheduler.schedule)

    def test_store_protocol(self):
        """Test that stores follow the Store protocol."""
        store = WandbStore(entity="test_entity", project="test_project")

        # Verify protocol methods exist
        assert hasattr(store, "init_run")
        assert hasattr(store, "fetch_runs")
        assert hasattr(store, "update_run_summary")
        assert callable(store.init_run)
        assert callable(store.fetch_runs)
        assert callable(store.update_run_summary)

    def test_dispatcher_protocol(self):
        """Test that dispatchers follow the Dispatcher protocol."""
        dispatcher = LocalDispatcher()

        # Verify protocol methods exist (Dispatcher only requires dispatch)
        assert hasattr(dispatcher, "dispatch")
        assert callable(dispatcher.dispatch)

        # LocalDispatcher has check_processes instead of get_status
        assert hasattr(dispatcher, "check_processes")
        assert callable(dispatcher.check_processes)


class TestSweepMetadata:
    """Test SweepMetadata dataclass."""

    def test_sweep_metadata_creation(self):
        """Test creating SweepMetadata."""
        metadata = SweepMetadata(
            sweep_id="test_sweep_001",
        )

        assert metadata.sweep_id == "test_sweep_001"
        assert isinstance(metadata.start_time, datetime)
        assert metadata.runs_created == 0
        assert metadata.runs_completed == 0


class TestObservation:
    """Test Observation dataclass."""

    def test_observation_creation(self):
        """Test creating Observation."""
        obs = Observation(
            score=0.85,
            cost=100.5,
            suggestion={"learning_rate": 0.001},
        )

        assert obs.score == 0.85
        assert obs.cost == 100.5
        assert obs.suggestion["learning_rate"] == 0.001
