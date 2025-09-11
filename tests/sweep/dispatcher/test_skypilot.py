"""Comprehensive tests for SkypilotDispatcher implementation."""

import os
import subprocess
import uuid
from unittest.mock import MagicMock, patch

import pytest

from metta.sweep import Dispatcher, JobDefinition, JobTypes
from metta.sweep.dispatcher.skypilot import SkypilotDispatcher


# Helper to get the expected launch script path
def get_launch_script_path():
    """Get the absolute path to the launch script as SkypilotDispatcher computes it."""
    import metta.sweep.dispatcher.skypilot

    return os.path.abspath(
        os.path.join(
            os.path.dirname(metta.sweep.dispatcher.skypilot.__file__),
            "..",
            "..",
            "..",
            "devops",
            "skypilot",
            "launch.py",
        )
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_popen():
    """Mock subprocess.Popen to avoid actual process launches."""
    with patch("subprocess.Popen") as mock:
        # Create a mock process with a fake PID
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock.return_value = mock_process
        yield mock


@pytest.fixture
def mock_uuid():
    """Mock uuid.uuid4 for deterministic dispatch_ids."""
    with patch("uuid.uuid4") as mock:
        mock.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        yield mock


@pytest.fixture
def basic_training_job():
    """Basic training job with minimal configuration."""
    return JobDefinition(
        run_id="sweep_test_trial_001",
        cmd="experiments.recipes.arena.train",
        type=JobTypes.LAUNCH_TRAINING,
    )


@pytest.fixture
def basic_eval_job():
    """Basic evaluation job."""
    return JobDefinition(
        run_id="sweep_test_trial_001_eval",
        cmd="experiments.recipes.arena.evaluate",
        type=JobTypes.LAUNCH_EVAL,
        metadata={"policy_uri": "file://./checkpoints/policy.pt"},
    )


@pytest.fixture
def complex_job():
    """Complex job with all parameters set."""
    return JobDefinition(
        run_id="sweep_complex_trial_042",
        cmd="experiments.recipes.navigation.train_shaped",
        gpus=4,
        nodes=2,
        args=["positional_arg1", "positional_arg2"],
        overrides={"trainer.batch_size": 256, "trainer.learning_rate": 0.001},
        config={"optimizer.momentum": 0.9, "optimizer.weight_decay": 0.0001},
        type=JobTypes.LAUNCH_TRAINING,
        metadata={"experiment": "navigation", "variant": "shaped"},
    )


# ============================================================================
# Command Construction Tests
# ============================================================================


class TestCommandConstruction:
    """Test command construction for various job configurations."""

    def test_basic_training_job(self, mock_popen, mock_uuid, basic_training_job):
        """Test basic training job command construction."""
        dispatcher = SkypilotDispatcher()
        dispatch_id = dispatcher.dispatch(basic_training_job)

        # Verify dispatch_id is the mocked UUID
        assert dispatch_id == "12345678-1234-5678-1234-567812345678"

        # Verify command construction (JobDefinition defaults to gpus=1)
        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "--gpus=1",
            "experiments.recipes.arena.train",
            "--args",
            "run=sweep_test_trial_001",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )

    def test_eval_job_no_run_id(self, mock_popen, mock_uuid, basic_eval_job):
        """Test that evaluation jobs don't include run_id in args."""
        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(basic_eval_job)

        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "--gpus=1",  # JobDefinition defaults to gpus=1
            "experiments.recipes.arena.evaluate",
            "--args",
            "policy_uri=file://./checkpoints/policy.pt",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )

    def test_job_with_gpus(self, mock_popen, mock_uuid):
        """Test job with GPU configuration."""
        job = JobDefinition(
            run_id="gpu_test", cmd="experiments.recipes.arena.train", gpus=8, type=JobTypes.LAUNCH_TRAINING
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "--gpus=8",
            "experiments.recipes.arena.train",
            "--args",
            "run=gpu_test",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )

    def test_job_with_nodes(self, mock_popen, mock_uuid):
        """Test job with multi-node configuration."""
        job = JobDefinition(
            run_id="node_test", cmd="experiments.recipes.arena.train", nodes=4, type=JobTypes.LAUNCH_TRAINING
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "--gpus=1",  # JobDefinition defaults to gpus=1
            "--nodes=4",
            "experiments.recipes.arena.train",
            "--args",
            "run=node_test",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )

    def test_job_with_gpus_and_nodes(self, mock_popen, mock_uuid):
        """Test job with both GPU and node configuration."""
        job = JobDefinition(
            run_id="distributed_test",
            cmd="experiments.recipes.arena.train",
            gpus=4,
            nodes=2,
            type=JobTypes.LAUNCH_TRAINING,
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "--gpus=4",
            "--nodes=2",
            "experiments.recipes.arena.train",
            "--args",
            "run=distributed_test",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )

    def test_complex_job_all_parameters(self, mock_popen, mock_uuid, complex_job):
        """Test complex job with all parameters set."""
        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(complex_job)

        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "--gpus=4",
            "--nodes=2",
            "experiments.recipes.navigation.train_shaped",
            "positional_arg1",
            "positional_arg2",
            "--args",
            "run=sweep_complex_trial_042",
            "experiment=navigation",
            "variant=shaped",
            "--overrides",
            "trainer.batch_size=256",
            "trainer.learning_rate=0.001",
            "optimizer.momentum=0.9",
            "optimizer.weight_decay=0.0001",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )

    def test_zero_gpus_not_included(self, mock_popen, mock_uuid):
        """Test that gpus=0 doesn't add --gpus flag."""
        job = JobDefinition(
            run_id="no_gpu_test", cmd="experiments.recipes.arena.train", gpus=0, type=JobTypes.LAUNCH_TRAINING
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "experiments.recipes.arena.train",
            "--args",
            "run=no_gpu_test",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )

    def test_single_node_not_included(self, mock_popen, mock_uuid):
        """Test that nodes=1 doesn't add --nodes flag."""
        job = JobDefinition(
            run_id="single_node_test", cmd="experiments.recipes.arena.train", nodes=1, type=JobTypes.LAUNCH_TRAINING
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        expected_cmd = [
            get_launch_script_path(),
            "--no-spot",
            "--gpus=1",  # JobDefinition defaults to gpus=1
            "experiments.recipes.arena.train",
            "--args",
            "run=single_node_test",
        ]

        mock_popen.assert_called_once_with(
            expected_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
        )


# ============================================================================
# Flag Ordering Tests
# ============================================================================


class TestFlagOrdering:
    """Test that flags are placed in the correct order."""

    def test_no_spot_always_first(self, mock_popen, mock_uuid):
        """Verify --no-spot is always the first flag."""
        job = JobDefinition(
            run_id="flag_test",
            cmd="experiments.recipes.arena.train",
            gpus=2,
            nodes=3,
            type=JobTypes.LAUNCH_TRAINING,
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        # Get the actual command
        args, kwargs = mock_popen.call_args
        cmd = args[0]

        # Check that --no-spot is the first flag (after the script)
        assert cmd[0] == get_launch_script_path()
        assert cmd[1] == "--no-spot"

    def test_gpus_before_nodes(self, mock_popen, mock_uuid):
        """Verify --gpus comes before --nodes when both present."""
        job = JobDefinition(
            run_id="ordering_test",
            cmd="experiments.recipes.arena.train",
            gpus=4,
            nodes=2,
            type=JobTypes.LAUNCH_TRAINING,
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        args, kwargs = mock_popen.call_args
        cmd = args[0]

        # Find positions of flags
        gpus_idx = next(i for i, arg in enumerate(cmd) if arg.startswith("--gpus="))
        nodes_idx = next(i for i, arg in enumerate(cmd) if arg.startswith("--nodes="))

        assert gpus_idx < nodes_idx

    def test_flags_before_command(self, mock_popen, mock_uuid):
        """Verify all flags come before the job command."""
        job = JobDefinition(
            run_id="position_test",
            cmd="experiments.recipes.arena.train",
            gpus=2,
            nodes=2,
            type=JobTypes.LAUNCH_TRAINING,
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        args, kwargs = mock_popen.call_args
        cmd = args[0]

        # Find position of the command
        cmd_idx = cmd.index("experiments.recipes.arena.train")

        # Check all flags come before command
        for i, arg in enumerate(cmd[:cmd_idx]):
            if i > 0:  # Skip the script path
                assert arg.startswith("--"), f"Non-flag argument {arg} found before command"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in dispatcher."""

    def test_file_not_found_error(self, basic_training_job):
        """Test handling when launch.py doesn't exist."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = FileNotFoundError("launch.py not found")

            dispatcher = SkypilotDispatcher()

            with pytest.raises(FileNotFoundError, match="launch.py not found"):
                dispatcher.dispatch(basic_training_job)

    def test_permission_error(self, basic_training_job):
        """Test handling of permission errors."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = PermissionError("Permission denied")

            dispatcher = SkypilotDispatcher()

            with pytest.raises(PermissionError, match="Permission denied"):
                dispatcher.dispatch(basic_training_job)

    def test_generic_exception(self, basic_training_job):
        """Test that generic exceptions are re-raised."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = RuntimeError("Unexpected error")

            dispatcher = SkypilotDispatcher()

            with pytest.raises(RuntimeError, match="Unexpected error"):
                dispatcher.dispatch(basic_training_job)

    def test_logging_on_error(self, caplog, basic_training_job):
        """Test that errors are logged before re-raising."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = FileNotFoundError("launch.py not found")

            dispatcher = SkypilotDispatcher()

            with pytest.raises(FileNotFoundError):
                dispatcher.dispatch(basic_training_job)

        # Check that error was logged
        assert "Failed to launch job" in caplog.text
        assert "sweep_test_trial_001" in caplog.text


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for SkypilotDispatcher."""

    def test_implements_dispatcher_protocol(self):
        """Verify SkypilotDispatcher implements Dispatcher protocol."""
        dispatcher = SkypilotDispatcher()
        assert isinstance(dispatcher, Dispatcher)

    def test_dispatch_returns_valid_uuid(self, mock_popen, basic_training_job):
        """Test that dispatch returns a valid UUID string."""
        dispatcher = SkypilotDispatcher()
        dispatch_id = dispatcher.dispatch(basic_training_job)

        # Verify it's a string
        assert isinstance(dispatch_id, str)

        # Verify it's a valid UUID format
        try:
            uuid.UUID(dispatch_id)
        except ValueError:
            pytest.fail(f"dispatch_id '{dispatch_id}' is not a valid UUID")

    def test_process_pid_logged(self, mock_popen, caplog, basic_training_job):
        """Test that process PID is logged for debugging."""
        import logging

        mock_process = MagicMock()
        mock_process.pid = 99999
        mock_popen.return_value = mock_process

        with caplog.at_level(logging.INFO):
            dispatcher = SkypilotDispatcher()
            dispatcher.dispatch(basic_training_job)

        assert "PID 99999" in caplog.text

    def test_fire_and_forget_behavior(self, mock_popen, basic_training_job):
        """Test that dispatcher doesn't wait for or track the process."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(basic_training_job)

        # Verify process methods weren't called (fire-and-forget)
        mock_process.wait.assert_not_called()
        mock_process.poll.assert_not_called()
        mock_process.communicate.assert_not_called()

        # Verify DEVNULL is used for output
        args, kwargs = mock_popen.call_args
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL


# ============================================================================
# Comparison Tests with LocalDispatcher
# ============================================================================


class TestDispatcherComparison:
    """Test that SkypilotDispatcher and LocalDispatcher build identical command portions."""

    def test_command_equivalence_basic(self, basic_training_job):
        """Test that both dispatchers build the same args/overrides for basic job."""
        from metta.sweep import LocalDispatcher

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # Dispatch with LocalDispatcher
            local_dispatcher = LocalDispatcher(capture_output=False)
            local_dispatcher.dispatch(basic_training_job)
            local_call = mock_popen.call_args_list[0]

            # Reset mock
            mock_popen.reset_mock()

            # Dispatch with SkypilotDispatcher
            sky_dispatcher = SkypilotDispatcher()
            sky_dispatcher.dispatch(basic_training_job)
            sky_call = mock_popen.call_args_list[0]

            # Extract commands
            local_cmd = local_call[0][0]
            sky_cmd = sky_call[0][0]

            # Compare everything after the base command
            # LocalDispatcher: ["uv", "run", "./tools/run.py", cmd, ...]
            # SkypilotDispatcher: ["./devops/skypilot/launch.py", "--no-spot", cmd, ...]
            local_after_cmd = local_cmd[local_cmd.index("experiments.recipes.arena.train") :]
            sky_after_cmd = sky_cmd[sky_cmd.index("experiments.recipes.arena.train") :]

            assert local_after_cmd == sky_after_cmd

    def test_command_equivalence_complex(self, complex_job):
        """Test command equivalence for complex job with all parameters."""
        from metta.sweep import LocalDispatcher

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # Dispatch with LocalDispatcher
            local_dispatcher = LocalDispatcher(capture_output=False)
            local_dispatcher.dispatch(complex_job)
            local_call = mock_popen.call_args_list[0]

            # Reset mock
            mock_popen.reset_mock()

            # Dispatch with SkypilotDispatcher
            sky_dispatcher = SkypilotDispatcher()
            sky_dispatcher.dispatch(complex_job)
            sky_call = mock_popen.call_args_list[0]

            # Extract commands
            local_cmd = local_call[0][0]
            sky_cmd = sky_call[0][0]

            # Find where the job command starts
            local_cmd_start = local_cmd.index("experiments.recipes.navigation.train_shaped")
            sky_cmd_start = sky_cmd.index("experiments.recipes.navigation.train_shaped")

            # Everything from job command onwards should be identical
            local_job_portion = local_cmd[local_cmd_start:]
            sky_job_portion = sky_cmd[sky_cmd_start:]

            assert local_job_portion == sky_job_portion

    def test_eval_job_equivalence(self, basic_eval_job):
        """Test that eval jobs are handled identically by both dispatchers."""
        from metta.sweep import LocalDispatcher

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # Dispatch with LocalDispatcher
            local_dispatcher = LocalDispatcher(capture_output=False)
            local_dispatcher.dispatch(basic_eval_job)
            local_call = mock_popen.call_args_list[0]

            # Reset mock
            mock_popen.reset_mock()

            # Dispatch with SkypilotDispatcher
            sky_dispatcher = SkypilotDispatcher()
            sky_dispatcher.dispatch(basic_eval_job)
            sky_call = mock_popen.call_args_list[0]

            # Extract commands
            local_cmd = local_call[0][0]
            sky_cmd = sky_call[0][0]

            # Check that neither includes run_id (since it's LAUNCH_EVAL)
            local_args_portion = local_cmd[local_cmd.index("--args") + 1 :]
            sky_args_portion = sky_cmd[sky_cmd.index("--args") + 1 :]

            # Both should have metadata but no run_id
            assert "policy_uri=file://./checkpoints/policy.pt" in local_args_portion
            assert "policy_uri=file://./checkpoints/policy.pt" in sky_args_portion
            assert not any("run=" in arg for arg in local_args_portion)
            assert not any("run=" in arg for arg in sky_args_portion)


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_args_no_flag(self, mock_popen, mock_uuid):
        """Test that --args flag is not added when there are no args."""
        job = JobDefinition(
            run_id="no_args_test",
            cmd="experiments.recipes.arena.evaluate",
            type=JobTypes.LAUNCH_EVAL,
            # No metadata, no args
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        args, kwargs = mock_popen.call_args
        cmd = args[0]

        # Should not contain --args flag
        assert "--args" not in cmd

    def test_empty_overrides_no_flag(self, mock_popen, mock_uuid):
        """Test that --overrides flag is not added when there are no overrides."""
        job = JobDefinition(
            run_id="no_overrides_test",
            cmd="experiments.recipes.arena.train",
            type=JobTypes.LAUNCH_TRAINING,
            # No overrides, no config
        )

        dispatcher = SkypilotDispatcher()
        dispatcher.dispatch(job)

        args, kwargs = mock_popen.call_args
        cmd = args[0]

        # Should not contain --overrides flag
        assert "--overrides" not in cmd

    def test_trial_id_extraction(self, mock_popen, mock_uuid, caplog):
        """Test that trial IDs are extracted for cleaner logging."""
        import logging

        job = JobDefinition(
            run_id="sweep_experiment_trial_042", cmd="experiments.recipes.arena.train", type=JobTypes.LAUNCH_TRAINING
        )

        with caplog.at_level(logging.INFO):
            dispatcher = SkypilotDispatcher()
            dispatcher.dispatch(job)

        # Check that the log shows the cleaned trial ID
        assert "trial_042" in caplog.text

    def test_non_trial_run_id(self, mock_popen, mock_uuid, caplog):
        """Test handling of run_ids without trial pattern."""
        import logging

        job = JobDefinition(
            run_id="custom_run_name", cmd="experiments.recipes.arena.train", type=JobTypes.LAUNCH_TRAINING
        )

        with caplog.at_level(logging.INFO):
            dispatcher = SkypilotDispatcher()
            dispatcher.dispatch(job)

        # Should use the full run_id in logs
        assert "custom_run_name" in caplog.text
