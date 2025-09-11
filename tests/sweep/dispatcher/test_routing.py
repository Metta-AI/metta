"""Tests for RoutingDispatcher implementation."""

from unittest.mock import MagicMock, patch

import pytest

from metta.sweep import Dispatcher, JobDefinition, JobTypes, LocalDispatcher
from metta.sweep.dispatcher.routing import RoutingDispatcher
from metta.sweep.dispatcher.skypilot import SkypilotDispatcher

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_local_dispatcher():
    """Mock LocalDispatcher for testing."""
    mock = MagicMock(spec=LocalDispatcher)
    mock.dispatch.return_value = "local-dispatch-id-123"
    return mock


@pytest.fixture
def mock_skypilot_dispatcher():
    """Mock SkypilotDispatcher for testing."""
    mock = MagicMock(spec=SkypilotDispatcher)
    mock.dispatch.return_value = "sky-dispatch-id-456"
    return mock


@pytest.fixture
def training_job():
    """Sample training job."""
    return JobDefinition(
        run_id="sweep_test_trial_001",
        cmd="experiments.recipes.arena.train",
        type=JobTypes.LAUNCH_TRAINING,
    )


@pytest.fixture
def eval_job():
    """Sample evaluation job."""
    return JobDefinition(
        run_id="sweep_test_trial_001_eval",
        cmd="experiments.recipes.arena.evaluate",
        type=JobTypes.LAUNCH_EVAL,
        metadata={"policy_uri": "file://./checkpoints/policy.pt"},
    )


# ============================================================================
# Basic Routing Tests
# ============================================================================


class TestBasicRouting:
    """Test basic routing functionality."""

    def test_route_training_to_correct_dispatcher(self, mock_local_dispatcher, mock_skypilot_dispatcher, training_job):
        """Test that training jobs are routed to the correct dispatcher."""
        router = RoutingDispatcher(
            routes={
                JobTypes.LAUNCH_TRAINING: mock_skypilot_dispatcher,
                JobTypes.LAUNCH_EVAL: mock_local_dispatcher,
            }
        )

        dispatch_id = router.dispatch(training_job)

        # Should have called skypilot dispatcher
        mock_skypilot_dispatcher.dispatch.assert_called_once_with(training_job)
        mock_local_dispatcher.dispatch.assert_not_called()
        assert dispatch_id == "sky-dispatch-id-456"

    def test_route_eval_to_correct_dispatcher(self, mock_local_dispatcher, mock_skypilot_dispatcher, eval_job):
        """Test that eval jobs are routed to the correct dispatcher."""
        router = RoutingDispatcher(
            routes={
                JobTypes.LAUNCH_TRAINING: mock_skypilot_dispatcher,
                JobTypes.LAUNCH_EVAL: mock_local_dispatcher,
            }
        )

        dispatch_id = router.dispatch(eval_job)

        # Should have called local dispatcher
        mock_local_dispatcher.dispatch.assert_called_once_with(eval_job)
        mock_skypilot_dispatcher.dispatch.assert_not_called()
        assert dispatch_id == "local-dispatch-id-123"

    def test_default_dispatcher_fallback(self, mock_local_dispatcher):
        """Test fallback to default dispatcher for unmapped job types."""
        router = RoutingDispatcher(
            routes={JobTypes.LAUNCH_TRAINING: mock_local_dispatcher},
            default_dispatcher=mock_local_dispatcher,
        )

        # Create an eval job (not in routes)
        eval_job = JobDefinition(
            run_id="test_eval", cmd="experiments.recipes.arena.evaluate", type=JobTypes.LAUNCH_EVAL
        )

        dispatch_id = router.dispatch(eval_job)

        # Should use default dispatcher
        mock_local_dispatcher.dispatch.assert_called_once_with(eval_job)
        assert dispatch_id == "local-dispatch-id-123"

    def test_error_when_no_route_and_no_default(self, training_job):
        """Test that error is raised when no route exists and no default."""
        router = RoutingDispatcher(
            routes={JobTypes.LAUNCH_EVAL: MagicMock()}  # Only eval route
        )

        with pytest.raises(ValueError, match="No dispatcher configured for job type"):
            router.dispatch(training_job)


# ============================================================================
# Hybrid Mode Tests
# ============================================================================


class TestHybridModes:
    """Test hybrid execution modes."""

    def test_hybrid_remote_train_mode(self, training_job, eval_job):
        """Test hybrid mode with remote training and local evaluation."""
        mock_local = MagicMock()
        mock_local.dispatch.return_value = "local-123"

        mock_sky = MagicMock()
        mock_sky.dispatch.return_value = "sky-456"

        # Create router for hybrid remote train mode
        router = RoutingDispatcher(
            routes={
                JobTypes.LAUNCH_TRAINING: mock_sky,
                JobTypes.LAUNCH_EVAL: mock_local,
            }
        )

        # Dispatch training job - should go to Skypilot
        train_id = router.dispatch(training_job)
        assert train_id == "sky-456"
        mock_sky.dispatch.assert_called_once_with(training_job)

        # Dispatch eval job - should go to Local
        eval_id = router.dispatch(eval_job)
        assert eval_id == "local-123"
        mock_local.dispatch.assert_called_once_with(eval_job)


# ============================================================================
# Introspection Tests
# ============================================================================


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Test logging functionality."""

    def test_initialization_no_verbose_logging(self, caplog, mock_local_dispatcher, mock_skypilot_dispatcher):
        """Test that initialization doesn't produce verbose logs."""
        import logging

        with caplog.at_level(logging.INFO):
            RoutingDispatcher(
                routes={
                    JobTypes.LAUNCH_TRAINING: mock_skypilot_dispatcher,
                    JobTypes.LAUNCH_EVAL: mock_local_dispatcher,
                },
                default_dispatcher=mock_local_dispatcher,
            )

        # Check that initialization is silent at INFO level (minimal logging)
        assert caplog.text == ""

    def test_minimal_logging_at_info_level(self, caplog, mock_local_dispatcher):
        """Test that routing produces minimal logs at INFO level."""
        import logging

        router = RoutingDispatcher(routes={JobTypes.LAUNCH_TRAINING: mock_local_dispatcher})

        job = JobDefinition(
            run_id="sweep_experiment_trial_042", cmd="experiments.recipes.arena.train", type=JobTypes.LAUNCH_TRAINING
        )

        with caplog.at_level(logging.INFO):
            router.dispatch(job)

        # Check that no logs are produced at INFO level (minimal logging)
        assert caplog.text == ""


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with real dispatcher instances."""

    def test_with_real_dispatchers(self):
        """Test with real LocalDispatcher and SkypilotDispatcher instances."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # Create real dispatchers
            local = LocalDispatcher(capture_output=False)
            sky = SkypilotDispatcher()

            # Create router
            router = RoutingDispatcher(
                routes={
                    JobTypes.LAUNCH_TRAINING: sky,
                    JobTypes.LAUNCH_EVAL: local,
                }
            )

            # Test training job
            train_job = JobDefinition(run_id="train_001", cmd="train", type=JobTypes.LAUNCH_TRAINING)
            train_id = router.dispatch(train_job)
            assert isinstance(train_id, str)

            # Test eval job
            eval_job = JobDefinition(run_id="eval_001", cmd="eval", type=JobTypes.LAUNCH_EVAL)
            eval_id = router.dispatch(eval_job)
            assert isinstance(eval_id, str)

            # Verify both dispatchers were called
            assert mock_popen.call_count == 2

    def test_implements_dispatcher_protocol(self):
        """Verify RoutingDispatcher implements Dispatcher protocol."""
        router = RoutingDispatcher(routes={})
        assert isinstance(router, Dispatcher)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_routes_with_default(self, mock_local_dispatcher, training_job):
        """Test router with empty routes but default dispatcher."""
        router = RoutingDispatcher(routes={}, default_dispatcher=mock_local_dispatcher)

        dispatch_id = router.dispatch(training_job)

        # Should use default
        mock_local_dispatcher.dispatch.assert_called_once_with(training_job)
        assert dispatch_id == "local-dispatch-id-123"

    def test_multiple_job_types_same_dispatcher(self, mock_local_dispatcher, training_job, eval_job):
        """Test routing multiple job types to the same dispatcher."""
        router = RoutingDispatcher(
            routes={
                JobTypes.LAUNCH_TRAINING: mock_local_dispatcher,
                JobTypes.LAUNCH_EVAL: mock_local_dispatcher,
            }
        )

        # Both should go to the same dispatcher
        train_id = router.dispatch(training_job)
        eval_id = router.dispatch(eval_job)

        assert mock_local_dispatcher.dispatch.call_count == 2
        assert train_id == "local-dispatch-id-123"
        assert eval_id == "local-dispatch-id-123"

    def test_dispatcher_exception_propagation(self, training_job):
        """Test that exceptions from dispatchers are propagated."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.side_effect = RuntimeError("Dispatch failed")

        router = RoutingDispatcher(routes={JobTypes.LAUNCH_TRAINING: mock_dispatcher})

        with pytest.raises(RuntimeError, match="Dispatch failed"):
            router.dispatch(training_job)
