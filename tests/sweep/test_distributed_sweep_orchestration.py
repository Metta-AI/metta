"""Test distributed sweep orchestration without actual training."""

import os
from unittest.mock import Mock, patch


class TestDistributedSweepOrchestration:
    """Test that sweep orchestration works correctly in distributed settings."""

    def test_distributed_initialization_single_node(self):
        """Test that single-node runs don't initialize distributed."""
        from metta.common.util.lock import _init_process_group

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with patch("torch.distributed.init_process_group") as mock_init:
                result = _init_process_group()
                assert result is False
                assert not mock_init.called

    def test_distributed_initialization_multi_node(self):
        """Test that multi-node runs properly initialize distributed."""
        from metta.common.util.lock import _init_process_group

        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}):
            with patch("torch.distributed.is_initialized", return_value=False):
                with patch("torch.distributed.init_process_group") as mock_init:
                    result = _init_process_group()
                    assert result is True
                    mock_init.assert_called_once_with(backend="nccl", init_method="env://", world_size=2, rank=0)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.broadcast_object_list")
    def test_run_once_master_execution(self, mock_broadcast, mock_get_rank, mock_is_initialized):
        """Test that run_once executes function only on master and broadcasts result."""
        from metta.common.util.lock import run_once

        # Setup mocks
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0  # Master node

        # Function to test
        test_fn = Mock(return_value={"result": "from_master"})

        # Run the function
        result = run_once(test_fn, destroy_on_finish=False)

        # Verify function was called on master
        assert test_fn.called
        assert result == {"result": "from_master"}

        # Verify broadcast was called
        mock_broadcast.assert_called_once()
        broadcast_args = mock_broadcast.call_args[0]
        assert broadcast_args[0] == [{"result": "from_master"}]
        assert mock_broadcast.call_args[1]["src"] == 0

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.broadcast_object_list")
    def test_run_once_worker_receives_broadcast(self, mock_broadcast, mock_get_rank, mock_is_initialized):
        """Test that run_once on worker nodes receives broadcasted result without executing."""
        from metta.common.util.lock import run_once

        # Setup mocks
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 1  # Worker node

        # Mock broadcast to simulate receiving data
        def simulate_broadcast(objects, src=0):
            objects[0] = {"result": "from_master"}

        mock_broadcast.side_effect = simulate_broadcast

        # Function that should NOT be called on worker
        test_fn = Mock(return_value={"result": "should_not_see_this"})

        # Run the function
        result = run_once(test_fn, destroy_on_finish=False)

        # Verify function was NOT called on worker
        assert not test_fn.called

        # Verify we got the broadcasted result
        assert result == {"result": "from_master"}

    @patch("subprocess.run")
    def test_training_command_construction(self, mock_subprocess):
        """Test that training commands are properly constructed for distributed runs."""
        mock_subprocess.return_value = Mock(returncode=0)

        from tools.sweep_rollout import train_for_run

        # Test with additional arguments
        original_args = ["hardware=h100", "wandb.mode=offline", "trainer.num_workers=4"]

        train_for_run(
            run_name="test_run", dist_cfg_path="/tmp/dist.yaml", data_dir="/tmp/data", original_args=original_args
        )

        # Verify the command includes both required and passthrough args
        call_args = mock_subprocess.call_args[0][0]
        assert "./devops/train.sh" in call_args
        assert "run=test_run" in call_args
        assert "dist_cfg_path=/tmp/dist.yaml" in call_args
        assert "data_dir=/tmp/data" in call_args
        assert "hardware=h100" in call_args
        assert "wandb.mode=offline" in call_args
        assert "trainer.num_workers=4" in call_args

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.broadcast_object_list")
    def test_multiple_run_once_calls(self, mock_broadcast, mock_get_rank, mock_is_initialized):
        """Test multiple run_once calls in sequence work correctly."""
        from metta.common.util.lock import run_once

        # Setup mocks
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0  # Master node

        # Track broadcasts
        broadcasts = []

        def capture_broadcast(objects, src=0):
            broadcasts.append(objects[0])

        mock_broadcast.side_effect = capture_broadcast

        # Multiple functions to run
        fn1 = Mock(return_value="result1")
        fn2 = Mock(return_value="result2")
        fn3 = Mock(return_value="result3")

        # Run them in sequence
        r1 = run_once(fn1, destroy_on_finish=False)
        r2 = run_once(fn2, destroy_on_finish=False)
        r3 = run_once(fn3, destroy_on_finish=False)

        # Verify all functions were called
        assert fn1.called
        assert fn2.called
        assert fn3.called

        # Verify results
        assert r1 == "result1"
        assert r2 == "result2"
        assert r3 == "result3"

        # Verify broadcasts
        assert len(broadcasts) == 3
        assert broadcasts[0] == "result1"
        assert broadcasts[1] == "result2"
        assert broadcasts[2] == "result3"

    def test_sweep_lifecycle_imports(self):
        """Test that sweep lifecycle functions handle distributed execution properly."""
        # Just verify the imports work and functions exist
        from metta.sweep.sweep_lifecycle import evaluate_rollout, prepare_sweep_run, setup_sweep
        from tools.sweep_rollout import run_single_rollout

        # Verify they're callable
        assert callable(setup_sweep)
        assert callable(prepare_sweep_run)
        assert callable(evaluate_rollout)
        assert callable(run_single_rollout)
