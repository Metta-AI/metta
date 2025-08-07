"""Tests for metta.common.util.lock module."""

import os
from unittest.mock import Mock, patch

import pytest

from metta.common.util.lock import _init_process_group, run_once


class TestInitProcessGroup:
    """Test cases for the _init_process_group function."""

    def test_single_node_returns_false(self):
        """Test that single node setup returns False (no distributed training)."""
        with patch.dict(os.environ, {}, clear=True):
            # No environment variables set, defaults to single node
            result = _init_process_group()
            assert result is False

    def test_single_node_explicit_world_size_returns_false(self):
        """Test that explicitly setting WORLD_SIZE=1 returns False."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            result = _init_process_group()
            assert result is False

    def test_single_node_num_nodes_returns_false(self):
        """Test that NUM_NODES=1 also results in single node behavior."""
        with patch.dict(os.environ, {"NUM_NODES": "1"}, clear=True):
            result = _init_process_group()
            assert result is False

    @patch("torch.distributed.is_initialized")
    def test_already_initialized_returns_false(self, mock_is_initialized):
        """Test that if distributed is already initialized, returns False."""
        mock_is_initialized.return_value = True

        with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True):
            result = _init_process_group()
            assert result is False
            mock_is_initialized.assert_called_once()

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized")
    def test_multi_node_world_size_initializes(self, mock_is_initialized, mock_init_process_group):
        """Test that multi-node setup with WORLD_SIZE initializes distributed training."""
        mock_is_initialized.return_value = False

        env_vars = {
            "WORLD_SIZE": "4",
            "RANK": "0",
            "DIST_URL": "tcp://localhost:23456"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = _init_process_group()

            assert result is True
            mock_is_initialized.assert_called_once()
            mock_init_process_group.assert_called_once_with(
                backend="nccl",
                init_method="tcp://localhost:23456",
                world_size=4,
                rank=0
            )

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized")
    def test_multi_node_num_nodes_initializes(self, mock_is_initialized, mock_init_process_group):
        """Test that multi-node setup with NUM_NODES also works."""
        mock_is_initialized.return_value = False

        env_vars = {
            "NUM_NODES": "3",
            "NODE_INDEX": "1",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = _init_process_group()

            assert result is True
            mock_init_process_group.assert_called_once_with(
                backend="nccl",
                init_method="env://",  # default when DIST_URL not set
                world_size=3,
                rank=1
            )

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized")
    def test_default_values_when_missing(self, mock_is_initialized, mock_init_process_group):
        """Test default values are used when rank/dist_url environment variables are missing."""
        mock_is_initialized.return_value = False

        with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True):
            result = _init_process_group()

            assert result is True
            mock_init_process_group.assert_called_once_with(
                backend="nccl",
                init_method="env://",  # default
                world_size=2,
                rank=0  # default
            )

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized")
    def test_env_var_priority_world_size_over_num_nodes(self, mock_is_initialized, mock_init_process_group):
        """Test that WORLD_SIZE takes priority over NUM_NODES when both are set."""
        mock_is_initialized.return_value = False

        env_vars = {
            "WORLD_SIZE": "5",
            "NUM_NODES": "3",  # This should be ignored
            "RANK": "2",
            "NODE_INDEX": "1",  # This should be ignored
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = _init_process_group()

            assert result is True
            mock_init_process_group.assert_called_once_with(
                backend="nccl",
                init_method="env://",
                world_size=5,  # Uses WORLD_SIZE
                rank=2  # Uses RANK
            )


class TestRunOnce:
    """Test cases for the run_once function."""

    def test_single_node_executes_function_directly(self):
        """Test that in single node setup, function executes directly without distributed logic."""
        test_value = "test_result"
        test_function = Mock(return_value=test_value)

        with patch("metta.common.util.lock._init_process_group", return_value=False):
            with patch("torch.distributed.is_initialized", return_value=False):
                result = run_once(test_function)

                assert result == test_value
                test_function.assert_called_once()

    @patch("torch.distributed.destroy_process_group")
    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.is_initialized")
    @patch("metta.common.util.lock._init_process_group")
    def test_rank_0_executes_function_and_broadcasts(
        self,
        mock_init_process_group,
        mock_is_initialized,
        mock_get_rank,
        mock_broadcast_object_list,
        mock_destroy_process_group,
    ):
        """Test that rank 0 executes function and broadcasts result to other ranks."""
        test_value = "rank_0_result"
        test_function = Mock(return_value=test_value)

        # Setup distributed training scenario where process group is initialized
        mock_init_process_group.return_value = True
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0  # This is rank 0

        # Mock the broadcast behavior
        def mock_broadcast(result_list, src):
            # Simulate broadcast by ensuring the list contains the result
            assert result_list == [test_value]
            assert src == 0

        mock_broadcast_object_list.side_effect = mock_broadcast

        result = run_once(test_function)

        assert result == test_value
        test_function.assert_called_once()
        mock_broadcast_object_list.assert_called_once_with([test_value], src=0)
        mock_destroy_process_group.assert_called_once()

    @patch("torch.distributed.destroy_process_group")
    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.is_initialized")
    @patch("metta.common.util.lock._init_process_group")
    def test_non_rank_0_waits_for_broadcast(
        self,
        mock_init_process_group,
        mock_is_initialized,
        mock_get_rank,
        mock_broadcast_object_list,
        mock_destroy_process_group,
    ):
        """Test that non-rank 0 processes don't execute function but receive broadcasted result."""
        test_value = "broadcasted_result"
        test_function = Mock(return_value="should_not_be_called")

        # Setup distributed training scenario
        mock_init_process_group.return_value = True
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 1  # This is rank 1 (not rank 0)

        # Mock the broadcast to simulate receiving the result from rank 0
        def mock_broadcast(result_list, src):
            result_list[0] = test_value  # Simulate receiving the broadcasted value

        mock_broadcast_object_list.side_effect = mock_broadcast

        result = run_once(test_function)

        assert result == test_value
        test_function.assert_not_called()  # Function should not be called on non-rank 0
        # Check that broadcast was called with None initially (before the mock modified it)
        args, kwargs = mock_broadcast_object_list.call_args
        assert args[0] == [test_value]  # After mock modification
        assert kwargs == {"src": 0}
        mock_destroy_process_group.assert_called_once()

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.is_initialized")
    @patch("metta.common.util.lock._init_process_group")
    def test_pre_initialized_distributed_no_destroy(
        self,
        mock_init_process_group,
        mock_is_initialized,
        mock_get_rank,
        mock_broadcast_object_list,
    ):
        """Test that when distributed is pre-initialized, process group is not destroyed."""
        test_value = "pre_initialized_result"
        test_function = Mock(return_value=test_value)

        # Process group was already initialized (not by this function)
        mock_init_process_group.return_value = False
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0

        mock_broadcast_object_list.side_effect = lambda result_list, src: None

        with patch("torch.distributed.destroy_process_group") as mock_destroy:
            result = run_once(test_function)

            assert result == test_value
            test_function.assert_called_once()
            mock_broadcast_object_list.assert_called_once()
            mock_destroy.assert_not_called()  # Should not destroy pre-existing group

    def test_function_with_arguments_and_complex_return(self):
        """Test that run_once works with functions that return complex objects."""
        complex_result = {
            "data": [1, 2, 3],
            "metadata": {"status": "success", "count": 3},
            "nested": {"deep": {"value": 42}}
        }

        def complex_function():
            return complex_result

        with patch("metta.common.util.lock._init_process_group", return_value=False):
            with patch("torch.distributed.is_initialized", return_value=False):
                result = run_once(complex_function)

                assert result == complex_result
                assert isinstance(result, dict)
                assert result["nested"]["deep"]["value"] == 42

    def test_function_that_raises_exception(self):
        """Test that exceptions in the function are propagated correctly."""
        def failing_function():
            raise ValueError("Test exception")

        with patch("metta.common.util.lock._init_process_group", return_value=False):
            with patch("torch.distributed.is_initialized", return_value=False):
                with pytest.raises(ValueError, match="Test exception"):
                    run_once(failing_function)

    def test_function_returns_none(self):
        """Test that functions returning None work correctly."""
        def none_function():
            return None

        with patch("metta.common.util.lock._init_process_group", return_value=False):
            with patch("torch.distributed.is_initialized", return_value=False):
                result = run_once(none_function)
                assert result is None

    def test_type_safety_with_typed_function(self):
        """Test that the function maintains type safety for the return value."""
        def string_function() -> str:
            return "typed_result"

        def int_function() -> int:
            return 42

        with patch("metta.common.util.lock._init_process_group", return_value=False):
            with patch("torch.distributed.is_initialized", return_value=False):
                string_result = run_once(string_function)
                int_result = run_once(int_function)

                assert isinstance(string_result, str)
                assert string_result == "typed_result"

                assert isinstance(int_result, int)
                assert int_result == 42

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.is_initialized")
    @patch("metta.common.util.lock._init_process_group")
    def test_broadcast_preserves_object_type(
        self,
        mock_init_process_group,
        mock_is_initialized,
        mock_get_rank,
        mock_broadcast_object_list,
    ):
        """Test that the broadcast mechanism preserves object types correctly."""
        original_object = {"key": "value", "number": 123}
        test_function = Mock(return_value=original_object)

        mock_init_process_group.return_value = False
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 1  # Non-rank 0

        # Simulate broadcast preserving the object
        def mock_broadcast(result_list, src):
            result_list[0] = original_object

        mock_broadcast_object_list.side_effect = mock_broadcast

        result = run_once(test_function)

        assert result == original_object
        assert type(result) == type(original_object)
        test_function.assert_not_called()
