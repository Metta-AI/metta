"""Unit tests for protein utility functions."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from omegaconf import DictConfig

from metta.sweep.protein_utils import (
    apply_protein_suggestion,
    generate_protein_suggestion,
    validate_protein_suggestion,
)


class TestGenerateProteinSuggestion:
    """Test the generate_protein_suggestion function."""

    @pytest.fixture
    def mock_trainer_config(self):
        """Create a mock trainer configuration."""
        return {
            "batch_size": 2048,
            "minibatch_size": 64,
            "bptt_horizon": 32,
            "learning_rate": 0.001,
        }

    def test_generate_protein_suggestion_valid(self, mock_trainer_config):
        """Test successful protein suggestion generation."""
        # Mock protein
        mock_protein = Mock()
        mock_protein.suggest.return_value = (
            {
                "trainer": {
                    "batch_size": 4096,
                    "minibatch_size": 128,
                    "learning_rate": np.float64(0.005),
                }
            },
            {"cost": 100.0},
        )

        # Call function
        result = generate_protein_suggestion(mock_trainer_config, mock_protein)

        # Assertions
        assert result["trainer"]["batch_size"] == 4096
        assert result["trainer"]["learning_rate"] == 0.005
        assert isinstance(result["trainer"]["learning_rate"], float)  # numpy cleaned

    @patch("metta.sweep.protein_utils.logger")
    def test_generate_protein_suggestion_retry_on_invalid(self, mock_logger, mock_trainer_config):
        """Test retry mechanism when invalid suggestions are generated."""
        # Mock protein
        mock_protein = Mock()

        # First suggestion invalid, second valid
        mock_protein.suggest.side_effect = [
            (
                {
                    "trainer": {
                        "batch_size": 1000,  # Invalid: not divisible by minibatch_size
                        "minibatch_size": 64,
                    }
                },
                {},
            ),
            (
                {
                    "trainer": {
                        "batch_size": 2048,
                        "minibatch_size": 64,
                    }
                },
                {},
            ),
        ]

        # Call function
        result = generate_protein_suggestion(mock_trainer_config, mock_protein)

        # Assertions
        assert result["trainer"]["batch_size"] == 2048
        mock_protein.observe_failure.assert_called_once()
        mock_logger.warning.assert_called_once()

    def test_generate_protein_suggestion_max_retries(self, mock_trainer_config):
        """Test that max retries are enforced."""
        # Mock protein that always returns invalid suggestions
        mock_protein = Mock()
        mock_protein.suggest.return_value = (
            {
                "trainer": {
                    "batch_size": 1000,  # Always invalid
                    "minibatch_size": 64,
                }
            },
            {},
        )

        # Call function and expect exception
        with pytest.raises(ValueError, match="Batch size 1000 must be divisible"):
            generate_protein_suggestion(mock_trainer_config, mock_protein)

        # Should have tried 10 times
        assert mock_protein.suggest.call_count == 10
        assert mock_protein.observe_failure.call_count == 10


class TestValidateProteinSuggestion:
    """Test the validate_protein_suggestion function."""

    @pytest.fixture
    def mock_trainer_config(self):
        """Create a mock trainer configuration."""
        return {
            "batch_size": 2048,
            "minibatch_size": 64,
            "bptt_horizon": 32,
        }

    def test_validate_valid_constraints(self, mock_trainer_config):
        """Test validation passes for valid constraints."""
        suggestion = {
            "trainer": {
                "batch_size": 4096,
                "minibatch_size": 128,
                "bptt_horizon": 64,
            }
        }

        # Should not raise
        validate_protein_suggestion(mock_trainer_config, suggestion)

    def test_validate_batch_size_constraint(self, mock_trainer_config):
        """Test batch_size divisibility constraint."""
        suggestion = {
            "trainer": {
                "batch_size": 1000,  # Not divisible by 64
            }
        }

        with pytest.raises(ValueError, match="Batch size 1000 must be divisible by minibatch size 64"):
            validate_protein_suggestion(mock_trainer_config, suggestion)

    def test_validate_minibatch_bptt_constraint(self, mock_trainer_config):
        """Test minibatch_size divisibility by bptt constraint."""
        suggestion = {
            "trainer": {
                "batch_size": 3200,  # Divisible by 100
                "minibatch_size": 100,  # Not divisible by 32
            }
        }

        with pytest.raises(ValueError, match="Minibatch size 100 must be divisible by bppt 32"):
            validate_protein_suggestion(mock_trainer_config, suggestion)

    def test_validate_partial_update(self, mock_trainer_config):
        """Test validation with partial parameter updates."""
        # Only updating learning rate, constraints should still pass
        suggestion = {
            "trainer": {
                "learning_rate": 0.01,
            }
        }

        # Should not raise
        validate_protein_suggestion(mock_trainer_config, suggestion)

    def test_validate_nested_suggestion(self, mock_trainer_config):
        """Test validation with deeply nested suggestions."""
        suggestion = {
            "trainer": {
                "optimizer": {
                    "learning_rate": 0.01,
                },
                "batch_size": 4096,
                "minibatch_size": 128,
            }
        }

        # Should not raise
        validate_protein_suggestion(mock_trainer_config, suggestion)


class TestApplyProteinSuggestion:
    """Test the apply_protein_suggestion function."""

    def test_apply_simple_values(self):
        """Test applying simple value updates."""
        config = DictConfig(
            {
                "learning_rate": 0.001,
                "batch_size": 2048,
                "epochs": 10,
            }
        )

        suggestion = {
            "learning_rate": 0.005,
            "batch_size": 4096,
        }

        apply_protein_suggestion(config, suggestion)

        assert config.learning_rate == 0.005
        assert config.batch_size == 4096
        assert config.epochs == 10  # Unchanged

    def test_apply_deep_merge(self):
        """Test deep merge behavior for nested configs."""
        config = DictConfig(
            {
                "trainer": {
                    "optimizer": {
                        "type": "adam",
                        "learning_rate": 0.001,
                        "beta1": 0.9,
                    },
                    "batch_size": 2048,
                }
            }
        )

        suggestion = {
            "trainer": {
                "optimizer": {
                    "learning_rate": 0.005,
                    "beta2": 0.999,  # New field
                },
                "batch_size": 4096,
            }
        }

        apply_protein_suggestion(config, suggestion)

        # Check merge behavior
        assert config.trainer.optimizer.type == "adam"  # Preserved
        assert config.trainer.optimizer.learning_rate == 0.005  # Updated
        assert config.trainer.optimizer.beta1 == 0.9  # Preserved
        assert config.trainer.optimizer.beta2 == 0.999  # Added
        assert config.trainer.batch_size == 4096  # Updated

    def test_apply_numpy_cleaning(self):
        """Test that numpy types are cleaned during application."""
        config = DictConfig(
            {
                "params": {},
            }
        )

        suggestion = {
            "learning_rate": np.float64(0.001),
            "batch_size": np.int64(2048),
            "use_cuda": np.bool_(True),
            "params": {
                "hidden_size": np.int32(256),
                "dropout": np.float32(0.1),
            },
        }

        apply_protein_suggestion(config, suggestion)

        # All numpy types should be converted
        assert isinstance(config.learning_rate, float)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.use_cuda, bool)
        assert isinstance(config.params.hidden_size, int)
        assert isinstance(config.params.dropout, float)

    def test_apply_skip_suggestion_uuid(self):
        """Test that suggestion_uuid is skipped."""
        config = DictConfig(
            {
                "learning_rate": 0.001,
            }
        )

        suggestion = {
            "learning_rate": 0.005,
            "suggestion_uuid": "test-uuid-12345",
        }

        apply_protein_suggestion(config, suggestion)

        assert config.learning_rate == 0.005
        assert "suggestion_uuid" not in config

    def test_apply_overwrite_scalar_with_dict(self):
        """Test overwriting scalar values with dictionaries."""
        config = DictConfig(
            {
                "optimizer": "adam",  # String scalar
            }
        )

        suggestion = {
            "optimizer": {  # Dict replacement
                "type": "adam",
                "lr": 0.001,
                "weight_decay": 0.0001,
            }
        }

        apply_protein_suggestion(config, suggestion)

        assert isinstance(config.optimizer, DictConfig)
        assert config.optimizer.type == "adam"
        assert config.optimizer.lr == 0.001
        assert config.optimizer.weight_decay == 0.0001

    def test_apply_empty_suggestion(self):
        """Test that empty suggestions don't modify config."""
        original = {
            "learning_rate": 0.001,
            "batch_size": 2048,
        }
        config = DictConfig(original.copy())

        apply_protein_suggestion(config, {})

        assert config.learning_rate == original["learning_rate"]
        assert config.batch_size == original["batch_size"]

    def test_apply_list_values(self):
        """Test handling of list values in suggestions."""
        config = DictConfig(
            {
                "layers": [128, 256, 128],
            }
        )

        suggestion = {
            "layers": [256, 512, 256, 128],
        }

        apply_protein_suggestion(config, suggestion)

        assert config.layers == [256, 512, 256, 128]

    def test_apply_none_values(self):
        """Test handling of None values in suggestions."""
        config = DictConfig(
            {
                "dropout": 0.1,
                "weight_decay": 0.0001,
            }
        )

        suggestion = {
            "dropout": None,  # Disable dropout
            "weight_decay": 0.0,
        }

        apply_protein_suggestion(config, suggestion)

        assert config.dropout is None
        assert config.weight_decay == 0.0
