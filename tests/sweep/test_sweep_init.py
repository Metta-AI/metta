"""Tests for sweep initialization and protein suggestion validation."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from tools.sweep_prepare_run import (
    apply_protein_suggestion,
    generate_protein_suggestion,
    validate_protein_suggestion,
)


class TestValidateProteinSuggestion:
    """Test the validate_protein_suggestion function."""

    def test_valid_suggestion_no_changes(self):
        """Test validation passes when suggestion doesn't change parameters."""
        config = DictConfig({"trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}})
        suggestion = {"learning_rate": 0.001}  # No batch-related changes

        # Should not raise
        validate_protein_suggestion(config, suggestion)

    def test_valid_suggestion_with_changes(self):
        """Test validation passes with valid batch size changes."""
        config = DictConfig({"trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}})
        suggestion = {"trainer": {"batch_size": 4096, "minibatch_size": 128, "bptt_horizon": 64}}

        # Should not raise - 4096 % 128 == 0 and 128 % 64 == 0
        validate_protein_suggestion(config, suggestion)

    def test_invalid_batch_size_not_divisible_by_minibatch(self):
        """Test validation fails when batch_size is not divisible by minibatch_size."""
        config = DictConfig({"trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}})
        suggestion = {
            "trainer": {
                "batch_size": 1000,  # Not divisible by 64
                "minibatch_size": 64,
            }
        }

        with pytest.raises(ValueError, match="Batch size 1000 must be divisible by minibatch size 64"):
            validate_protein_suggestion(config, suggestion)

    def test_invalid_minibatch_not_divisible_by_bppt(self):
        """Test validation fails when minibatch_size is not divisible by bppt."""
        config = DictConfig({"trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}})
        suggestion = {
            "trainer": {
                "batch_size": 2048,  # Keep batch_size valid
                "minibatch_size": 64,  # Valid for batch_size
                "bptt_horizon": 50,  # 64 is not divisible by 50
            }
        }

        with pytest.raises(ValueError, match="Minibatch size 64 must be divisible by bppt 50"):
            validate_protein_suggestion(config, suggestion)

    def test_suggestion_updates_only_some_parameters(self):
        """Test validation when suggestion updates only some batch parameters."""
        config = DictConfig({"trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}})
        suggestion = {
            "trainer": {
                "batch_size": 4096  # Only updating batch_size
            }
        }

        # Should not raise - 4096 % 64 == 0
        validate_protein_suggestion(config, suggestion)

    def test_edge_case_all_equal(self):
        """Test edge case where all parameters are equal."""
        config = DictConfig({"trainer": {"batch_size": 64, "minibatch_size": 64, "bptt_horizon": 64}})
        suggestion = {}

        # Should not raise - 64 % 64 == 0
        validate_protein_suggestion(config, suggestion)

    def test_suggestion_with_nested_structure(self):
        """Test validation handles nested suggestion structures."""
        config = DictConfig({"trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}})
        suggestion = {
            "trainer": {
                "batch_size": 4096,  # Nested structure
                "minibatch_size": 128,
            }
        }

        # Should not raise - 4096 % 128 == 0
        validate_protein_suggestion(config, suggestion)


class TestGenerateProteinSuggestion:
    """Test the generate_protein_suggestion function."""

    @patch("tools.sweep_prepare_run.MettaProtein")
    def test_successful_generation(self, mock_protein_class):
        """Test successful protein suggestion generation."""
        # Mock protein instance
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        # Mock suggestion with numpy types
        mock_protein.suggest.return_value = (
            {"learning_rate": np.float64(0.001), "batch_size": np.int64(2048), "suggestion_uuid": "test-uuid"},
            {},  # info dict
        )

        config = DictConfig(
            {"sweep": {"parameters": {}}, "trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}}
        )

        result = generate_protein_suggestion(config, mock_protein)

        # Check numpy types were cleaned
        assert isinstance(result["learning_rate"], float)
        assert isinstance(result["batch_size"], int)
        assert result["learning_rate"] == 0.001
        assert result["batch_size"] == 2048

    @patch("tools.sweep_prepare_run.MettaProtein")
    @patch("tools.sweep_prepare_run.logger")
    def test_invalid_suggestion_retry(self, mock_logger, mock_protein_class):
        """Test that invalid suggestions trigger retry and record failure."""
        # Mock protein instance
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        # First suggestion is invalid, second is valid
        mock_protein.suggest.side_effect = [
            ({"trainer": {"batch_size": 1000, "minibatch_size": 64}}, {}),  # Invalid
            ({"trainer": {"batch_size": 2048, "minibatch_size": 64}}, {}),  # Valid
        ]

        config = DictConfig(
            {"sweep": {"parameters": {}}, "trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}}
        )

        result = generate_protein_suggestion(config, mock_protein)

        # Should have called record_failure for the invalid suggestion
        mock_protein.record_failure.assert_called_once()
        assert "Batch size 1000 must be divisible by minibatch size 64" in str(
            mock_protein.record_failure.call_args[0][0]
        )

        # Should return the valid suggestion
        assert result["trainer"]["batch_size"] == 2048

    @patch("tools.sweep_prepare_run.MettaProtein")
    def test_max_retries_exceeded(self, mock_protein_class):
        """Test that exception is raised after max retries."""
        # Mock protein instance
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        # Always return invalid suggestion
        mock_protein.suggest.return_value = ({"trainer": {"batch_size": 1000, "minibatch_size": 64}}, {})

        config = DictConfig(
            {"sweep": {"parameters": {}}, "trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32}}
        )

        with pytest.raises(ValueError, match="Batch size 1000 must be divisible by minibatch size 64"):
            generate_protein_suggestion(config, mock_protein)

        # Should have recorded 11 failures (1 initial + 10 retries)
        assert mock_protein.record_failure.call_count == 11


class TestApplyProteinSuggestion:
    """Test the apply_protein_suggestion function."""

    def test_apply_simple_values(self):
        """Test applying simple value suggestions."""
        config = DictConfig({"learning_rate": 0.001, "batch_size": 2048, "epochs": 10})

        suggestion = {"learning_rate": 0.0001, "batch_size": 4096}

        apply_protein_suggestion(config, suggestion)

        assert config.learning_rate == 0.0001
        assert config.batch_size == 4096
        assert config.epochs == 10  # Unchanged

    def test_apply_nested_dict_merge(self):
        """Test deep merge behavior for nested dictionaries."""
        config = DictConfig(
            {"trainer": {"ppo": {"learning_rate": 0.001, "batch_size": 2048, "clip_range": 0.2}, "epochs": 10}}
        )

        suggestion = {"trainer": {"ppo": {"learning_rate": 0.0001, "batch_size": 4096}}}

        apply_protein_suggestion(config, suggestion)

        # Should merge, not replace
        assert config.trainer.ppo.learning_rate == 0.0001
        assert config.trainer.ppo.batch_size == 4096
        assert config.trainer.ppo.clip_range == 0.2  # Should be preserved
        assert config.trainer.epochs == 10  # Should be preserved

    def test_skip_suggestion_uuid(self):
        """Test that suggestion_uuid is skipped."""
        config = DictConfig({"learning_rate": 0.001})

        suggestion = {"learning_rate": 0.0001, "suggestion_uuid": "test-uuid-123"}

        apply_protein_suggestion(config, suggestion)

        assert config.learning_rate == 0.0001
        assert "suggestion_uuid" not in config

    def test_numpy_type_cleaning(self):
        """Test that numpy types are cleaned during application."""
        config = DictConfig({"params": {}})

        suggestion = {
            "learning_rate": np.float64(0.001),
            "batch_size": np.int64(2048),
            "use_cuda": np.bool_(True),
            "params": {"hidden_size": np.int32(256)},
        }

        apply_protein_suggestion(config, suggestion)

        # All numpy types should be converted to Python types
        assert isinstance(config.learning_rate, float)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.use_cuda, bool)
        assert isinstance(config.params.hidden_size, int)

    def test_overwrite_non_dict_with_dict(self):
        """Test that non-dict values are overwritten by dict suggestions."""
        config = DictConfig(
            {
                "optimizer": "adam"  # String value
            }
        )

        suggestion = {
            "optimizer": {  # Dict value
                "type": "adam",
                "lr": 0.001,
            }
        }

        apply_protein_suggestion(config, suggestion)

        assert isinstance(config.optimizer, DictConfig)
        assert config.optimizer.type == "adam"
        assert config.optimizer.lr == 0.001

    def test_empty_suggestion(self):
        """Test that empty suggestion doesn't modify config."""
        original_config = {"learning_rate": 0.001, "batch_size": 2048}
        config = DictConfig(original_config.copy())

        apply_protein_suggestion(config, {})

        assert config.learning_rate == 0.001
        assert config.batch_size == 2048


class TestIntegration:
    """Integration tests for the validation pipeline."""

    @patch("tools.sweep_prepare_run.MettaProtein")
    def test_full_pipeline_flow(self, mock_protein_class):
        """Test the full flow from generation through validation to application."""
        # Mock protein instance
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        # Mock suggestion
        mock_protein.suggest.return_value = (
            {
                "learning_rate": np.float64(0.0001),
                "batch_size": 4096,
                "minibatch_size": 128,
                "trainer": {"ppo": {"clip_range": 0.3}},
            },
            {},
        )

        # Initial config
        config = DictConfig(
            {
                "sweep": {"parameters": {}},
                "learning_rate": 0.001,
                "batch_size": 2048,
                "trainer": {
                    "ppo": {
                        "batch_size": 2048,
                        "minibatch_size": 64,
                        "bppt": 32,
                        "clip_range": 0.2,
                        "entropy_coef": 0.01,
                    }
                },
            }
        )

        # Generate suggestion
        suggestion = generate_protein_suggestion(config, mock_protein)

        # Apply suggestion to a copy of the config
        config_copy = DictConfig(OmegaConf.to_container(config, resolve=True))
        apply_protein_suggestion(config_copy, suggestion)

        # Verify results
        assert config_copy.learning_rate == 0.0001
        assert config_copy.batch_size == 4096
        assert config_copy.trainer.ppo.clip_range == 0.3
        assert config_copy.trainer.ppo.entropy_coef == 0.01  # Preserved

        # Original config should be unchanged
        assert config.learning_rate == 0.001
        assert config.batch_size == 2048
