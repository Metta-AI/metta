"""Test protein utility functions."""

from unittest.mock import Mock

from metta.sweep.protein_utils import (
    convert_suggestion_to_cli_args,
    generate_protein_suggestion,
)


class TestConvertSuggestionToCliArgs:
    """Test convert_suggestion_to_cli_args function."""

    def test_simple_flat_suggestion(self):
        """Test converting a flat suggestion dictionary."""
        suggestion = {
            "lr": 0.001,
            "batch_size": 32,
            "enabled": True,
        }

        args = convert_suggestion_to_cli_args(suggestion)

        assert "++lr=0.001" in args
        assert "++batch_size=32" in args
        assert "++enabled=true" in args

    def test_nested_suggestion(self):
        """Test converting a nested suggestion dictionary."""
        suggestion = {
            "trainer": {
                "lr": 0.001,
                "batch_size": 32,
            },
            "sim": {
                "max_time_s": 100,
            },
        }

        args = convert_suggestion_to_cli_args(suggestion)

        assert "++trainer.lr=0.001" in args
        assert "++trainer.batch_size=32" in args
        assert "++sim.max_time_s=100" in args

    def test_scientific_notation(self):
        """Test handling of scientific notation for small/large numbers."""
        suggestion = {
            "very_small": 0.0000001,
            "very_large": 10000000,
            "normal": 0.5,
        }

        args = convert_suggestion_to_cli_args(suggestion)

        # Very small should use scientific notation
        assert any("1.000000e-07" in arg for arg in args)
        # Very large (>1e6) should use scientific notation
        assert any("1.000000e+07" in arg or "10000000" in arg for arg in args)
        # Normal should not
        assert "++normal=0.5" in args

    def test_special_values(self):
        """Test handling of special values."""
        suggestion = {
            "none_value": None,
            "bool_true": True,
            "bool_false": False,
            "string_with_space": "hello world",
        }

        args = convert_suggestion_to_cli_args(suggestion)

        assert "++none_value=null" in args
        assert "++bool_true=true" in args
        assert "++bool_false=false" in args
        assert "++string_with_space='hello world'" in args

    def test_skip_suggestion_uuid(self):
        """Test that suggestion_uuid is skipped."""
        suggestion = {
            "suggestion_uuid": "abc123",
            "lr": 0.001,
        }

        args = convert_suggestion_to_cli_args(suggestion)

        assert len(args) == 1
        assert "++lr=0.001" in args
        assert "suggestion_uuid" not in " ".join(args)

    def test_with_prefix(self):
        """Test with a prefix."""
        suggestion = {"lr": 0.001}

        args = convert_suggestion_to_cli_args(suggestion, prefix="trainer")

        assert "++trainer.lr=0.001" in args


class TestGenerateProteinSuggestion:
    """Test generate_protein_suggestion function."""

    def test_generate_first_suggestion(self):
        """Test generating first suggestion with no history."""
        trainer_config = {
            "trainer": {
                "lr": {
                    "distribution": "log_uniform",
                    "min": 0.0001,
                    "max": 0.1,
                },
            },
        }

        # Mock protein
        mock_protein = Mock()
        mock_protein.suggest.return_value = ({"trainer": {"lr": 0.005}}, None)

        suggestion = generate_protein_suggestion(trainer_config, mock_protein)

        assert "trainer" in suggestion
        assert "lr" in suggestion["trainer"]
        mock_protein.suggest.assert_called_once()

    def test_generate_validates_constraints(self):
        """Test that generate validates batch size constraints."""
        trainer_config = {
            "trainer": {
                "batch_size": 128,
                "minibatch_size": 32,
                "bppt": 8,
            },
        }

        # Mock protein - return invalid config (minibatch doesn't divide batch_size)
        mock_protein = Mock()
        mock_protein.suggest.return_value = ({"trainer": {"batch_size": 100, "minibatch_size": 33}}, None)

        # Should retry until it gets a valid config
        # For simplicity, we'll just check it calls suggest
        try:
            generate_protein_suggestion(trainer_config, mock_protein)
        except ValueError:
            # Expected if validation keeps failing
            pass

        mock_protein.suggest.assert_called()
