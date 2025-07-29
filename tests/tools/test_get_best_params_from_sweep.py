"""Tests for get_best_params_from_sweep.py tool."""

from unittest.mock import MagicMock, mock_open, patch

from tools.get_best_params_from_sweep import (
    extract_hyperparameters_from_run,
    format_hyperparameters_yaml,
    generate_config_patch,
    generate_override_args,
)


class TestExtractHyperparametersFromRun:
    """Test hyperparameter extraction from WandB runs."""

    def test_extract_hyperparameters_from_suggestion(self):
        """Test extracting hyperparameters from protein.suggestion."""
        run = MagicMock()
        run.summary = {
            "protein.suggestion": {"trainer": {"learning_rate": 0.001, "batch_size": 64}, "agent": {"hidden_size": 256}}
        }
        run.config = {}  # Config should be ignored when suggestion exists

        params = extract_hyperparameters_from_run(run)

        assert params == {"trainer": {"learning_rate": 0.001, "batch_size": 64}, "agent": {"hidden_size": 256}}

    def test_extract_hyperparameters_from_config_fallback(self):
        """Test extracting hyperparameters from config when suggestion is missing."""
        run = MagicMock()
        run.summary = {}  # No protein.suggestion
        run.config = {
            "trainer": {"learning_rate": 0.001, "batch_size": 64},
            "_wandb": {"some": "internal"},  # Should be filtered out
            "dummy_param": "ignored",  # Should be filtered out
        }

        params = extract_hyperparameters_from_run(run)

        assert params == {"trainer": {"learning_rate": 0.001, "batch_size": 64}}

    def test_extract_hyperparameters_empty_suggestion(self):
        """Test handling empty suggestion field."""
        run = MagicMock()
        run.summary = {"protein.suggestion": {}}
        run.config = {"trainer": {"learning_rate": 0.001}}

        params = extract_hyperparameters_from_run(run)

        # Should fallback to config
        assert params == {"trainer": {"learning_rate": 0.001}}


class TestFormatHyperparametersYaml:
    """Test YAML formatting of hyperparameters."""

    def test_format_simple_params(self):
        """Test formatting simple parameters."""
        params = {
            "learning_rate": 0.001,  # This will be formatted as 1.00e-03
            "batch_size": 64,
            "epochs": 10,
        }

        formatted = format_hyperparameters_yaml(params)

        assert "learning_rate: 1.00e-03" in formatted  # Scientific notation for < 0.01
        assert "batch_size: 64" in formatted
        assert "epochs: 10" in formatted

    def test_format_nested_params(self):
        """Test formatting nested parameters."""
        params = {
            "trainer": {
                "optimizer": {
                    "learning_rate": 0.1,  # This won't use scientific notation
                    "eps": 1e-5,
                },
                "batch_size": 64,
            }
        }

        formatted = format_hyperparameters_yaml(params)

        assert "trainer:" in formatted
        assert "  optimizer:" in formatted
        assert "    learning_rate: 0.1" in formatted
        assert "    eps: 1.00e-05" in formatted
        assert "  batch_size: 64" in formatted

    def test_format_with_lists(self):
        """Test formatting parameters with lists."""
        params = {"layers": [128, 256, 128], "dropout_rates": [0.1, 0.2, 0.1]}

        formatted = format_hyperparameters_yaml(params)

        assert "layers: [128, 256, 128]" in formatted  # Lists are formatted inline
        assert "dropout_rates: [0.1, 0.2, 0.1]" in formatted

    def test_format_with_indent(self):
        """Test formatting with custom indentation."""
        params = {"key": "value"}

        formatted = format_hyperparameters_yaml(params, indent=4)

        assert "    key: value" in formatted


class TestGenerateConfigPatch:
    """Test config patch file generation."""

    @patch("builtins.open", new_callable=mock_open)
    def test_generate_config_patch(self, mock_file):
        """Test generating a config patch file."""
        params = {"trainer": {"learning_rate": 0.001, "batch_size": 64}}

        generate_config_patch(params, "/output/patch.yaml")

        mock_file.assert_called_once_with("/output/patch.yaml", "w")
        handle = mock_file()
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        assert "# Best hyperparameters from sweep" in written_content
        assert "trainer:" in written_content
        assert "learning_rate: 1.00e-03" in written_content  # Scientific notation


class TestGenerateOverrideArgs:
    """Test command-line override argument generation."""

    def test_generate_override_args_simple(self):
        """Test generating simple override arguments."""
        params = {
            "learning_rate": 0.001,  # Will be formatted as scientific notation
            "batch_size": 64,
        }

        args = generate_override_args(params)

        assert "learning_rate=1.00e-03" in args  # Scientific notation
        assert "batch_size=64" in args

    def test_generate_override_args_nested(self):
        """Test generating nested override arguments."""
        params = {
            "trainer": {
                "optimizer": {
                    "learning_rate": 0.001  # Will be formatted as scientific notation
                },
                "batch_size": 64,
            }
        }

        args = generate_override_args(params)

        assert "trainer.optimizer.learning_rate=1.00e-03" in args  # Scientific notation
        assert "trainer.batch_size=64" in args

    def test_generate_override_args_with_lists(self):
        """Test generating override arguments with lists."""
        params = {"layers": [128, 256, 128]}

        args = generate_override_args(params)

        assert "layers=[128, 256, 128]" in args

    def test_generate_override_args_with_prefix(self):
        """Test generating override arguments with prefix."""
        params = {
            "learning_rate": 0.001  # Will be formatted as scientific notation
        }

        args = generate_override_args(params, prefix="model")

        assert "model.learning_rate=1.00e-03" in args  # Scientific notation


# Note: Testing the main() function directly is challenging because it's decorated
# with @hydra.main and @metta_script, which require specific configuration setup.
# The individual functions are thoroughly tested above, which provides good coverage
# of the actual logic in the script.
