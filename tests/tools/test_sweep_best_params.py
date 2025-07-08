"""Tests for sweep_best_params.py tool."""

from unittest.mock import MagicMock, mock_open, patch

from omegaconf import OmegaConf

from tools.sweep_best_params import (
    extract_hyperparameters_from_run,
    format_hyperparameters_yaml,
    generate_config_patch,
    generate_override_args,
    get_sweep_runs,
    load_local_hyperparameters,
)


class TestGetSweepRuns:
    """Test sweep runs retrieval functionality."""

    def test_get_sweep_runs_success(self):
        """Test successful retrieval of sweep runs."""
        mock_api = MagicMock()
        mock_sweep = MagicMock()

        run1 = MagicMock()
        run1.summary = {"protein.state": "success", "score": 0.5}
        run1.config = {"trainer": {"learning_rate": 0.001}}
        run1.name = "run1"
        run1.id = "run1_id"

        run2 = MagicMock()
        run2.summary = {"protein.state": "success", "score": 0.8}
        run2.config = {"trainer": {"learning_rate": 0.0005}}
        run2.name = "run2"
        run2.id = "run2_id"

        # Add a failed run that should be filtered out
        run3 = MagicMock()
        run3.summary = {"protein.state": "failed", "score": 0.3}
        run3.config = {"trainer": {"learning_rate": 0.002}}
        run3.name = "run3"
        run3.id = "run3_id"

        mock_sweep.runs = [run1, run2, run3]
        mock_api.sweep.return_value = mock_sweep

        with patch("wandb.Api", return_value=mock_api):
            runs = get_sweep_runs("sweep123", "test_entity", "test_project")

        assert len(runs) == 2  # Only successful runs
        assert runs[0].name == "run2"  # Sorted by score descending
        assert runs[1].name == "run1"

    def test_get_sweep_runs_empty(self):
        """Test handling of sweep with no runs."""
        mock_api = MagicMock()
        mock_sweep = MagicMock()
        mock_sweep.runs = []
        mock_api.sweep.return_value = mock_sweep

        with patch("wandb.Api", return_value=mock_api):
            runs = get_sweep_runs("sweep123", "test_entity", "test_project")

        assert runs == []

    def test_get_sweep_runs_no_successful(self):
        """Test handling of sweep with no successful runs."""
        mock_api = MagicMock()
        mock_sweep = MagicMock()

        run1 = MagicMock()
        run1.summary = {"protein.state": "failed", "score": 0.5}
        run1.name = "run1"

        run2 = MagicMock()
        run2.summary = {"protein.state": "running"}  # No score
        run2.name = "run2"

        mock_sweep.runs = [run1, run2]
        mock_api.sweep.return_value = mock_sweep

        with patch("wandb.Api", return_value=mock_api):
            runs = get_sweep_runs("sweep123", "test_entity", "test_project")

        assert runs == []

    def test_get_sweep_runs_with_protein_objective(self):
        """Test handling runs that use protein.objective instead of score."""
        mock_api = MagicMock()
        mock_sweep = MagicMock()

        run1 = MagicMock()
        run1.summary = {"protein.state": "success", "protein.objective": 0.7}
        run1.config = {"trainer": {"learning_rate": 0.001}}
        run1.name = "run1"

        mock_sweep.runs = [run1]
        mock_api.sweep.return_value = mock_sweep

        with patch("wandb.Api", return_value=mock_api):
            runs = get_sweep_runs("sweep123", "test_entity", "test_project")

        assert len(runs) == 1
        assert runs[0].name == "run1"


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


class TestLoadLocalHyperparameters:
    """Test loading hyperparameters from local files."""

    @patch("pathlib.Path.exists")
    @patch("omegaconf.OmegaConf.load")
    def test_load_local_hyperparameters_success(self, mock_load, mock_exists):
        """Test successful loading from local file."""
        mock_exists.return_value = True
        mock_load.return_value = OmegaConf.create({"trainer": {"learning_rate": 0.001, "batch_size": 64}})

        params = load_local_hyperparameters("test_sweep", "test_sweep.r.0", "/data")

        assert params == {"trainer": {"learning_rate": 0.001, "batch_size": 64}}

        # Verify the correct path was used
        mock_load.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_load_local_hyperparameters_file_not_found(self, mock_exists):
        """Test handling of missing local file."""
        mock_exists.return_value = False

        params = load_local_hyperparameters("test_sweep", "test_sweep.r.0", "/data")

        # Should return empty dict and log warning
        assert params == {}


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
