"""Tests for MettaProtein class."""

from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


class TestMettaProtein:
    """Test cases for MettaProtein class."""

    def test_metta_protein_init_with_full_config(self):
        """Test MettaProtein initialization with complete config."""
        config = OmegaConf.create(
            {
                "sweep": {
                    "protein": {
                        "max_suggestion_cost": 7200,
                        "num_random_samples": 100,
                        "global_search_scale": 2,
                    },
                    "parameters": {
                        "metric": "reward",
                        "goal": "maximize",
                        "trainer": {
                            "optimizer": {
                                "learning_rate": {
                                    "distribution": "log_normal",
                                    "min": 1e-5,
                                    "max": 1e-2,
                                    "scale": "auto",
                                    "mean": 3e-4,
                                }
                            }
                        },
                    },
                }
            }
        )

        mock_wandb_run = Mock()

        with patch("metta.sweep.protein_metta.Protein") as mock_protein:
            mock_protein_instance = Mock()
            mock_protein.return_value = mock_protein_instance

            with patch("metta.sweep.protein_metta.WandbProtein.__init__", return_value=None):
                _ = MettaProtein(config, mock_wandb_run)

                # Verify Protein was called with correct parameters
                mock_protein.assert_called_once()
                args, kwargs = mock_protein.call_args

                # Check that parameters were passed correctly
                protein_config = args[0]
                assert protein_config["metric"] == "reward"
                assert protein_config["goal"] == "maximize"
                assert "trainer" in protein_config

                # Check protein-specific parameters
                assert args[1] == 7200  # max_suggestion_cost
                assert args[3] == 100  # num_random_samples
                assert args[4] == 2  # global_search_scale

    def test_metta_protein_init_with_defaults(self):
        """Test MettaProtein initialization with minimal config (using defaults)."""
        config = OmegaConf.create(
            {
                "sweep": {
                    "protein": {},  # Empty protein config - should use defaults
                    "parameters": {
                        "metric": "accuracy",
                        "goal": "minimize",
                        "batch_size": {"distribution": "uniform", "min": 16, "max": 128, "scale": "auto", "mean": 64},
                    },
                }
            }
        )

        mock_wandb_run = Mock()

        with patch("metta.sweep.protein_metta.Protein") as mock_protein:
            mock_protein_instance = Mock()
            mock_protein.return_value = mock_protein_instance

            with patch("metta.sweep.protein_metta.WandbProtein.__init__", return_value=None):
                _ = MettaProtein(config, mock_wandb_run)

                # Verify Protein was called with defaults
                mock_protein.assert_called_once()
                args, kwargs = mock_protein.call_args

                # Check that parameters were passed correctly
                protein_config = args[0]
                assert protein_config["metric"] == "accuracy"
                assert protein_config["goal"] == "minimize"

                # Check defaults were used
                assert args[1] == 3600  # max_suggestion_cost default
                assert args[3] == 50  # num_random_samples default
                assert args[4] == 1  # global_search_scale default

    def test_metta_protein_config_interpolation(self):
        """Test that OmegaConf interpolations are resolved correctly."""
        config = OmegaConf.create(
            {
                "sweep": {
                    "protein": {},
                    "parameters": {
                        "metric": "loss",
                        "goal": "minimize",
                        "trainer": {
                            "batch_size": 32,
                            "optimizer": {
                                "learning_rate": {
                                    "distribution": "log_normal",
                                    "min": 1e-5,
                                    "max": 1e-2,
                                    "scale": "auto",
                                    "mean": "${sweep.parameters.batch_size}",
                                }
                            },
                        },
                    },
                }
            }
        )

        mock_wandb_run = Mock()

        with patch("metta.sweep.protein_metta.Protein") as mock_protein:
            mock_protein_instance = Mock()
            mock_protein.return_value = mock_protein_instance

            with patch("metta.sweep.protein_metta.WandbProtein.__init__", return_value=None):
                _ = MettaProtein(config, mock_wandb_run)

                # Verify interpolation was resolved
                mock_protein.assert_called_once()
                args, _ = mock_protein.call_args
                protein_config = args[0]

                # Check that interpolation was resolved
                assert protein_config["trainer"]["optimizer"]["learning_rate"]["mean"] == 32

    def test_transform_suggestion(self):
        """Test the _transform_suggestion method."""
        config = OmegaConf.create({"sweep": {"protein": {}, "parameters": {"metric": "reward", "goal": "maximize"}}})

        mock_wandb_run = Mock()

        with patch("metta.sweep.protein_metta.Protein") as mock_protein:
            mock_protein_instance = Mock()
            mock_protein.return_value = mock_protein_instance

            with patch("metta.sweep.protein_metta.WandbProtein.__init__", return_value=None):
                metta_protein = MettaProtein(config, mock_wandb_run)

                # Test numpy type conversion
                import numpy as np

                suggestion_with_numpy = {
                    "learning_rate": np.float64(0.001),
                    "batch_size": np.int32(64),
                    "nested": {"value": np.array([1, 2, 3])},
                }

                result = metta_protein._transform_suggestion(suggestion_with_numpy)

                # Check that numpy types were converted
                assert isinstance(result["learning_rate"], float)
                assert isinstance(result["batch_size"], int)
                assert isinstance(result["nested"]["value"], list)
