#!/usr/bin/env python3
"""Integration tests for MettaProtein sweep functionality."""

import pytest
import yaml
from omegaconf import OmegaConf

from metta.rl.carbs.metta_protein import MettaProtein


class MockWandbRun:
    """Mock WandB run for testing."""

    def __init__(self, name: str = "test_protein_run"):
        self.name = name


def coerce_numbers(d):
    """Convert string numbers to proper numeric types."""
    if isinstance(d, dict):
        return {k: coerce_numbers(v) for k, v in d.items()}
    elif isinstance(d, str):
        try:
            if "." in d or "e" in d:
                return float(d)
            else:
                return int(d)
        except ValueError:
            return d
    else:
        return d


class TestMettaProteinIntegration:
    """Test suite for MettaProtein integration with sweep workflow."""

    def test_basic_functionality(self):
        """Test basic MettaProtein functionality."""
        cfg = OmegaConf.create(
            {
                "learning_rate": {"min": 0.001, "max": 0.01, "scale": 1, "mean": 0.005, "distribution": "log_normal"},
                "batch_size": {"min": 16, "max": 128, "scale": 1, "mean": 64, "distribution": "int_uniform"},
            }
        )

        mock_run = MockWandbRun()
        protein = MettaProtein(cfg, wandb_run=mock_run)

        # Test suggest method
        params = protein.suggest()
        assert isinstance(params, dict)
        assert "learning_rate" in params
        assert "batch_size" in params
        assert 0.001 <= params["learning_rate"] <= 0.01
        assert 16 <= params["batch_size"] <= 128

        # Test observe method
        protein.observe(params, score=0.85, cost=120.5)
        assert len(protein.success_observations) == 1

    def test_wandb_stubs(self):
        """Test WandB stub functionality."""
        mock_run = MockWandbRun("test_stub_run")

        # Test static method
        MettaProtein._record_observation(mock_run, 0.75, 95.0)

        # Test failure recording
        cfg = OmegaConf.create({"param": {"min": 0, "max": 1, "scale": 1, "mean": 0.5, "distribution": "uniform"}})
        protein = MettaProtein(cfg, wandb_run=mock_run)
        protein.record_failure()

    def test_config_format_compatibility(self):
        """Test that protein config format works correctly."""
        try:
            with open("configs/sweep/minimal_protein_sweep.yaml") as f:
                protein_cfg = yaml.safe_load(f)["sweep"]
        except FileNotFoundError:
            pytest.skip("minimal_protein_sweep.yaml not found")

        protein_cfg = coerce_numbers(protein_cfg)
        protein_cfg = OmegaConf.create(protein_cfg)

        protein = MettaProtein(protein_cfg)
        params = protein.suggest()

        assert isinstance(params, dict)
        assert len(params) > 0

    def test_sweep_init_integration(self):
        """Test integration with sweep_init functionality."""
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "../../tools"))

        try:
            from sweep_init import apply_protein_suggestion
        except ImportError:
            pytest.skip("sweep_init module not available")

        # Test the function
        test_cfg = OmegaConf.create({"learning_rate": 0.001, "batch_size": 64})
        test_suggestion = {"learning_rate": 0.005, "batch_size": 32}

        apply_protein_suggestion(test_cfg, test_suggestion)

        assert test_cfg.learning_rate == 0.005
        assert test_cfg.batch_size == 32

    def test_multiple_observations(self):
        """Test protein with multiple observations."""
        cfg = OmegaConf.create({"x": {"min": 0.0, "max": 1.0, "scale": 1, "mean": 0.5, "distribution": "uniform"}})

        protein = MettaProtein(cfg)

        # Add multiple observations
        for i in range(5):
            params = protein.suggest()
            score = 1.0 - abs(params["x"] - 0.7)  # Target 0.7
            protein.observe(params, score=score, cost=10.0)

        assert len(protein.success_observations) == 5

        # Next suggestion should be learning from observations
        final_params = protein.suggest()
        assert isinstance(final_params, dict)
        assert "x" in final_params
