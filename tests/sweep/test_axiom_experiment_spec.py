"""Tests for spec-driven AxiomExperiment."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from metta.sweep.axiom.core import Pipeline
from metta.sweep.axiom.experiment import AxiomExperiment, RunHandle
from metta.sweep.axiom.experiment_spec import (
    AxiomControls,
    ComparisonExperimentSpec,
    ExperimentSpec,
    TrainingExperimentSpec,
    load_experiment_spec,
    save_experiment_spec,
)
from metta.sweep.axiom.manifest import diff_manifests


def simple_pipeline_factory(config: dict) -> Pipeline:
    """Simple pipeline factory for testing."""
    def init(_):
        return {"config": config, "initialized": True}
    
    def process(data):
        data["processed"] = True
        return data
    
    return (
        Pipeline()
        .stage("init", init)
        .stage("process", process)
    )


def test_experiment_spec_creation():
    """Test creating different types of experiment specs."""
    # Basic spec
    spec = ExperimentSpec(
        name="test_exp",
        description="Test experiment",
        pipeline_config={"key": "value"},
    )
    assert spec.name == "test_exp"
    assert spec.pipeline_config["key"] == "value"
    
    # Training spec with defaults
    train_spec = TrainingExperimentSpec(
        name="training_test",
    )
    assert train_spec.pipeline_type == "training"
    assert "trainer.optimizer" in train_spec.exposed_joins
    
    # Comparison spec with single-factor enforcement
    comp_spec = ComparisonExperimentSpec(
        name="comparison_test",
    )
    assert comp_spec.controls.single_factor_enforce is True


def test_axiom_controls():
    """Test AxiomControls configuration."""
    # Default controls
    controls = AxiomControls()
    assert controls.seed == 42
    assert controls.enforce_determinism is True
    assert controls.single_factor_enforce is False
    
    # Custom controls
    controls = AxiomControls(
        seed=1337,
        enforce_determinism=False,
        single_factor_enforce=True,
    )
    assert controls.seed == 1337
    assert controls.enforce_determinism is False
    assert controls.single_factor_enforce is True


def test_spec_save_and_load():
    """Test saving and loading experiment specs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create spec
        original_spec = TrainingExperimentSpec(
            name="save_test",
            description="Test saving",
            pipeline_config={
                "trainer": {"total_timesteps": 1000},
            },
            exposed_joins=["trainer.optimizer"],
        )
        
        # Save as JSON
        json_path = Path(tmpdir) / "spec.json"
        save_experiment_spec(original_spec, str(json_path))
        assert json_path.exists()
        
        # Load spec
        loaded_spec = load_experiment_spec(str(json_path))
        assert loaded_spec.name == "save_test"
        assert loaded_spec.description == "Test saving"
        assert loaded_spec.pipeline_config["trainer"]["total_timesteps"] == 1000
        assert "trainer.optimizer" in loaded_spec.exposed_joins


def test_basic_experiment_run():
    """Test basic experiment execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create spec
        spec = ExperimentSpec(
            name="basic_test",
            pipeline_config={"test": "value"},
            run_dir=tmpdir,
        )
        
        # Create experiment with custom factory
        exp = AxiomExperiment(spec, pipeline_factory=simple_pipeline_factory)
        
        # Prepare
        exp.prepare()
        assert os.environ.get("PYTHONHASHSEED") == "42"
        
        # Run
        result = exp.run("test_run")
        assert isinstance(result, RunHandle)
        assert result.tag == "test_run"
        
        # Check manifest
        manifest = result.manifest()
        assert manifest["experiment"] == "basic_test"
        assert manifest["tag"] == "test_run"
        
        # Check files were created
        manifest_path = Path(tmpdir) / "test_run.manifest.json"
        spec_path = Path(tmpdir) / "test_run.spec.json"
        assert manifest_path.exists()
        assert spec_path.exists()


def test_deterministic_seeding():
    """Test deterministic seed fan-out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = ExperimentSpec(
            name="seed_test",
            controls=AxiomControls(seed=999, enforce_determinism=True),
            run_dir=tmpdir,
        )
        
        exp = AxiomExperiment(spec, pipeline_factory=simple_pipeline_factory)
        exp.prepare()
        
        # Check seeds were set
        assert os.environ["PYTHONHASHSEED"] == "999"
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


def test_override_joins():
    """Test overriding joins at runtime."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = TrainingExperimentSpec(
            name="join_test",
            exposed_joins=["trainer.optimizer"],
            provided_joins={"trainer.optimizer": "adam"},
            run_dir=tmpdir,
        )
        
        exp = AxiomExperiment(spec, pipeline_factory=simple_pipeline_factory)
        exp.prepare()
        
        # Run with default
        default_run = exp.run("default")
        default_manifest = default_run.manifest()
        assert default_manifest["joins"]["implementations"]["trainer.optimizer"] == "adam"
        
        # Run with override
        sgd_run = exp.run(
            "sgd_variant",
            override_joins={"trainer.optimizer": "sgd"}
        )
        sgd_manifest = sgd_run.manifest()
        assert sgd_manifest["joins"]["implementations"]["trainer.optimizer"] == "sgd"


def test_manifest_diff():
    """Test manifest diffing between runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = ExperimentSpec(
            name="diff_test",
            controls=AxiomControls(seed=1),
            run_dir=tmpdir,
        )
        
        # Run baseline
        exp1 = AxiomExperiment(spec, pipeline_factory=simple_pipeline_factory)
        exp1.prepare()
        baseline = exp1.run("baseline")
        
        # Run variant with different seed
        spec2 = ExperimentSpec(
            name="diff_test",
            controls=AxiomControls(seed=2),
            run_dir=tmpdir,
        )
        exp2 = AxiomExperiment(spec2, pipeline_factory=simple_pipeline_factory)
        exp2.prepare()
        variant = exp2.run("variant")
        
        # Compute diff
        diff_str = exp1.diff(baseline, variant)
        assert "Controls" in diff_str or "controls" in diff_str
        assert "seed" in diff_str


def test_single_factor_enforcement():
    """Test single-factor enforcement in comparisons."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = ComparisonExperimentSpec(
            name="single_factor_test",
            exposed_joins=["trainer.optimizer", "trainer.advantage"],
            run_dir=tmpdir,
        )
        
        exp = AxiomExperiment(spec, pipeline_factory=simple_pipeline_factory)
        
        # This should pass (one change)
        try:
            exp._assert_single_factor_change(
                {"trainer.optimizer": "adam"},
                {"trainer.optimizer": "sgd"},
            )
        except ValueError:
            pytest.fail("Single factor check should pass with one change")
        
        # This should fail (two changes)
        with pytest.raises(ValueError, match="Single-factor enforcement failed"):
            exp._assert_single_factor_change(
                {"trainer.optimizer": "adam"},
                {"trainer.optimizer": "sgd", "trainer.advantage": "vtrace"},
            )


def test_run_comparison():
    """Test running baseline and variants."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = TrainingExperimentSpec(
            name="comparison_test",
            exposed_joins=["trainer.optimizer"],
            run_dir=tmpdir,
        )
        
        exp = AxiomExperiment(spec, pipeline_factory=simple_pipeline_factory)
        
        # Run comparison
        results = exp.run_comparison(
            baseline_joins={"trainer.optimizer": "adam"},
            variants={
                "sgd": {"trainer.optimizer": "sgd"},
                "muon": {"trainer.optimizer": "muon"},
            }
        )
        
        assert len(results) == 3
        assert "baseline" in results
        assert "sgd" in results
        assert "muon" in results
        
        # Check each has correct optimizer
        assert results["baseline"].manifest()["joins"]["implementations"].get("trainer.optimizer") == "adam"
        assert results["sgd"].manifest()["joins"]["implementations"].get("trainer.optimizer") == "sgd"
        assert results["muon"].manifest()["joins"]["implementations"].get("trainer.optimizer") == "muon"


def test_spec_inheritance():
    """Test creating specialized specs through composition."""
    base_config = {
        "trainer": {"num_workers": 4},
        "wandb": {"project": "test"},
    }
    
    # Fast iteration spec
    fast_spec = TrainingExperimentSpec(
        name="fast",
        pipeline_config={
            **base_config,
            "trainer": {
                **base_config["trainer"],
                "total_timesteps": 100,
            }
        }
    )
    
    # Production spec
    prod_spec = TrainingExperimentSpec(
        name="production",
        pipeline_config={
            **base_config,
            "trainer": {
                **base_config["trainer"],
                "total_timesteps": 1000000,
            }
        }
    )
    
    # Both inherit base config
    assert fast_spec.pipeline_config["trainer"]["num_workers"] == 4
    assert prod_spec.pipeline_config["trainer"]["num_workers"] == 4
    
    # But have different timesteps
    assert fast_spec.pipeline_config["trainer"]["total_timesteps"] == 100
    assert prod_spec.pipeline_config["trainer"]["total_timesteps"] == 1000000


def test_environment_capture():
    """Test environment variable capture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set test env var
        os.environ["TEST_VAR"] = "test_value"
        
        spec = ExperimentSpec(
            name="env_test",
            controls=AxiomControls(
                capture_env_vars=["TEST_VAR", "NONEXISTENT"]
            ),
            run_dir=tmpdir,
        )
        
        exp = AxiomExperiment(spec, pipeline_factory=simple_pipeline_factory)
        exp.prepare()
        
        assert exp.captured_env["TEST_VAR"] == "test_value"
        assert exp.captured_env["NONEXISTENT"] is None
        
        # Run and check manifest
        result = exp.run("test")
        manifest = result.manifest()
        assert manifest["environment"]["captured_vars"]["TEST_VAR"] == "test_value"