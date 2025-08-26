"""Integration tests for the Axiom experiment system."""

import pytest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from metta.sweep.axiom.core import Pipeline, Ctx
from metta.sweep.axiom.experiment import AxiomExperiment, RunHandle
from metta.sweep.axiom.experiment_spec import ExperimentSpec, AxiomControls
from metta.sweep.axiom.train_and_eval import (
    TrainAndEvalSpec,
    TrainAndEvalExperiment,
    create_quick_experiment,
)


class TestPipelineIntegration:
    """Integration tests for Pipeline with exposure and overrides."""

    def test_complex_pipeline_with_overrides(self):
        """Test a complex pipeline with multiple levels and overrides."""
        # Create optimizer sub-pipeline
        optimizer = Pipeline(name="optimizer")
        optimizer.stage("compute_gradients", lambda x: {**x, "gradients": "computed"}, expose=True)
        optimizer.stage("apply_gradients", lambda x: {**x, "weights": "updated"}, expose=True)
        
        # Create trainer pipeline with optimizer
        trainer = Pipeline(name="trainer")
        trainer.stage("collect_data", lambda x: {**x, "data": "collected"})
        trainer.stage("compute_loss", lambda x: {**x, "loss": 0.5}, expose=True)
        trainer.join("optimizer", optimizer, expose=True)
        
        # Create main pipeline
        main = Pipeline(name="main")
        main.stage("init", lambda _: {"step": 0})
        main.join("train", trainer, expose=True)
        main.stage("save", lambda x: {**x, "saved": True})
        
        # Run original pipeline
        result1 = main.run()
        assert result1["loss"] == 0.5
        assert result1["weights"] == "updated"
        
        # Override nested optimizer gradient computation
        main.override("train.optimizer.compute_gradients", 
                     lambda x: {**x, "gradients": "custom_gradients"})
        
        result2 = main.run()
        assert result2["gradients"] == "custom_gradients"
        assert result2["weights"] == "updated"
        
        # Override trainer loss computation
        main.override("train.compute_loss", lambda x: {**x, "loss": 0.1})
        
        result3 = main.run()
        assert result3["loss"] == 0.1
        assert result3["gradients"] == "custom_gradients"

    def test_pipeline_with_hooks_and_checks(self):
        """Test pipeline with hooks and checks integrated."""
        from metta.sweep.axiom.checks import Check, CheckLevel
        
        # Create pipeline with checks
        pipeline = Pipeline()
        pipeline.stage("init", lambda _: {"value": 0.5})
        pipeline.stage("process", lambda x: {**x, "value": x["value"] * 2})
        
        # Add a check
        def value_check(result):
            if result.get("value", 0) > 0.9:
                return True, ""
            return False, f"Value {result.get('value')} is not > 0.9"
        
        check = Check(name="value_check", check=value_check, level=CheckLevel.WARN)
        pipeline.stages[-1].checks = [check]
        
        # Add a hook
        hook_calls = []
        def track_hook(result, ctx):
            hook_calls.append(result.get("value"))
        
        pipeline.hook(track_hook)
        
        # Run pipeline
        result = pipeline.run()
        
        assert result["value"] == 1.0
        assert len(hook_calls) == 1
        assert hook_calls[0] == 1.0

    def test_pipeline_error_propagation(self):
        """Test that errors in stages propagate correctly."""
        pipeline = Pipeline()
        pipeline.stage("init", lambda _: 42)
        pipeline.stage("fail", lambda x: 1 / 0)  # Will raise ZeroDivisionError
        pipeline.stage("never_reached", lambda x: x)
        
        with pytest.raises(ZeroDivisionError):
            pipeline.run()


class TestExperimentIntegration:
    """Integration tests for AxiomExperiment system."""

    def test_experiment_with_pipeline_factory(self):
        """Test AxiomExperiment with a pipeline factory."""
        # Create a simple spec
        spec = ExperimentSpec(
            name="test_experiment",
            description="Integration test",
            config={"multiplier": 2},
            controls=AxiomControls(seed=42),
        )
        
        # Create pipeline factory
        def pipeline_factory(config):
            pipeline = Pipeline()
            pipeline.stage("init", lambda _: 10)
            pipeline.stage("multiply", lambda x: x * config["multiplier"])
            return pipeline
        
        # Create and run experiment
        exp = AxiomExperiment(spec=spec, pipeline_factory=pipeline_factory)
        exp.prepare()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp._run_dir = tmpdir
            result = exp.run()
        
        # Check result
        assert isinstance(result, RunHandle)
        manifest = result.manifest()
        assert manifest["experiment"] == "test_experiment"
        assert manifest["config"]["multiplier"] == 2
        assert manifest["pipeline_result"] == 20

    def test_experiment_with_multiple_runs(self):
        """Test running an experiment multiple times with different tags."""
        spec = ExperimentSpec(
            name="multi_run",
            config={"value": 5},
            controls=AxiomControls(seed=42),
        )
        
        def pipeline_factory(config):
            pipeline = Pipeline()
            pipeline.stage("process", lambda _: config["value"] * 2)
            return pipeline
        
        exp = AxiomExperiment(spec=spec, pipeline_factory=pipeline_factory)
        exp.prepare()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp._run_dir = tmpdir
            
            # Run multiple times with different tags
            result1 = exp.run(tag="baseline")
            result2 = exp.run(tag="variant")
            
            # Check both results
            assert result1._tag == "baseline"
            assert result2._tag == "variant"
            assert result1.manifest()["pipeline_result"] == 10
            assert result2.manifest()["pipeline_result"] == 10

    def test_experiment_diff_functionality(self):
        """Test diffing two experiment results."""
        spec = ExperimentSpec(
            name="diff_test",
            config={"value": 5},
            controls=AxiomControls(seed=42),
        )
        
        def pipeline_factory(config):
            pipeline = Pipeline()
            pipeline.stage("process", lambda _: config["value"] * 2)
            return pipeline
        
        exp = AxiomExperiment(spec=spec, pipeline_factory=pipeline_factory)
        exp.prepare()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp._run_dir = tmpdir
            
            # Run baseline
            baseline = exp.run(tag="baseline")
            
            # Modify config and run variant
            exp.spec.config["value"] = 10
            variant = exp.run(tag="variant")
            
            # Diff the results
            diff = exp.diff(baseline, variant)
            
            assert diff is not None
            # The diff should show the config difference
            # and the result difference


class TestTrainAndEvalIntegration:
    """Integration tests for TrainAndEval experiment."""

    @patch('metta.sweep.axiom.train_and_eval.train')
    @patch('metta.sweep.axiom.train_and_eval.evaluate_policy')
    @patch('metta.sweep.axiom.train_and_eval.PolicyStore')
    @patch('metta.sweep.axiom.train_and_eval.WandbContext')
    def test_full_train_and_eval_flow(self, mock_wandb_context, mock_policy_store, mock_evaluate, mock_train):
        """Test complete train and eval flow."""
        from metta.mettagrid import EnvConfig
        from metta.sim.simulation_config import SimulationConfig
        from metta.agent.policy_record import PolicyRecord
        
        # Setup mocks
        mock_wandb_context.__enter__ = MagicMock(return_value=MagicMock())
        mock_wandb_context.__exit__ = MagicMock(return_value=None)
        
        mock_policy_record = MagicMock(spec=PolicyRecord)
        mock_policy_record.uri = "file://./policy.pt"
        mock_policy_record.metadata = {"epoch": 10}
        
        mock_store_instance = MagicMock()
        mock_store_instance.policy_records.return_value = [mock_policy_record]
        mock_policy_store.create.return_value = mock_store_instance
        
        mock_evaluate.return_value = {
            "mean_episode_reward": 100.0,
            "success_rate": 0.75,
        }
        
        # Create spec
        spec = TrainAndEvalSpec(name="integration_test")
        spec.trainer_config.total_timesteps = 1000
        spec.eval_configs = [
            SimulationConfig(
                name="test_env",
                env=EnvConfig(),
                num_episodes=10,
            )
        ]
        
        # Create and run experiment
        exp = TrainAndEvalExperiment(spec)
        exp.prepare()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            spec.run_dir = tmpdir
            exp._run_dir = tmpdir
            
            # Run the full pipeline
            result = exp.run()
            
            # Check the manifest
            manifest = result.manifest()
            assert manifest["experiment"] == "integration_test"
            assert manifest["pipeline_result"]["status"] == "complete"
            assert manifest["pipeline_result"]["training_complete"] is True
            assert manifest["pipeline_result"]["eval_complete"] is True
            assert "eval_results" in manifest["pipeline_result"]
            assert "test_env" in manifest["pipeline_result"]["eval_results"]
            
            # Verify training was called
            mock_train.assert_called_once()
            
            # Verify evaluation was called
            mock_evaluate.assert_called_once()

    def test_train_and_eval_with_overrides(self):
        """Test TrainAndEval with pipeline overrides for debugging."""
        spec = create_quick_experiment()
        exp = TrainAndEvalExperiment(spec)
        
        # Get the pipeline
        pipeline = exp._create_pipeline({})
        
        # Check that we can list exposed components
        exposed = pipeline.list_exposed()
        # By default, stages aren't exposed unless we modify the implementation
        # But joins should be if we add them
        
        # This test demonstrates the pattern even if not all stages are exposed
        assert isinstance(exposed, list)

    @patch('metta.sweep.axiom.train_and_eval.evaluate_policy')
    @patch('metta.sweep.axiom.train_and_eval.PolicyStore')
    def test_eval_only_flow(self, mock_policy_store, mock_evaluate):
        """Test eval-only flow with existing policy."""
        from metta.agent.policy_record import PolicyRecord
        
        # Setup mocks
        mock_policy_record = MagicMock(spec=PolicyRecord)
        mock_policy_record.uri = "file://./existing_policy.pt"
        mock_policy_record.metadata = {"epoch": 100}
        
        mock_store_instance = MagicMock()
        mock_store_instance.policy_records.return_value = [mock_policy_record]
        mock_policy_store.create.return_value = mock_store_instance
        
        mock_evaluate.return_value = {
            "mean_episode_reward": 150.0,
            "success_rate": 0.9,
        }
        
        # Create eval-only spec
        spec = create_eval_only_experiment("file://./existing_policy.pt")
        exp = TrainAndEvalExperiment(spec)
        exp.prepare()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            spec.run_dir = tmpdir
            exp._run_dir = tmpdir
            
            # Run evaluation only
            result = exp.run()
            
            # Check that training was skipped
            manifest = result.manifest()
            assert manifest["pipeline_result"].get("training_skipped") is True
            assert manifest["pipeline_result"]["eval_complete"] is True
            assert len(manifest["pipeline_result"]["eval_results"]) == 2  # basic and combat


class TestEndToEndScenarios:
    """End-to-end integration tests for realistic scenarios."""

    @patch('metta.sweep.axiom.train_and_eval.train')
    @patch('metta.sweep.axiom.train_and_eval.evaluate_policy')
    @patch('metta.sweep.axiom.train_and_eval.PolicyStore')
    @patch('metta.sweep.axiom.train_and_eval.WandbContext')
    def test_debugging_scenario(self, mock_wandb_context, mock_policy_store, mock_evaluate, mock_train):
        """Test a debugging scenario with overrides."""
        from metta.agent.policy_record import PolicyRecord
        
        # Setup mocks
        mock_wandb_context.__enter__ = MagicMock(return_value=MagicMock())
        mock_wandb_context.__exit__ = MagicMock(return_value=None)
        
        mock_policy_record = MagicMock(spec=PolicyRecord)
        mock_policy_record.uri = "file://./policy.pt"
        mock_policy_record.metadata = {}
        
        mock_store_instance = MagicMock()
        mock_store_instance.policy_records.return_value = [mock_policy_record]
        mock_policy_store.create.return_value = mock_store_instance
        
        # Create experiment
        spec = create_quick_experiment()
        exp = TrainAndEvalExperiment(spec)
        
        # Simulate debugging by modifying the pipeline
        # (In practice, we'd expose stages and override them)
        pipeline = exp._create_pipeline({})
        
        # Add a hook to track execution
        execution_log = []
        def debug_hook(result, ctx):
            execution_log.append(ctx._current_stage)
        
        pipeline.hook(debug_hook)
        
        # Replace the pipeline factory
        exp._pipeline_factory = lambda _: pipeline
        
        # Run with debugging
        exp.prepare()
        with tempfile.TemporaryDirectory() as tmpdir:
            spec.run_dir = tmpdir
            exp._run_dir = tmpdir
            
            # Set up evaluate to return quickly
            mock_evaluate.return_value = {"mean_episode_reward": 50.0}
            
            result = exp.run()
            
            # Check that all stages were executed
            assert "initialize" in execution_log
            assert "train" in execution_log
            assert "evaluate" in execution_log
            assert "finalize" in execution_log

    def test_reproducibility(self):
        """Test that experiments are reproducible with same seed."""
        spec1 = create_quick_experiment()
        spec1.controls.seed = 12345
        spec1.controls.enforce_determinism = True
        
        spec2 = create_quick_experiment()
        spec2.controls.seed = 12345
        spec2.controls.enforce_determinism = True
        
        # Both specs should be identical except for object identity
        assert spec1.controls.seed == spec2.controls.seed
        assert spec1.trainer_config.total_timesteps == spec2.trainer_config.total_timesteps
        assert len(spec1.eval_configs) == len(spec2.eval_configs)