"""Unit tests for TrainAndEval experiment implementation."""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch, call
from typing import Any

from metta.sweep.axiom.train_and_eval import (
    AxiomSpec,
    TrainAndEvalSpec,
    TrainAndEvalExperiment,
    create_quick_experiment,
    create_full_experiment,
    create_eval_only_experiment,
)
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.axiom.experiment_spec import AxiomControls
from metta.rl.system_config import SystemConfig
from metta.agent.agent_config import AgentConfig
from metta.common.wandb.wandb_context import WandbConfig


class TestAxiomSpec:
    """Test the base AxiomSpec class."""

    def test_axiom_spec_defaults(self):
        """Test AxiomSpec default values."""
        spec = AxiomSpec(name="test")
        
        assert spec.name == "test"
        assert spec.description == ""
        assert isinstance(spec.controls, AxiomControls)
        assert spec.run_dir == "./experiments"
        assert isinstance(spec.system_config, SystemConfig)
        assert isinstance(spec.agent_config, AgentConfig)
        assert isinstance(spec.wandb_config, WandbConfig)

    def test_axiom_spec_custom_values(self):
        """Test AxiomSpec with custom values."""
        controls = AxiomControls(seed=123, enforce_determinism=True)
        system = SystemConfig(device="cuda")
        agent = AgentConfig()
        
        spec = AxiomSpec(
            name="custom_test",
            description="A custom test",
            controls=controls,
            run_dir="/tmp/experiments",
            system_config=system,
            agent_config=agent,
        )
        
        assert spec.name == "custom_test"
        assert spec.description == "A custom test"
        assert spec.controls.seed == 123
        assert spec.run_dir == "/tmp/experiments"
        assert spec.system_config.device == "cuda"

    def test_axiom_spec_forbids_extra_fields(self):
        """Test that AxiomSpec forbids extra fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            AxiomSpec(name="test", unknown_field="value")


class TestTrainAndEvalSpec:
    """Test the TrainAndEvalSpec class."""

    def test_train_and_eval_spec_defaults(self):
        """Test TrainAndEvalSpec default values."""
        spec = TrainAndEvalSpec(name="test")
        
        assert spec.name == "test"
        assert isinstance(spec.trainer_config, TrainerConfig)
        assert spec.eval_configs == []
        assert spec.policy_path is None

    def test_train_and_eval_spec_with_configs(self):
        """Test TrainAndEvalSpec with configurations."""
        from metta.mettagrid import EnvConfig
        
        trainer = TrainerConfig()
        trainer.total_timesteps = 1000
        
        eval_configs = [
            SimulationConfig(
                name="test_sim",
                env=EnvConfig(),
                num_episodes=10,
            )
        ]
        
        spec = TrainAndEvalSpec(
            name="test",
            trainer_config=trainer,
            eval_configs=eval_configs,
            policy_path="file://./policy.pt",
        )
        
        assert spec.trainer_config.total_timesteps == 1000
        assert len(spec.eval_configs) == 1
        assert spec.eval_configs[0].name == "test_sim"
        assert spec.policy_path == "file://./policy.pt"


class TestTrainAndEvalExperiment:
    """Test the TrainAndEvalExperiment class."""

    def test_experiment_initialization(self):
        """Test TrainAndEvalExperiment initialization."""
        spec = TrainAndEvalSpec(name="test")
        exp = TrainAndEvalExperiment(spec)
        
        assert exp.spec == spec
        assert exp._pipeline_factory is not None
        assert hasattr(exp, '_create_pipeline')

    def test_create_pipeline(self):
        """Test pipeline creation."""
        spec = TrainAndEvalSpec(name="test")
        exp = TrainAndEvalExperiment(spec)
        
        pipeline = exp._create_pipeline({})
        
        # Check pipeline stages
        assert len(pipeline.stages) == 4
        stage_names = [s.name for s in pipeline.stages]
        assert "initialize" in stage_names
        assert "train" in stage_names
        assert "evaluate" in stage_names
        assert "finalize" in stage_names

    @patch('metta.sweep.axiom.train_and_eval.train')
    @patch('metta.sweep.axiom.train_and_eval.PolicyStore')
    @patch('metta.sweep.axiom.train_and_eval.WandbContext')
    def test_training_stage_execution(self, mock_wandb_context, mock_policy_store, mock_train):
        """Test the training stage execution."""
        # Setup mocks
        mock_wandb_context.__enter__ = MagicMock(return_value=MagicMock())
        mock_wandb_context.__exit__ = MagicMock(return_value=None)
        mock_policy_store.create.return_value = MagicMock()
        
        spec = TrainAndEvalSpec(name="test")
        spec.trainer_config.total_timesteps = 100
        exp = TrainAndEvalExperiment(spec)
        
        pipeline = exp._create_pipeline({})
        
        # Get the train stage function
        train_stage = None
        for stage in pipeline.stages:
            if stage.name == "train":
                train_stage = stage.func
                break
        
        assert train_stage is not None
        
        # Execute train stage
        with tempfile.TemporaryDirectory() as tmpdir:
            spec.run_dir = tmpdir
            state = {}
            result = train_stage(state)
        
        # Check results
        assert result["training_complete"] is True
        assert result["final_timestep"] == 100
        assert "policy_uri" in result
        assert result["policy_uri"].startswith("file://")
        assert "run_name" in result
        assert "run_dir" in result
        
        # Check that train was called
        mock_train.assert_called_once()

    def test_training_skipped_with_policy_path(self):
        """Test that training is skipped when policy_path is provided."""
        spec = TrainAndEvalSpec(
            name="test",
            policy_path="file://./existing_policy.pt"
        )
        exp = TrainAndEvalExperiment(spec)
        
        pipeline = exp._create_pipeline({})
        
        # Get the train stage function
        train_stage = None
        for stage in pipeline.stages:
            if stage.name == "train":
                train_stage = stage.func
                break
        
        # Execute train stage
        state = {}
        result = train_stage(state)
        
        # Check that training was skipped
        assert result["training_skipped"] is True
        assert result["policy_uri"] == "file://./existing_policy.pt"
        assert "training_complete" not in result

    @patch('metta.sweep.axiom.train_and_eval.evaluate_policy')
    @patch('metta.sweep.axiom.train_and_eval.PolicyStore')
    def test_evaluation_stage_execution(self, mock_policy_store, mock_evaluate):
        """Test the evaluation stage execution."""
        from metta.mettagrid import EnvConfig
        from metta.agent.policy_record import PolicyRecord
        
        # Setup mocks
        mock_policy_record = MagicMock(spec=PolicyRecord)
        mock_policy_record.uri = "file://./policy.pt"
        mock_policy_record.metadata = {}
        
        mock_store_instance = MagicMock()
        mock_store_instance.policy_records.return_value = [mock_policy_record]
        mock_policy_store.create.return_value = mock_store_instance
        
        mock_evaluate.return_value = {
            "mean_episode_reward": 42.0,
            "success_rate": 0.8,
        }
        
        # Create spec with eval configs
        spec = TrainAndEvalSpec(name="test")
        spec.eval_configs = [
            SimulationConfig(
                name="test_sim",
                env=EnvConfig(),
                num_episodes=5,
            )
        ]
        
        exp = TrainAndEvalExperiment(spec)
        pipeline = exp._create_pipeline({})
        
        # Get the eval stage function
        eval_stage = None
        for stage in pipeline.stages:
            if stage.name == "evaluate":
                eval_stage = stage.func
                break
        
        # Execute eval stage
        state = {"policy_uri": "file://./policy.pt"}
        result = eval_stage(state)
        
        # Check results
        assert result["eval_complete"] is True
        assert "eval_results" in result
        assert "test_sim" in result["eval_results"]
        assert result["eval_results"]["test_sim"]["mean_reward"] == 42.0
        assert result["eval_results"]["test_sim"]["episodes"] == 5

    def test_evaluation_fails_without_policy(self):
        """Test that evaluation fails without a policy URI."""
        from metta.mettagrid import EnvConfig
        
        spec = TrainAndEvalSpec(name="test")
        spec.eval_configs = [
            SimulationConfig(
                name="test_sim",
                env=EnvConfig(),
                num_episodes=5,
            )
        ]
        
        exp = TrainAndEvalExperiment(spec)
        pipeline = exp._create_pipeline({})
        
        # Get the eval stage function
        eval_stage = None
        for stage in pipeline.stages:
            if stage.name == "evaluate":
                eval_stage = stage.func
                break
        
        # Execute eval stage without policy_uri
        state = {}
        with pytest.raises(ValueError, match="No policy URI available"):
            eval_stage(state)


class TestFactoryFunctions:
    """Test the factory functions for creating experiments."""

    def test_create_quick_experiment(self):
        """Test create_quick_experiment factory."""
        spec = create_quick_experiment()
        
        assert spec.name == "quick_test"
        assert spec.description == "Quick training and evaluation test"
        assert spec.trainer_config.total_timesteps == 10000
        assert spec.trainer_config.rollout_workers == 2
        assert len(spec.eval_configs) == 1
        assert spec.eval_configs[0].name == "arena/basic"
        assert spec.controls.seed == 42

    def test_create_full_experiment(self):
        """Test create_full_experiment factory."""
        spec = create_full_experiment()
        
        assert spec.name == "full_training"
        assert spec.description == "Complete training and evaluation"
        assert spec.trainer_config.total_timesteps == 10000000
        assert spec.trainer_config.rollout_workers == 8
        assert len(spec.eval_configs) == 2
        assert spec.eval_configs[0].name == "arena/basic"
        assert spec.eval_configs[1].name == "arena/combat"
        assert spec.controls.enforce_determinism is True

    def test_create_eval_only_experiment(self):
        """Test create_eval_only_experiment factory."""
        policy_path = "file://./test_policy.pt"
        spec = create_eval_only_experiment(policy_path)
        
        assert spec.name == "eval_only"
        assert spec.description == "Evaluation of existing policy"
        assert spec.policy_path == policy_path
        assert len(spec.eval_configs) == 2
        assert spec.eval_configs[0].name == "arena/basic"
        assert spec.eval_configs[1].name == "arena/combat"

    def test_factory_functions_valid_configs(self):
        """Test that factory functions create valid configurations."""
        # This would normally fail if batch_size < minibatch_size
        specs = [
            create_quick_experiment(),
            create_full_experiment(),
            create_eval_only_experiment("file://./test.pt"),
        ]
        
        for spec in specs:
            # Check trainer config is valid
            assert spec.trainer_config.batch_size >= spec.trainer_config.minibatch_size
            # Check system config has device
            assert spec.system_config.device in ["cpu", "cuda"]
            # Check agent config exists
            assert isinstance(spec.agent_config, AgentConfig)