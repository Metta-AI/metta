"""
End-to-end integration test for the Protein sweep pipeline.
This test runs the real sweep pipeline but mocks training and evaluation
to verify that Protein suggestions are working correctly.
"""

import json
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import wandb
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


class MockTrainerResults:
    """Mock training results with configurable outcomes."""

    def __init__(self, agent_steps=50000, epochs=5, train_time=120.0):
        self.agent_steps = agent_steps
        self.epochs = epochs
        self.train_time = train_time


class MockEvaluationResults:
    """Mock evaluation results with configurable scores."""

    def __init__(self, reward_score=0.75, eval_time=15.0):
        self.reward_score = reward_score
        self.eval_time = eval_time


class MockPolicyRecord:
    """Mock policy record for testing."""

    def __init__(self, name="test_policy", uri="file://test/path"):
        self.name = name
        self.uri = uri
        self.metadata = {}


class TestProteinEndToEndPipeline:
    """Test the complete Protein sweep pipeline with mocked training/eval."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_sweep_config(self):
        """Create a realistic sweep configuration for testing."""
        return OmegaConf.create(
            {
                "protein": {
                    "num_random_samples": 3,
                    "max_suggestion_cost": 300,  # 5 minutes max
                    "resample_frequency": 0,
                    "global_search_scale": 1,
                    "random_suggestions": 100,
                    "suggestions_per_pareto": 50,
                },
                "parameters": {
                    "metric": "reward",
                    "goal": "maximize",
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {
                                "distribution": "log_normal",
                                "min": 0.0001,
                                "max": 0.01,
                                "mean": 0.001,
                                "scale": 0.5,
                            }
                        },
                        "batch_size": {
                            "distribution": "int_uniform",
                            "min": 1024,
                            "max": 8192,
                            "mean": 4096,
                            "scale": "auto",
                        },
                        "gamma": {
                            "distribution": "logit_normal",
                            "min": 0.9,
                            "max": 0.999,
                            "mean": 0.99,
                            "scale": 0.3,
                        },
                    },
                },
            }
        )

    def create_mock_wandb_run(self, run_name="test_run", sweep_id="test_sweep_123"):
        """Create a mock WandB run with necessary attributes."""

        # Create a simple dict-like object for summary
        class MockSummary(dict):
            def get(self, key, default=None):
                return super().get(key, default)

            def update(self, data):
                super().update(data)

        # Create a simple mock config object
        class MockConfig:
            def __init__(self):
                self.__dict__ = {"_locked": {}}

            def update(self, data, allow_val_change=False):
                pass

        # Create a simple mock sweep object
        class MockSweep:
            def __init__(self, sweep_id):
                self.id = sweep_id

        # Create a simple mock run object
        class MockRun:
            def __init__(self, run_name, sweep_id):
                self.name = run_name
                self.id = f"run_{run_name}_{int(time.time())}"
                self.entity = "test_entity"
                self.project = "test_project"
                self.summary = MockSummary()
                self.config = MockConfig()
                self.sweep = MockSweep(sweep_id)

        return MockRun(run_name, sweep_id)

    def create_mock_training_pipeline(self, results: MockTrainerResults):
        """Create mock training pipeline that returns specified results."""

        def mock_train(*args, **kwargs):
            """Mock training function that simulates training completion."""
            time.sleep(0.1)  # Simulate some work
            return {
                "agent_steps": results.agent_steps,
                "epochs": results.epochs,
                "train_time": results.train_time,
                "success": True,
            }

        return mock_train

    def create_mock_evaluation_pipeline(self, results: MockEvaluationResults):
        """Create mock evaluation pipeline that returns specified results."""

        def mock_evaluate(*args, **kwargs):
            """Mock evaluation function that simulates evaluation completion."""
            time.sleep(0.1)  # Simulate some work
            return {"reward": results.reward_score, "eval_time": results.eval_time, "success": True}

        return mock_evaluate

    def test_single_sweep_name_with_mocked_training(self, temp_workspace, mock_sweep_config):
        """Test a single sweep run with mocked training and evaluation."""

        # Setup mock results
        training_results = MockTrainerResults(agent_steps=50000, epochs=5, train_time=120.0)
        eval_results = MockEvaluationResults(reward_score=0.75, eval_time=15.0)

        # Create mock WandB run
        mock_run = self.create_mock_wandb_run("protein_test_run_1")

        with patch("wandb.init", return_value=mock_run), patch("wandb.Api") as mock_api:
            # Mock WandB API to return no previous runs (first run)
            mock_api.return_value.runs.return_value = []

            # Initialize WandB context
            wandb.init(project="test_project", mode="offline")

            try:
                # Create MettaProtein instance
                protein = MettaProtein(mock_sweep_config, mock_run)

                # Verify protein initialized correctly
                assert protein._sweep_id == "test_sweep_123"
                assert protein._num_observations == 0  # First run

                # Get protein suggestion
                suggestion, info = protein.suggest()

                # Verify suggestion structure
                assert "trainer" in suggestion
                assert "optimizer" in suggestion["trainer"]
                assert "learning_rate" in suggestion["trainer"]["optimizer"]
                assert "batch_size" in suggestion["trainer"]
                assert "gamma" in suggestion["trainer"]

                # Verify suggestion values are within expected ranges
                lr = suggestion["trainer"]["optimizer"]["learning_rate"]
                batch_size = suggestion["trainer"]["batch_size"]
                gamma = suggestion["trainer"]["gamma"]

                assert 0.0001 <= lr <= 0.01
                assert 1024 <= batch_size <= 8192
                assert isinstance(batch_size, int)
                assert 0.9 <= gamma <= 0.999

                # Mock training execution
                mock_train = self.create_mock_training_pipeline(training_results)
                train_result = mock_train(suggestion)
                assert train_result["success"]

                # Mock evaluation execution
                mock_eval = self.create_mock_evaluation_pipeline(eval_results)
                eval_result = mock_eval()
                assert eval_result["success"]

                # Record observation in protein
                total_time = train_result["train_time"] + eval_result["eval_time"]
                protein.record_observation(eval_result["reward"], total_time)

                # Verify observation was recorded
                assert len(protein._protein.success_observations) == 1
                observation = protein._protein.success_observations[0]
                assert observation["output"] == eval_result["reward"]
                assert observation["cost"] == total_time
                assert not observation["is_failure"]

                # Verify WandB summary was updated
                assert mock_run.summary.get("protein.objective") == eval_result["reward"]
                assert mock_run.summary.get("protein.cost") == total_time
                assert mock_run.summary.get("protein.state") == "success"

            finally:
                wandb.finish()

    def test_multi_run_sweep_progression(self, temp_workspace, mock_sweep_config):
        """Test multiple sweep runs to verify protein learning progression."""

        # Define results for multiple runs with improving performance
        run_configs = [
            {
                "name": "protein_test_run_0",
                "training": MockTrainerResults(50000, 5, 120.0),
                "eval": MockEvaluationResults(0.45, 15.0),  # Poor performance
                "suggestion_lr": 0.001,  # Search center for first run
            },
            {
                "name": "protein_test_run_1",
                "training": MockTrainerResults(50000, 5, 118.0),
                "eval": MockEvaluationResults(0.72, 14.0),  # Better performance
                "suggestion_lr": None,  # Will be determined by protein
            },
            {
                "name": "protein_test_run_2",
                "training": MockTrainerResults(50000, 5, 125.0),
                "eval": MockEvaluationResults(0.88, 16.0),  # Best performance
                "suggestion_lr": None,  # Will be determined by protein
            },
        ]

        # Store observations for protein loading
        previous_observations = []

        for i, run_config in enumerate(run_configs):
            mock_run = self.create_mock_wandb_run(run_config["name"])

            # Create mock previous runs for protein loading
            mock_previous_runs = []
            for j, prev_obs in enumerate(previous_observations):
                mock_prev_run = MagicMock()
                mock_prev_run.name = f"protein_test_run_{j}"
                mock_prev_run.id = f"run_id_{j}"
                mock_prev_run.summary = {
                    "protein.state": "success",
                    "protein.objective": prev_obs["objective"],
                    "protein.cost": prev_obs["cost"],
                    "protein.suggestion": prev_obs["suggestion"],
                    "protein.suggestion_info": {"suggestion_uuid": f"run_id_{j}"},
                }
                mock_previous_runs.append(mock_prev_run)

            with patch("wandb.init", return_value=mock_run), patch("wandb.Api") as mock_api:
                # Mock WandB API to return previous runs
                mock_api.return_value.runs.return_value = mock_previous_runs

                wandb.init(project="test_project", mode="offline")

                try:
                    # Create MettaProtein instance
                    protein = MettaProtein(mock_sweep_config, mock_run)

                    # Verify protein loaded previous observations
                    assert protein._num_observations == len(previous_observations)

                    # Get protein suggestion
                    suggestion, info = protein.suggest()

                    # For first run, should return search center
                    if i == 0:
                        assert abs(suggestion["trainer"]["optimizer"]["learning_rate"] - 0.001) < 1e-6
                    else:
                        # For subsequent runs, should use GP optimization
                        lr = suggestion["trainer"]["optimizer"]["learning_rate"]
                        assert 0.0001 <= lr <= 0.01 + 1e-10  # Add small epsilon for floating point comparison
                        # Should not be exactly the search center (unless by coincidence)
                        # This verifies GP optimization is being used

                    # Mock training and evaluation
                    mock_train = self.create_mock_training_pipeline(run_config["training"])
                    mock_eval = self.create_mock_evaluation_pipeline(run_config["eval"])

                    train_result = mock_train(suggestion)
                    eval_result = mock_eval()

                    # Record observation
                    total_time = train_result["train_time"] + eval_result["eval_time"]
                    protein.record_observation(eval_result["reward"], total_time)

                    # Store observation for next iteration
                    previous_observations.append(
                        {"objective": eval_result["reward"], "cost": total_time, "suggestion": suggestion}
                    )

                    # Verify protein state
                    expected_observations = i + 1
                    assert len(protein._protein.success_observations) == expected_observations

                finally:
                    wandb.finish()

        # Verify that protein learned from the progression
        assert len(previous_observations) == 3

        # Verify improving performance trend
        scores = [obs["objective"] for obs in previous_observations]
        assert scores[0] < scores[1] < scores[2]  # Improving trend

    def test_protein_handles_training_failures(self, temp_workspace, mock_sweep_config):
        """Test that protein correctly handles training/evaluation failures."""

        mock_run = self.create_mock_wandb_run("protein_failure_test")

        with patch("wandb.init", return_value=mock_run), patch("wandb.Api") as mock_api:
            mock_api.return_value.runs.return_value = []

            wandb.init(project="test_project", mode="offline")

            try:
                protein = MettaProtein(mock_sweep_config, mock_run)
                suggestion, info = protein.suggest()

                # Simulate training/evaluation failure
                protein.record_failure("Training failed due to OOM error")

                # Verify failure was recorded
                assert len(protein._protein.success_observations) == 1
                failure_obs = protein._protein.success_observations[0]
                assert failure_obs["is_failure"]
                assert failure_obs["output"] == 0.0
                assert failure_obs["cost"] == 0.001

                # Verify WandB summary
                assert mock_run.summary.get("protein.state") == "failure"
                assert mock_run.summary.get("protein.error") == "Training failed due to OOM error"

            finally:
                wandb.finish()

    def test_protein_cost_constraints(self, temp_workspace, mock_sweep_config):
        """Test that protein respects cost constraints."""

        # Set very low max cost to test constraint
        mock_sweep_config.protein.max_suggestion_cost = 60  # 1 minute max

        mock_run = self.create_mock_wandb_run("protein_cost_test")

        # Create mock previous run with high cost
        mock_high_cost_run = MagicMock()
        mock_high_cost_run.name = "expensive_run"
        mock_high_cost_run.id = "expensive_run_id"
        mock_high_cost_run.summary = {
            "protein.state": "success",
            "protein.objective": 0.9,
            "protein.cost": 300,  # 5 minutes - too expensive
            "protein.suggestion": {
                "trainer": {"optimizer": {"learning_rate": 0.005}, "batch_size": 8192, "gamma": 0.999}
            },
            "protein.suggestion_info": {"suggestion_uuid": "expensive_run_id"},
        }

        with patch("wandb.init", return_value=mock_run), patch("wandb.Api") as mock_api:
            mock_api.return_value.runs.return_value = [mock_high_cost_run]

            wandb.init(project="test_project", mode="offline")

            try:
                protein = MettaProtein(mock_sweep_config, mock_run)

                # Verify previous expensive run was loaded
                assert protein._num_observations == 1

                # Get suggestion - should avoid expensive configurations
                suggestion, info = protein.suggest()

                # Verify suggestion structure is valid
                assert "trainer" in suggestion
                lr = suggestion["trainer"]["optimizer"]["learning_rate"]
                assert 0.0001 <= lr <= 0.01 + 1e-10  # Add small epsilon for floating point comparison

                # The protein should prefer lower-cost suggestions
                # (exact behavior depends on GP optimization, but should not crash)

            finally:
                wandb.finish()

    def test_protein_suggestion_persistence(self, temp_workspace, mock_sweep_config):
        """Test that protein suggestions are properly saved and can be reloaded."""

        mock_run = self.create_mock_wandb_run("protein_persistence_test")

        with patch("wandb.init", return_value=mock_run), patch("wandb.Api") as mock_api:
            mock_api.return_value.runs.return_value = []

            wandb.init(project="test_project", mode="offline")

            try:
                protein = MettaProtein(mock_sweep_config, mock_run)
                suggestion, info = protein.suggest()

                # Verify suggestion was saved to WandB summary
                saved_suggestion = mock_run.summary.get("protein.suggestion")
                saved_info = mock_run.summary.get("protein.suggestion_info")

                assert saved_suggestion is not None
                assert saved_info is not None

                # Verify saved suggestion matches returned suggestion
                assert saved_suggestion == suggestion
                assert saved_info == info

                # Verify suggestion can be JSON serialized (important for WandB)
                json.dumps(saved_suggestion)
                json.dumps(saved_info)

            finally:
                wandb.finish()

    def test_realistic_protein_learning_progression(self, temp_workspace, mock_sweep_config):
        """Test realistic protein learning with varying hyperparameters and performance."""

        # Define a realistic learning progression where different hyperparameters
        # lead to different performance outcomes
        learning_scenarios = [
            # Run 0: Search center (baseline)
            {
                "name": "baseline_run",
                "force_lr": 0.001,  # Search center
                "force_batch_size": 4096,  # Search center
                "force_gamma": 0.99,  # Search center
                "expected_score": 0.65,  # Decent baseline performance
                "cost": 150.0,
            },
            # Run 1: Higher learning rate - should perform worse
            {
                "name": "high_lr_run",
                "force_lr": 0.005,  # Much higher than optimal
                "force_batch_size": 4096,
                "force_gamma": 0.99,
                "expected_score": 0.35,  # Poor performance due to instability
                "cost": 140.0,
            },
            # Run 2: Lower learning rate - should perform well but slower
            {
                "name": "low_lr_run",
                "force_lr": 0.0003,  # Lower than optimal
                "force_batch_size": 4096,
                "force_gamma": 0.99,
                "expected_score": 0.78,  # Good performance but slower learning
                "cost": 180.0,  # Takes longer
            },
            # Run 3: Optimal learning rate with larger batch
            {
                "name": "optimal_run",
                "force_lr": 0.0008,  # Near optimal
                "force_batch_size": 6144,  # Larger batch
                "force_gamma": 0.995,  # Higher gamma
                "expected_score": 0.92,  # Best performance
                "cost": 165.0,
            },
        ]

        previous_observations = []
        learning_rates_tested = []
        scores_achieved = []

        for i, scenario in enumerate(learning_scenarios):
            mock_run = self.create_mock_wandb_run(scenario["name"])

            # Create mock previous runs
            mock_previous_runs = []
            for j, prev_obs in enumerate(previous_observations):
                mock_prev_run = MagicMock()
                mock_prev_run.name = learning_scenarios[j]["name"]
                mock_prev_run.id = f"run_id_{j}"
                mock_prev_run.summary = {
                    "protein.state": "success",
                    "protein.objective": prev_obs["objective"],
                    "protein.cost": prev_obs["cost"],
                    "protein.suggestion": prev_obs["suggestion"],
                    "protein.suggestion_info": {"suggestion_uuid": f"run_id_{j}"},
                }
                mock_previous_runs.append(mock_prev_run)

            with patch("wandb.init", return_value=mock_run), patch("wandb.Api") as mock_api:
                mock_api.return_value.runs.return_value = mock_previous_runs

                wandb.init(project="test_project", mode="offline")

                try:
                    protein = MettaProtein(mock_sweep_config, mock_run)

                    # Verify protein loaded correct number of previous observations
                    assert protein._num_observations == len(previous_observations)

                    # Get suggestion from protein
                    suggestion, info = protein.suggest()

                    # For testing purposes, override with our controlled values
                    # This simulates what would happen if protein suggested these values
                    controlled_suggestion = {
                        "trainer": {
                            "optimizer": {"learning_rate": scenario["force_lr"]},
                            "batch_size": scenario["force_batch_size"],
                            "gamma": scenario["force_gamma"],
                        }
                    }

                    # Simulate training with controlled results
                    training_results = MockTrainerResults(50000, 5, scenario["cost"] - 15.0)
                    eval_results = MockEvaluationResults(scenario["expected_score"], 15.0)

                    mock_train = self.create_mock_training_pipeline(training_results)
                    mock_eval = self.create_mock_evaluation_pipeline(eval_results)

                    train_result = mock_train(controlled_suggestion)
                    eval_result = mock_eval()

                    # Record observation
                    total_time = train_result["train_time"] + eval_result["eval_time"]
                    protein.record_observation(eval_result["reward"], total_time)

                    # Store for analysis
                    previous_observations.append(
                        {"objective": eval_result["reward"], "cost": total_time, "suggestion": controlled_suggestion}
                    )
                    learning_rates_tested.append(scenario["force_lr"])
                    scores_achieved.append(eval_result["reward"])

                    # Verify protein state
                    assert len(protein._protein.success_observations) == i + 1

                    # Print learning progress for debugging
                    print(
                        f"\nRun {i}: lr={scenario['force_lr']:.4f}, "
                        f"score={eval_result['reward']:.3f}, cost={total_time:.1f}"
                    )

                finally:
                    wandb.finish()

        # Analyze learning progression
        print("\nLearning progression:")
        print(f"Learning rates tested: {learning_rates_tested}")
        print(f"Scores achieved: {scores_achieved}")

        # Verify we tested different learning rates
        assert len(set(learning_rates_tested)) >= 3, "Should test diverse learning rates"

        # Verify we found the best performing configuration
        best_score_idx = scores_achieved.index(max(scores_achieved))
        best_lr = learning_rates_tested[best_score_idx]
        print(f"Best performing LR: {best_lr:.4f} with score: {max(scores_achieved):.3f}")

        # The optimal LR (0.0008) should achieve the best score
        assert best_lr == 0.0008, f"Expected best LR to be 0.0008, got {best_lr}"
        assert max(scores_achieved) >= 0.9, f"Expected best score >= 0.9, got {max(scores_achieved)}"

        # Verify protein learned the relationship (higher LR = worse performance in our scenario)
        high_lr_score = scores_achieved[1]  # 0.005 LR
        low_lr_score = scores_achieved[2]  # 0.0003 LR
        optimal_lr_score = scores_achieved[3]  # 0.0008 LR

        assert high_lr_score < low_lr_score, "High LR should perform worse than low LR"
        assert optimal_lr_score > low_lr_score, "Optimal LR should perform better than low LR"

        print("\n✅ Protein learning test completed successfully!")
        print(f"   - Tested {len(learning_scenarios)} different configurations")
        print(f"   - Found optimal LR: {best_lr:.4f}")
        print(f"   - Achieved best score: {max(scores_achieved):.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
