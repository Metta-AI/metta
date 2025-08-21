"""Lightweight test suite for sweep infrastructure via Tools.

Following the testing philosophy: focused integration tests with minimal unit tests.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestSweepToolIntegration:
    """Integration tests for the SweepTool."""

    def test_sweep_tool_initialization(self):
        """Test that SweepTool initializes correctly with required configs."""
        from metta.sweep.protein_config import ProteinConfig
        from metta.sweep.sweep_config import SweepConfig
        from metta.tools.sweep import SweepTool

        # Create minimal protein config
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={"trainer": {"optimizer": {"learning_rate": {"min": 1e-5, "max": 1e-3}}}},
        )

        # Create sweep config with proper evaluation simulations
        from metta.mettagrid import EnvConfig
        from metta.sim.simulation_config import SimulationConfig

        sweep_config = SweepConfig(
            num_trials=2,
            protein=protein_config,
            evaluation_simulations=[SimulationConfig(name="test_sim", num_episodes=1, env=EnvConfig())],
        )

        # Create a mock train tool factory
        def train_factory(run_name: str):
            train_tool = MagicMock()
            train_tool.run = run_name
            return train_tool

        # Create SweepTool
        sweep_tool = SweepTool(
            sweep=sweep_config,
            sweep_name="test_sweep",
            train_tool_factory=train_factory,
        )

        assert sweep_tool.sweep_name == "test_sweep"
        assert sweep_tool.sweep.num_trials == 2
        assert sweep_tool.sweep.protein.metric == "test_metric"

    @patch("metta.sweep.sweep.CogwebClient")
    @patch("metta.sweep.sweep.wandb.Api")
    @patch("metta.sweep.sweep.fetch_protein_observations_from_wandb")
    def test_sweep_tool_invoke_mock(self, mock_fetch, mock_wandb_api, mock_cogweb):
        """Test SweepTool invoke with mocked dependencies."""
        from metta.sim.simulation_config import SimulationConfig
        from metta.sweep.protein_config import ParameterConfig, ProteinConfig
        from metta.sweep.sweep_config import SweepConfig
        from metta.tools.sweep import SweepTool

        # Setup mocks
        mock_fetch.return_value = []  # No previous observations

        mock_cogweb_instance = MagicMock()
        mock_sweep_client = MagicMock()
        mock_sweep_client.get_sweep.return_value.exists = False
        mock_sweep_client.get_next_run_id.return_value = "test_run_001"
        mock_cogweb_instance.sweep_client.return_value = mock_sweep_client
        mock_cogweb.get_client.return_value = mock_cogweb_instance

        # Mock wandb API
        mock_api = MagicMock()
        mock_run = MagicMock()
        mock_run.summary = {"evaluator/test_metric/score": 0.5, "total_time": 100}
        mock_api.run.return_value = mock_run
        mock_wandb_api.return_value = mock_api

        # Create protein config with complete parameters
        protein_config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            parameters={
                "trainer": {
                    "optimizer": {
                        "learning_rate": ParameterConfig(
                            min=1e-5, max=1e-3, distribution="log_normal", mean=1e-4, scale="auto"
                        )
                    }
                }
            },
        )

        # Create sweep config with 1 trial
        from metta.mettagrid import EnvConfig

        sweep_config = SweepConfig(
            num_trials=1,
            protein=protein_config,
            evaluation_simulations=[SimulationConfig(name="test_sim", num_episodes=1, env=EnvConfig())],
        )

        # Create mock train factory
        # Store created tools to verify they were invoked
        created_tools = []

        def train_factory(run_name: str):
            mock_train_tool = MagicMock()
            mock_train_tool.run = run_name
            mock_train_tool.invoke.return_value = 0
            mock_train_tool.system = MagicMock()
            mock_train_tool.system.data_dir = "/tmp"
            created_tools.append(mock_train_tool)
            return mock_train_tool

        # Create and invoke SweepTool
        sweep_tool = SweepTool(
            sweep=sweep_config,
            sweep_name="test_sweep",
            train_tool_factory=train_factory,
        )

        # Mock SimTool to avoid actual evaluation
        # Also mock apply_suggestion_to_tool to return the same tool
        with (
            patch("metta.sweep.sweep.SimTool") as mock_sim_tool,
            patch("metta.sweep.sweep.apply_suggestion_to_tool") as mock_apply,
        ):
            mock_sim_instance = MagicMock()
            mock_sim_instance.invoke.return_value = None
            mock_sim_tool.return_value = mock_sim_instance

            # Make apply_suggestion_to_tool return the same tool
            mock_apply.side_effect = lambda tool, suggestion: tool

            result = sweep_tool.invoke(args={}, overrides=[])

        # Verify invocation succeeded
        assert result == 0

        # Verify sweep was registered
        mock_sweep_client.create_sweep.assert_called_once()

        # Verify training was invoked
        assert len(created_tools) == 1, f"Expected 1 train tool to be created, got {len(created_tools)}"
        created_tools[0].invoke.assert_called_once()

        # Verify evaluation was attempted
        mock_sim_tool.assert_called_once()


class TestProteinConfig:
    """Unit tests for ProteinConfig."""

    def test_protein_config_to_dict(self):
        """Test ProteinConfig conversion to Protein dict format."""
        from metta.sweep.protein_config import ParameterConfig, ProteinConfig

        config = ProteinConfig(
            metric="arena",
            goal="maximize",
            parameters={
                "trainer": {
                    "optimizer": {
                        "learning_rate": ParameterConfig(
                            min=1e-5, max=1e-2, distribution="log_normal", mean=1e-3, scale="auto"
                        )
                    },
                    "ppo": {
                        "clip_coef": {"min": 0.1, "max": 0.3, "distribution": "uniform", "mean": 0.2, "scale": "auto"}
                    },
                }
            },
        )

        protein_dict = config.to_protein_dict()

        assert protein_dict["metric"] == "arena"
        assert protein_dict["goal"] == "maximize"
        assert "trainer.optimizer.learning_rate" in protein_dict
        assert protein_dict["trainer.optimizer.learning_rate"]["min"] == 1e-5
        assert protein_dict["trainer.optimizer.learning_rate"]["max"] == 1e-2
        assert "trainer.ppo.clip_coef" in protein_dict
        assert protein_dict["trainer.ppo.clip_coef"]["min"] == 0.1

    def test_parameter_config_creation(self):
        """Test ParameterConfig requires all fields."""
        from metta.sweep.protein_config import ParameterConfig

        # All fields are required now
        param = ParameterConfig(min=0.1, max=1.0, distribution="uniform", mean=0.55, scale="auto")

        assert param.distribution == "uniform"
        assert param.mean == 0.55
        assert param.scale == "auto"

        # Test log_normal config
        log_param = ParameterConfig(
            min=1e-4,
            max=1e-2,
            distribution="log_normal",
            mean=1e-3,  # Geometric mean
            scale="auto",
        )
        assert log_param.mean == 1e-3


class TestSweepExperiments:
    """Test the sweep experiment definitions."""

    def test_sweep_arena_experiments_import(self):
        """Test that sweep experiments can be imported and created."""
        from experiments.sweep_arena import sweep_hpo, sweep_hpo_quick
        from metta.tools.sweep import SweepTool

        # Create sweep instances
        hpo_sweep = sweep_hpo("test_hpo", num_trials=2)
        quick_sweep = sweep_hpo_quick("test_quick", num_trials=1)

        # Verify sweep tools were created
        assert isinstance(hpo_sweep, SweepTool)
        assert isinstance(quick_sweep, SweepTool)

        # Verify configurations
        assert hpo_sweep.sweep_name == "test_hpo"
        assert hpo_sweep.sweep.num_trials == 2
        assert hpo_sweep.sweep.protein.metric == "eval_arena"

        assert quick_sweep.sweep_name == "test_quick"
        assert quick_sweep.sweep.num_trials == 1

        # Verify quick sweep has reduced timesteps
        train_tool = quick_sweep.train_tool_factory("test_run")
        assert train_tool.trainer.total_timesteps == 10000


@pytest.mark.integration
class TestSweepEndToEnd:
    """End-to-end integration test with minimal mocking."""

    @patch("metta.sweep.sweep.CogwebClient")
    @patch("metta.sweep.sweep.fetch_protein_observations_from_wandb")
    @patch("metta.sweep.sweep.record_protein_observation_to_wandb")
    @patch("metta.sweep.sweep.wandb")
    @patch("metta.sweep.sweep.SimTool")
    @patch("metta.sweep.sweep.apply_suggestion_to_tool")
    def test_minimal_sweep_execution(self, mock_apply, mock_sim_tool, mock_wandb, mock_record, mock_fetch, mock_cogweb):
        """Test a minimal sweep execution with core components."""
        from metta.common.wandb.wandb_context import WandbConfig
        from metta.mettagrid import EnvConfig
        from metta.sim.simulation_config import SimulationConfig
        from metta.sweep.protein_config import ProteinConfig
        from metta.sweep.sweep import sweep

        # Setup minimal mocks
        mock_fetch.return_value = []
        mock_record.return_value = None

        # Mock SimTool
        mock_sim_instance = MagicMock()
        mock_sim_instance.invoke.return_value = None
        mock_sim_tool.return_value = mock_sim_instance

        # Make apply_suggestion_to_tool return the same tool
        mock_apply.side_effect = lambda tool, suggestion: tool

        # Mock wandb
        mock_api = MagicMock()
        mock_run = MagicMock()
        mock_run.summary = {"evaluator/test/score": 0.5, "total_time": 100}
        mock_api.run.return_value = mock_run
        mock_wandb.Api.return_value = mock_api

        mock_cogweb_instance = MagicMock()
        mock_sweep_client = MagicMock()
        mock_sweep_client.get_sweep.return_value.exists = True  # Sweep already exists
        mock_sweep_client.get_next_run_id.side_effect = ["run_001", "run_002"]
        mock_cogweb_instance.sweep_client.return_value = mock_sweep_client
        mock_cogweb.get_client.return_value = mock_cogweb_instance

        # Create real configurations with proper ParameterConfig
        from metta.sweep.protein_config import ParameterConfig

        protein_config = ProteinConfig(
            metric="test",
            goal="maximize",
            method="random",  # Use random for simplicity
            parameters={"test_param": ParameterConfig(min=0, max=1, distribution="uniform", mean=0.5, scale="auto")},
        )

        # Mock train factory
        # Store created tools to verify they were invoked
        created_tools = []

        def train_factory(run_name: str):
            mock_train_tool = MagicMock()
            mock_train_tool.run = run_name
            mock_train_tool.invoke.return_value = 0
            mock_train_tool.system = MagicMock()
            created_tools.append(mock_train_tool)
            return mock_train_tool

        # Run the actual sweep function with proper evaluation simulations
        sweep(
            sweep_name="integration_test",
            protein_config=protein_config,
            train_tool_factory=train_factory,
            wandb_cfg=WandbConfig(enabled=False, project="", entity=""),
            num_trials=2,
            evaluation_simulations=[SimulationConfig(name="test_sim", num_episodes=1, env=EnvConfig())],
            sweep_server_uri="mock://server",
            max_observations_to_load=10,
        )

        # Verify expected number of training runs
        assert len(created_tools) == 2, f"Expected 2 train tools to be created, got {len(created_tools)}"
        for tool in created_tools:
            tool.invoke.assert_called_once()

        # Verify run names were generated
        assert mock_sweep_client.get_next_run_id.call_count == 2


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running sweep infrastructure tests...")

    # Test basic imports
    try:
        from metta.sweep.protein_config import ProteinConfig
        from metta.tools.sweep import SweepTool

        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        exit(1)

    # Test basic object creation
    try:
        config = ProteinConfig(metric="test", parameters={"lr": {"min": 0.001, "max": 0.01}})
        print("✓ ProteinConfig creation successful")
    except Exception as e:
        print(f"✗ ProteinConfig creation failed: {e}")
        exit(1)

    # Test sweep config creation
    try:
        from metta.mettagrid import EnvConfig
        from metta.sim.simulation_config import SimulationConfig
        from metta.sweep.sweep_config import SweepConfig

        sweep_config = SweepConfig(
            num_trials=1,
            protein=config,
            evaluation_simulations=[SimulationConfig(name="test_sim", num_episodes=1, env=EnvConfig())],
        )
        print("✓ SweepConfig creation successful")
    except Exception as e:
        print(f"✗ SweepConfig creation failed: {e}")
        exit(1)

    # Test sweep tool creation with mock factory
    try:
        sweep_tool = SweepTool(
            sweep=sweep_config,
            sweep_name="smoke_test",
            train_tool_factory=lambda x: MagicMock(),
        )
        print("✓ SweepTool creation successful")
    except Exception as e:
        print(f"✗ SweepTool creation failed: {e}")
        exit(1)

    print("\nAll smoke tests passed! Run 'pytest tests/sweep/test_sweep_tools.py -v' for full test suite.")
