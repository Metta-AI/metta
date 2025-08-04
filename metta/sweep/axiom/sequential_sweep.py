"""Sequential sweep implementation using clean State/Context pattern."""

import logging
import time
from typing import Any, Callable, Optional

import wandb
from cogweb.cogweb_client import CogwebClient
from pydantic import Field

from metta.common.util.datastruct import flatten_config
from metta.sweep.axiom import Context, Pipeline, PipelineState
from metta.sweep.axiom.training_guards import master_process_only
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.sweep_config import SweepConfig
from metta.sweep.wandb_utils import (
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)


class SweepState(PipelineState):
    """Typed state for hyperparameter sweep - shows data flow clearly.

    This state accumulates throughout a single trial execution.
    """

    # Trial identification
    trial_index: int = 0
    run_name: str = ""

    # Hyperparameter suggestion
    suggestion: dict[str, Any] = Field(default_factory=dict)

    # Training artifacts
    train_tool: Optional[Any] = None  # TrainTool instance
    train_time: float = 0.0
    policy_uri: Optional[str] = None  # Populated after training

    # Evaluation results
    eval_time: float = 0.0
    score: Optional[float] = None
    cost_hours: float = 0.0
    eval_results: dict[str, Any] = Field(default_factory=dict)  # Store raw evaluation results

    # WandB tracking
    wandb_run: Optional[Any] = None  # wandb.Run instance
    summary: dict[str, Any] = Field(default_factory=dict)

    # Observations loaded
    observation_count: int = 0


class SequentialSweepPipeline:
    """Sequential sweep using clean State/Context separation.

    This class builds pipelines that operate on typed SweepState,
    following the canonical pattern where stages modify state directly.
    """

    def __init__(
        self,
        sweep_name: str,
        protein_config,
        train_tool_factory,
        eval_tool_factory,
        wandb_cfg,
        sweep_server_uri: str = "https://api.observatory.softmax-research.net",
        max_observations_to_load: int = 250,
        stats_client=None,
    ):
        """Initialize sweep with configuration."""
        self.sweep_name = sweep_name
        self.protein_config = protein_config
        self.train_tool_factory = train_tool_factory
        self.eval_tool_factory = eval_tool_factory
        self.wandb_cfg = wandb_cfg
        self.sweep_server_uri = sweep_server_uri
        self.max_observations_to_load = max_observations_to_load
        self.stats_client = stats_client

        # Initialize optimizer
        self.optimizer = MettaProtein(protein_config)

        # Services initialized during I/O
        self.cogweb_client = None
        self.sweep_client = None

    def build_pipeline(self) -> Pipeline:
        """Build pipeline for a single trial execution.

        Returns a pipeline that operates on SweepState with proper
        State/Context separation.
        """
        # Create typed state for this trial
        state = SweepState()

        return (
            Pipeline(state)
            # Load previous observations
            .io("load_observations", self.load_previous_observations)
            # Trial execution
            .stage("suggest", self.suggest_hyperparameters)
            .io("get_run_name", self.get_run_name)
            .stage("train", self.train_model)
            .stage("set_policy_uri", self.set_top_policy_uri)
            .stage("evaluate", self.evaluate_model)
            .io("fetch_eval_metrics", self.fetch_metrics)
            .stage("calculate_metrics", self.calculate_metrics)
            .io("record_metric_and_obs_to_wandb", self.record_to_wandb)
        )

    @master_process_only()
    def initialize_services(self):
        """Initialize external services (must be called before pipeline run)."""
        self.cogweb_client = CogwebClient.get_client(base_url=self.sweep_server_uri)
        self.sweep_client = self.cogweb_client.sweep_client()

        # Register sweep if it doesn't exist
        sweep_info = self.sweep_client.get_sweep(self.sweep_name)
        if not sweep_info.exists:
            logger.info(f"Registering sweep {self.sweep_name}")
            self.sweep_client.create_sweep(
                self.sweep_name,
                self.wandb_cfg.project,
                self.wandb_cfg.entity,
                self.sweep_name,
            )

    # ============= I/O Operations (External Input/Output) =============
    @master_process_only()
    def load_previous_observations(self, state: SweepState, ctx: Context) -> None:
        """I/O: Load previous observations from WandB."""
        previous_observations = fetch_protein_observations_from_wandb(
            wandb_entity=self.wandb_cfg.entity,
            wandb_project=self.wandb_cfg.project,
            sweep_name=self.sweep_name,
            max_observations=self.max_observations_to_load,
        )

        logger.info(f"Loaded {len(previous_observations)} previous observations")
        state.observation_count = len(previous_observations)

        # Update optimizer with observations
        for obs in previous_observations:
            self.optimizer.observe(obs["suggestion"], obs["objective"], obs["cost"], obs.get("is_failure", False))

    @master_process_only()
    def get_run_name(self, state: SweepState, ctx: Context) -> None:
        """I/O: Get next run ID from Cogweb."""
        if self.sweep_client is None:
            raise ValueError("Cogweb client not initialized")

        state.run_name = self.sweep_client.get_next_run_id(self.sweep_name)
        logger.info(f"Got run name: {state.run_name}")

    @master_process_only()
    def fetch_metrics(self, state: SweepState, ctx: Context) -> None:
        """I/O: Fetch metrics from WandB."""
        api = wandb.Api()
        wandb_run = api.run(f"{self.wandb_cfg.entity}/{self.wandb_cfg.project}/{state.run_name}")
        assert wandb_run is not None, f"Error fetching run: {state.run_name}"

        state.wandb_run = wandb_run
        state.summary = dict(wandb_run.summary)

    @master_process_only()
    def record_to_wandb(self, state: SweepState, ctx: Context) -> None:
        """I/O: Record results to WandB."""
        assert state.score is not None

        if state.wandb_run is None:
            logger.warning("No wandb run to record to")
            return

        # Update wandb config
        state.wandb_run.config.update(
            {
                "sweep_name": self.sweep_name,
                "protein_suggestion": state.suggestion,
            }
        )

        # Record protein observation
        record_protein_observation_to_wandb(
            state.wandb_run,
            state.suggestion,
            state.score,
            state.cost_hours,
            is_failure=False,
        )

        # Update summary
        state.wandb_run.summary.update(
            {
                "trial_index": state.trial_index,
                "trial_score": state.score,
                "trial_cost_hours": state.cost_hours,
                self.protein_config.metric: state.score,
            }
        )

    # ============= Stages (Deterministic Computation) =============

    @master_process_only()
    def suggest_hyperparameters(self, state: SweepState, ctx: Context) -> None:
        """Stage: Generate next hyperparameter suggestion."""
        suggestion, _ = self.optimizer.suggest()
        state.suggestion = suggestion

        # Get trial index from context if available
        state.trial_index = ctx.trial_index or 0

        logger.info(f"Trial {state.trial_index}: Generated suggestion")

    def train_model(self, state: SweepState, ctx: Context) -> None:
        """Stage: Execute training with suggested hyperparameters."""
        # Create and configure train tool
        train_tool = self.train_tool_factory(state.run_name)
        train_tool = self._apply_suggestion_to_tool(train_tool, state.suggestion)

        # Configure wandb
        train_tool.wandb = self.wandb_cfg
        train_tool.wandb.group = self.sweep_name
        train_tool.wandb.name = state.run_name
        train_tool.wandb.run_id = state.run_name
        if "sweep" not in train_tool.wandb.tags:
            train_tool.wandb.tags.append("sweep")

        # Train
        train_start_time = time.time()
        train_tool.invoke(args={}, overrides=[])
        state.train_time = time.time() - train_start_time

        state.train_tool = train_tool
        logger.info(f"Training completed for {state.run_name}")

    @master_process_only()
    def set_top_policy_uri(self, state: SweepState, ctx: Context) -> None:
        """Stage: Extract the policy URI from the trained model."""
        if state.train_tool is None:
            raise ValueError("No train_tool available - train_model must run before set_top_policy_uri")

        # Get policy_uri from the train tool
        # The train tool sets this during initialization
        state.policy_uri = state.train_tool.policy_uri

        logger.info(f"Policy URI set to: {state.policy_uri}")

    @master_process_only()
    def evaluate_model(self, state: SweepState, ctx: Context) -> None:
        """Stage: Run evaluation simulations."""
        assert state.policy_uri is not None

        # Convert file:// URI to wandb://run/ URI for proper metric logging
        # The SimTool needs the wandb run info to push metrics correctly
        eval_policy_uri = state.policy_uri
        if state.policy_uri.startswith("file://"):
            # Use wandb://run/{run_name} format for proper wandb integration
            eval_policy_uri = f"wandb://run/{state.run_name}"
            logger.info(f"Using wandb URI for evaluation: {eval_policy_uri}")

        # Use eval_tool_factory to create SimTool
        sim_tool = self.eval_tool_factory(eval_policy_uri)

        # Configure wandb to match training configuration
        sim_tool.wandb = self.wandb_cfg
        sim_tool.wandb.group = self.sweep_name
        sim_tool.wandb.name = state.run_name
        sim_tool.wandb.run_id = state.run_name
        if "sweep" not in sim_tool.wandb.tags:
            sim_tool.wandb.tags.append("sweep")

        # Enable pushing metrics to WandB
        sim_tool.push_metrics_to_wandb = True

        eval_start_time = time.time()
        sim_tool.invoke(args={}, overrides=[])
        state.eval_time = time.time() - eval_start_time

    @master_process_only()
    def calculate_metrics(self, state: SweepState, ctx: Context) -> None:
        """Stage: Calculate score and cost from fetched metrics."""
        # The metric is at evaluator/eval_{metric}/score for eval metrics
        metric_path = f"evaluator/eval_{self.protein_config.metric}/score"
        eval_score = state.summary.get(metric_path)

        if eval_score is None:
            logger.error(f"Could not find metric at path: {metric_path}")
            logger.error(f"Available metrics in summary: {list(state.summary.keys())}")
            raise ValueError(f"Error during evaluation, got score: None. Check metric: {self.protein_config.metric}")

        logger.info(f"Found metric at path: {metric_path} = {eval_score}")
        state.score = eval_score

        # Calculate cost
        if "monitor/cost/accrued_total" in state.summary:
            state.cost_hours = state.summary["monitor/cost/accrued_total"]
        elif "total_time" in state.summary:
            state.cost_hours = state.summary["total_time"] / 3600.0
        else:
            state.cost_hours = (state.train_time + state.eval_time) / 3600.0

        logger.info(f"Trial {state.trial_index}: score={state.score:.4f}, cost={state.cost_hours:.4f}h")

    # ============= Helper Methods =============

    def _apply_suggestion_to_tool(self, train_tool: TrainTool, suggestion: dict) -> TrainTool:
        """Apply optimizer suggestion to a TrainTool instance."""
        for key_path, value in flatten_config(suggestion).items():
            train_tool = train_tool.override(key_path, str(value))
            logger.debug(f"Applied {key_path} = {value}")
        return train_tool
