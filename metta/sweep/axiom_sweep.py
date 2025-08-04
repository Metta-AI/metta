"""tAXIOM-based sweep implementation - MVP single trial example."""

import logging
import time
from typing import Callable, Optional, Sequence

import wandb
from cogweb.cogweb_client import CogwebClient

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.datastruct import flatten_config
from metta.common.wandb.wandb_context import WandbConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.axiom import Ctx, Pipeline
from metta.sweep.axiom.sequential_sweep import SequentialSweepPipeline
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.sweep_config import SweepConfig
from metta.sweep.wandb_utils import (
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)


# DEPRECATED: Use SequentialSweep instead
# This class is kept for backward compatibility
class SweepExperiment:
    """Sweep experiment using tAXIOM pipeline architecture.

    This class encapsulates all sweep operations as methods that can be
    composed into a pipeline. Each method is classified as either a stage
    (deterministic computation) or I/O (external operation).
    """

    def __init__(
        self,
        sweep_name: str,
        protein_config: ProteinConfig,
        train_tool_factory: Callable[[str], TrainTool],
        wandb_cfg: WandbConfig,
        evaluation_simulations: Sequence[SimulationConfig],
        sweep_server_uri: str = "https://api.observatory.softmax-research.net",
        max_observations_to_load: int = 250,
        stats_client: Optional[StatsClient] = None,
    ):
        """Initialize sweep experiment with configuration."""
        self.cfg = {
            "sweep_name": sweep_name,
            "protein_config": protein_config,
            "train_tool_factory": train_tool_factory,
            "wandb_cfg": wandb_cfg,
            "evaluation_simulations": evaluation_simulations,
            "sweep_server_uri": sweep_server_uri,
            "max_observations_to_load": max_observations_to_load,
            "stats_client": stats_client,
        }

        # Initialize protein optimizer
        self.protein = MettaProtein(protein_config)

        # Will be initialized in I/O operations
        self.cogweb_client = None
        self.sweep_client = None

    # ============= I/O Operations =============

    def initialize_services(self):
        """I/O: Initialize external services (Cogweb)."""
        self.cogweb_client = CogwebClient.get_client(base_url=self.cfg["sweep_server_uri"])
        self.sweep_client = self.cogweb_client.sweep_client()

        # Register sweep if it doesn't exist
        sweep_info = self.sweep_client.get_sweep(self.cfg["sweep_name"])
        if not sweep_info.exists:
            logger.info(f"Registering sweep {self.cfg['sweep_name']}")
            self.sweep_client.create_sweep(
                self.cfg["sweep_name"],
                self.cfg["wandb_cfg"].project,
                self.cfg["wandb_cfg"].entity,
                self.cfg["sweep_name"],
            )

    def load_previous_observations(self):
        """I/O: Load previous observations from WandB."""
        previous_observations = fetch_protein_observations_from_wandb(
            wandb_entity=self.cfg["wandb_cfg"].entity,
            wandb_project=self.cfg["wandb_cfg"].project,
            sweep_name=self.cfg["sweep_name"],
            max_observations=self.cfg["max_observations_to_load"],
        )

        logger.info(f"Loaded {len(previous_observations)} previous observations")

        # Update protein optimizer with observations
        for obs in previous_observations:
            self.protein.observe(obs["suggestion"], obs["objective"], obs["cost"], obs.get("is_failure", False))

        return {"observation_count": len(previous_observations)}

    def get_run_name(self, suggestion):
        """I/O: Get next run ID from Cogweb."""
        if self.sweep_client is None:
            raise ValueError("Error connecting to Cogweb sweep client, cannot proceed.")
        run_name = self.sweep_client.get_next_run_id(self.cfg["sweep_name"])
        return {"suggestion": suggestion["suggestion"], "run_name": run_name}

    def train_model(self, config):
        """Stage: Execute training with the configured tool.

        Training is deterministic given the config and seed, so it's a stage,
        not I/O. The fact that it takes time doesn't make it I/O.
        """
        suggestion = config["suggestion"]
        run_name = config["run_name"]

        # Create and configure train tool
        train_tool = self.cfg["train_tool_factory"](run_name)
        train_tool = self._apply_suggestion_to_tool(train_tool, suggestion)

        # Configure wandb
        train_tool.wandb = self.cfg["wandb_cfg"]
        train_tool.wandb.group = self.cfg["sweep_name"]
        train_tool.wandb.name = run_name
        train_tool.wandb.run_id = run_name
        train_tool.wandb.tags.append("sweep")

        # Train
        train_start_time = time.time()
        train_tool.invoke(args={}, overrides=[])
        train_time = time.time() - train_start_time

        return {
            **config,
            "train_time": train_time,
            "train_tool": train_tool,  # Needed for sim tool
        }

    def evaluate_model(self, train_result):
        """Stage: Run evaluation simulations.

        Evaluation is deterministic given the model and config, so it's a stage.
        """
        run_name = train_result["run_name"]
        train_tool = train_result["train_tool"]

        sim_tool = SimTool(
            simulations=self.cfg["evaluation_simulations"],
            policy_uris=[f"wandb://run/{run_name}"],
            selector_type="latest",
            stats_dir=f"/tmp/stats/{run_name}",
            wandb=self.cfg["wandb_cfg"],
            push_metrics_to_wandb=True,
            stats_server_uri=self.cfg["sweep_server_uri"] if self.cfg["stats_client"] else None,
            system=train_tool.system,
        )

        eval_start_time = time.time()
        sim_tool.invoke(args={}, overrides=[])
        eval_time = time.time() - eval_start_time

        return {**train_result, "eval_time": eval_time}

    def fetch_metrics(self, eval_result):
        """I/O: Fetch metrics from WandB."""
        run_name = eval_result["run_name"]

        api = wandb.Api()
        wandb_run = api.run(f"{self.cfg['wandb_cfg'].entity}/{self.cfg['wandb_cfg'].project}/{run_name}")
        assert wandb_run is not None, f"Error fetching run: {run_name}"

        summary = dict(wandb_run.summary)

        return {**eval_result, "wandb_run": wandb_run, "summary": summary}

    def record_to_wandb(self, trial_result):
        """I/O: Record results to WandB."""
        wandb_run = trial_result["wandb_run"]
        suggestion = trial_result["suggestion"]
        score = trial_result["score"]
        cost_hours = trial_result["cost_hours"]
        trial_index = trial_result.get("trial_index", 0)

        # Update wandb config
        wandb_run.config.update(
            {
                "sweep_name": self.cfg["sweep_name"],
                "protein_suggestion": suggestion,
            }
        )

        # Record protein observation
        record_protein_observation_to_wandb(
            wandb_run,
            suggestion,
            score,
            cost_hours,
            is_failure=False,
        )

        # Update summary
        wandb_run.summary.update(
            {
                "trial_index": trial_index,
                "trial_score": score,
                "trial_cost_hours": cost_hours,
                self.cfg["protein_config"].metric: score,
            }
        )

        return trial_result

    # ============= Stages (Pure Computation) =============

    def suggest_hyperparameters(self, load_result):
        """Stage: Generate next hyperparameter suggestion."""
        suggestion, _ = self.protein.suggest()
        return {"suggestion": suggestion}

    def calculate_metrics(self, metrics_data):
        """Stage: Calculate score and cost from fetched metrics."""
        summary = metrics_data["summary"]
        train_time = metrics_data["train_time"]
        eval_time = metrics_data["eval_time"]

        # Get evaluation score
        eval_score = summary.get(f"evaluator/{self.cfg['protein_config'].metric}/score")

        if eval_score is None:
            raise ValueError(
                f"Error during evaluation, got score: None. Check metric: {self.cfg['protein_config'].metric}"
            )

        # Calculate cost
        if "monitor/cost/accrued_total" in summary:
            cost_hours = summary["monitor/cost/accrued_total"]
        elif "total_time" in summary:
            cost_hours = summary["total_time"] / 3600.0
        else:
            cost_hours = (train_time + eval_time) / 3600.0

        return {**metrics_data, "score": eval_score, "cost_hours": cost_hours}

    # ============= Helper Methods =============

    def _apply_suggestion_to_tool(self, train_tool: TrainTool, suggestion: dict) -> TrainTool:
        """Apply Protein suggestion to a TrainTool instance."""
        for key_path, value in flatten_config(suggestion).items():
            train_tool = train_tool.override(key_path, str(value))
            logger.debug(f"Applied {key_path} = {value}")
        return train_tool


def get_protein_sweep(
    sweep_name: str,
    protein_config: ProteinConfig,
    train_tool_factory: Callable[[str], TrainTool],
    wandb_cfg: WandbConfig,
    num_trials: int,
    evaluation_simulations: Sequence[SimulationConfig],
    **kwargs,
) -> Pipeline:
    """Get a sweep pipeline for use with external orchestration.

    Returns a Pipeline that should be run with a context containing trial_index.
    
    Note: Consider using SequentialSweep.get_pipeline() instead for cleaner API.
    """
    # Initialize experiment (shared across trials)
    exp = SweepExperiment(
        sweep_name=sweep_name,
        protein_config=protein_config,
        train_tool_factory=train_tool_factory,
        wandb_cfg=wandb_cfg,
        evaluation_simulations=evaluation_simulations,
        **kwargs,
    )

    # Build trial pipeline (run for each trial)
    trial_pipeline = (
        Pipeline()
        .io("init_services", exp.initialize_services)
        .io("load_previous_observations", exp.load_previous_observations)
        .stage("suggest", exp.suggest_hyperparameters)
        .through(
            dict,
            hooks=[
                lambda r, ctx: logger.info(f"Starting trial {ctx.metadata['trial_index'] + 1}: Generated suggestion")
            ],
        )
        .io("get_run_name", exp.get_run_name)
        .stage("train", exp.train_model)  # Training is computation, not I/O
        .stage("evaluate", exp.evaluate_model)  # Evaluation is computation, not I/O
        .io("fetch_metrics", exp.fetch_metrics)
        .stage("calculate_metrics", exp.calculate_metrics)
        .through(
            dict,
            hooks=[
                lambda r, ctx: logger.info(
                    f"Finished trial {ctx.metadata['trial_index'] + 1}: "
                    f"score={r['score']:.4f}, cost={r['cost_hours']:.4f}h"
                )
            ],
        )
        .io("record_wandb", exp.record_to_wandb)
    )

    return trial_pipeline


def get_sequential_sweep_pipeline(config: SweepConfig) -> Pipeline:
    """Get a sequential sweep pipeline using the new clean API.
    
    This is the preferred way to create sweep pipelines.
    
    Args:
        config: Complete sweep configuration
        
    Returns:
        Pipeline configured for sequential sweep execution
    """
    return SequentialSweepPipeline.get_pipeline(config)
