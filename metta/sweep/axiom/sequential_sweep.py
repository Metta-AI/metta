"""Sequential sweep implementation using tAXIOM pipeline architecture."""

import logging
import time
from typing import Callable, Optional, Sequence

import wandb
from cogweb.cogweb_client import CogwebClient

from metta.common.util.datastruct import flatten_config
from metta.sweep.axiom import Ctx, Pipeline
from metta.sweep.protein_metta import MettaProtein
from metta.sweep.sweep_config import SweepConfig
from metta.sweep.wandb_utils import (
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)


class SequentialSweepPipeline:
    """A sweep that runs trials sequentially (one after another).
    
    This class IS the sweep - it contains both the operations and
    knows how to build itself into a pipeline. The optimizer
    (Protein, Bayesian, etc.) is just an implementation detail.
    """
    
    def build_pipeline(self) -> Pipeline:
        """Build pipeline from this sweep's operations.
        
        Returns a pipeline configured for a single trial execution.
        Note: init_services should be called before running this pipeline.
        """
        return (
            Pipeline()
            # Load observations for each trial to get latest state
            .io("load_observations", self.load_previous_observations)
            
            # Trial execution sequence
            .stage("suggest", self.suggest_hyperparameters)
            .logf("Trial {ctx.metadata.trial_index}: Generated suggestion")
            .io("get_run_name", self.get_run_name)
            .stage("train", self.train_model)
            .logf("Training completed for {payload.run_name}")
            .stage("evaluate", self.evaluate_model)
            .io("fetch_metrics", self.fetch_metrics)
            .stage("calculate_metrics", self.calculate_metrics)
            .logf("Trial {ctx.metadata.trial_index}: score={payload.score}, cost={payload.cost_hours}h")
            .io("record_wandb", self.record_to_wandb)
            .stage("update_optimizer", self.update_optimizer)
        )
    
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
        
        # Initialize optimizer (could be swapped for different optimizers)
        self.optimizer = MettaProtein(protein_config)
        
        # Services initialized during I/O
        self.cogweb_client = None
        self.sweep_client = None
    
    # ============= I/O Operations (External Input/Output) =============
    
    def initialize_services(self):
        """I/O: Initialize external services (Cogweb)."""
        self.cogweb_client = CogwebClient.get_client(
            base_url=self.sweep_server_uri
        )
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
        
    def load_previous_observations(self):
        """I/O: Load previous observations from WandB."""
        previous_observations = fetch_protein_observations_from_wandb(
            wandb_entity=self.wandb_cfg.entity,
            wandb_project=self.wandb_cfg.project,
            sweep_name=self.sweep_name,
            max_observations=self.max_observations_to_load,
        )
        
        logger.info(f"Loaded {len(previous_observations)} previous observations")
        
        # Update optimizer with observations
        for obs in previous_observations:
            self.optimizer.observe(
                obs["suggestion"],
                obs["objective"],
                obs["cost"],
                obs.get("is_failure", False)
            )
        
        return {"observation_count": len(previous_observations)}
    
    def get_run_name(self, suggestion):
        """I/O: Get next run ID from Cogweb."""
        if self.sweep_client is None:
            raise ValueError("Cogweb client not initialized")
        
        run_name = self.sweep_client.get_next_run_id(self.sweep_name)
        return {
            "suggestion": suggestion["suggestion"],
            "run_name": run_name
        }
    
    def fetch_metrics(self, eval_result):
        """I/O: Fetch metrics from WandB."""
        run_name = eval_result["run_name"]
        
        api = wandb.Api()
        wandb_run = api.run(f"{self.wandb_cfg.entity}/{self.wandb_cfg.project}/{run_name}")
        assert wandb_run is not None, f"Error fetching run: {run_name}"
        
        summary = dict(wandb_run.summary)
        
        return {
            **eval_result,
            "wandb_run": wandb_run,
            "summary": summary
        }
    
    def record_to_wandb(self, trial_result):
        """I/O: Record results to WandB."""
        wandb_run = trial_result["wandb_run"]
        suggestion = trial_result["suggestion"]
        score = trial_result["score"]
        cost_hours = trial_result["cost_hours"]
        trial_index = trial_result.get("trial_index", 0)
        
        # Update wandb config
        wandb_run.config.update({
            "sweep_name": self.sweep_name,
            "protein_suggestion": suggestion,
        })
        
        # Record protein observation
        record_protein_observation_to_wandb(
            wandb_run,
            suggestion,
            score,
            cost_hours,
            is_failure=False,
        )
        
        # Update summary
        wandb_run.summary.update({
            "trial_index": trial_index,
            "trial_score": score,
            "trial_cost_hours": cost_hours,
            self.protein_config.metric: score,
        })
        
        return trial_result
    
    # ============= Stages (Deterministic Computation) =============
    
    def suggest_hyperparameters(self, load_result=None):
        """Stage: Generate next hyperparameter suggestion from optimizer."""
        suggestion, _ = self.optimizer.suggest()
        return {"suggestion": suggestion}
    
    def train_model(self, config):
        """Stage: Execute training with the configured tool.
        
        Training is deterministic given the config and seed, so it's a stage,
        not I/O. The fact that it takes time doesn't make it I/O.
        """
        suggestion = config["suggestion"]
        run_name = config["run_name"]
        
        # Create and configure train tool
        train_tool = self.train_tool_factory(run_name)
        train_tool = self._apply_suggestion_to_tool(train_tool, suggestion)
        
        # Configure wandb
        train_tool.wandb = self.wandb_cfg
        train_tool.wandb.group = self.sweep_name
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
            "train_tool": train_tool,
        }
    
    def evaluate_model(self, train_result):
        """Stage: Run evaluation simulations.
        
        Evaluation is deterministic given the model and config, so it's a stage.
        """
        run_name = train_result["run_name"]
        train_tool = train_result["train_tool"]
        
        # Use the eval_tool_factory to create the SimTool
        sim_tool = self.eval_tool_factory(run_name, train_tool)
        
        eval_start_time = time.time()
        sim_tool.invoke(args={}, overrides=[])
        eval_time = time.time() - eval_start_time
        
        return {**train_result, "eval_time": eval_time}
    
    def calculate_metrics(self, metrics_data):
        """Stage: Calculate score and cost from fetched metrics."""
        summary = metrics_data["summary"]
        train_time = metrics_data["train_time"]
        eval_time = metrics_data["eval_time"]
        
        # Get evaluation score
        eval_score = summary.get(f"evaluator/{self.protein_config.metric}/score")
        
        if eval_score is None:
            raise ValueError(
                f"Error during evaluation, got score: None. "
                f"Check metric: {self.protein_config.metric}"
            )
        
        # Calculate cost
        if "monitor/cost/accrued_total" in summary:
            cost_hours = summary["monitor/cost/accrued_total"]
        elif "total_time" in summary:
            cost_hours = summary["total_time"] / 3600.0
        else:
            cost_hours = (train_time + eval_time) / 3600.0
        
        return {
            **metrics_data,
            "score": eval_score,
            "cost_hours": cost_hours
        }
    
    def update_optimizer(self, trial_result):
        """Stage: Update optimizer with trial results."""
        self.optimizer.observe(
            trial_result["suggestion"],
            objective=trial_result["score"],
            cost=trial_result["cost_hours"],
            is_failure=False
        )
        
        logger.info(
            f"Trial complete: score={trial_result['score']:.4f}, "
            f"cost={trial_result['cost_hours']:.4f}h"
        )
        
        return trial_result
    
    # ============= Pipeline Factory =============
    
    @classmethod
    def get_pipeline(cls, config: SweepConfig) -> Pipeline:
        """Factory method that creates a configured pipeline.
        
        This class method creates an instance and builds its pipeline.
        """
        sweep = cls(
            sweep_name=config.sweep_name,
            protein_config=config.protein,
            train_tool_factory=config.train_tool_factory,
            eval_tool_factory=config.eval_tool_factory,
            wandb_cfg=config.wandb,
            sweep_server_uri=config.sweep_server_uri,
            max_observations_to_load=config.max_observations_to_load,
            stats_client=config.stats_client,
        )
        return sweep.build_pipeline()
    
    # ============= Helper Methods =============
    
    def _apply_suggestion_to_tool(self, train_tool: TrainTool, suggestion: dict) -> TrainTool:
        """Apply optimizer suggestion to a TrainTool instance."""
        for key_path, value in flatten_config(suggestion).items():
            train_tool = train_tool.override(key_path, str(value))
            logger.debug(f"Applied {key_path} = {value}")
        return train_tool
