"""Pipeline-based implementation of SimTool using tAXIOM.

This follows the same pattern as train_pipeline.py, decomposing the simulation
tool into an explicit pipeline with clear stages and I/O operations.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch

from metta.agent.agent_config import AgentConfig
from metta.agent.policy_store import PolicyStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.eval.eval_service import evaluate_policy
from metta.rl.stats import process_policy_evaluator_stats
from metta.common.util.constants import SOFTMAX_S3_BASE
from metta.rl.system_config import SystemConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.axiom import Pipeline
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """State object for simulation pipeline."""

    # Configuration
    run: str
    simulations: Sequence[SimulationConfig]
    policy_uri: str
    system: SystemConfig
    agent: AgentConfig
    wandb: WandbConfig
    stats_db_uri: Optional[str]  # Export stats to this URI

    # Runtime state
    device: Optional[torch.device] = None
    policy_store: Optional[PolicyStore] = None
    stats_client: Optional[StatsClient] = None
    policy_record: Optional[object] = None
    eval_results: Optional[object] = None  # Will be EvalResults
    eval_db_path: Optional[str] = None


class SimJobPipeline(Tool):
    """Simulation tool with pipeline support.

    This mirrors SimTool but supports the pipeline pattern.
    """

    # Tool configuration - matching SimTool interface
    run: str
    simulations: Sequence[SimulationConfig]  # List of simulations to run
    policy_uris: str | Sequence[str] | None = None  # Policy URI(s) to evaluate

    # Optional configurations
    agent: AgentConfig = AgentConfig()
    wandb: WandbConfig = auto_wandb_config()
    selector_type: str = "latest"  # Policy selector type
    
    # Stats and eval configuration
    stats_dir: Optional[str] = None
    stats_db_uri: Optional[str] = None  # Export stats to this URL
    stats_server_uri: Optional[str] = None
    eval_task_id: Optional[str] = None
    push_metrics_to_wandb: bool = False
    
    # Replay directory (default matches SimTool)
    replay_dir: str = f"{SOFTMAX_S3_BASE}/replays/{str(uuid.uuid4())}"

    def get_pipeline(self) -> Pipeline:
        """Build the simulation pipeline."""
        return (
            Pipeline()
            .stage("initialize", self._initialize)
            .io("setup_device", self._setup_device)
            .io("create_policy_store", self._create_policy_store)
            .io("load_policy", self._load_policy)
            .io("run_evaluation", self._run_evaluation)
            .io("save_results", self._save_results)
            .stage("finalize", self._finalize)
        )
    
    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Execute the simulation pipeline."""
        # Run the pipeline
        pipeline = self.get_pipeline()
        result = pipeline.run()

        # Return success
        return 0 if result.get("status") == "complete" else 1
    
    def _initialize(self, state: dict) -> SimulationState:
        """Initialize simulation state."""
        # Handle policy_uris being string or list
        if isinstance(self.policy_uris, str):
            policy_uri = self.policy_uris
        elif isinstance(self.policy_uris, list) and len(self.policy_uris) > 0:
            policy_uri = self.policy_uris[0]  # For now, use first policy
        else:
            raise ValueError("policy_uris must be provided")
        
        # Determine run name based on policy URI (matches SimTool)
        if not self.run:
            if policy_uri.startswith("file://"):
                checkpoint_path = Path(policy_uri.replace("file://", ""))
                self.run = f"eval_{checkpoint_path.stem}"
            elif policy_uri.startswith("wandb://"):
                artifact_part = policy_uri.split("/")[-1]
                self.run = f"eval_{artifact_part.replace(':', '_')}"
            else:
                self.run = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Initializing simulations: {self.run}")
        logger.info(f"  Policy: {policy_uri}")
        logger.info(f"  Number of simulations: {len(self.simulations)}")
        for sim in self.simulations:
            logger.info(f"    - {sim.name}: {sim.num_episodes} episodes")

        return SimulationState(
            run=self.run,
            simulations=self.simulations,
            policy_uri=policy_uri,
            system=self.system,
            agent=self.agent,
            wandb=self.wandb,
            stats_db_uri=self.stats_db_uri,
        )

    def _setup_device(self, state: SimulationState) -> SimulationState:
        """Set up compute device."""
        state.device = torch.device(state.system.device)
        logger.info(f"Using device: {state.device}")
        return state

    def _create_policy_store(self, state: SimulationState) -> SimulationState:
        """Create policy store for loading policies."""
        state.policy_store = PolicyStore.create(
            device=state.system.device,
            data_dir=state.system.data_dir,
            wandb_config=state.wandb,
            wandb_run=None,
        )
        logger.info("Policy store created")
        
        # Create stats client if configured
        if self.stats_server_uri:
            state.stats_client = StatsClient.create(self.stats_server_uri)
        
        return state

    def _load_policy(self, state: SimulationState) -> SimulationState:
        """Load policy from URI."""
        logger.info(f"Loading policy from: {state.policy_uri}")

        # Get policy records
        # Use first simulation's name for metric, matching SimTool behavior
        metric = state.simulations[0].name + "_score" if state.simulations else None
        policy_records = state.policy_store.policy_records(
            uri_or_config=state.policy_uri,
            selector_type="latest",
            n=1,
            metric=metric,
        )

        if not policy_records:
            raise ValueError(f"No policy found at {state.policy_uri}")

        state.policy_record = policy_records[0]
        logger.info(f"Loaded policy: {state.policy_record.uri}")

        return state

    def _run_evaluation(self, state: SimulationState) -> SimulationState:
        """Run the actual evaluations for all simulations."""
        logger.info(f"Running evaluations for {len(state.simulations)} simulations")
        
        # Get eval_task_id if configured
        eval_task_id = None
        if self.eval_task_id:
            eval_task_id = uuid.UUID(self.eval_task_id)
        
        # Build replay directory path
        replay_dir = None
        if self.replay_dir:
            replay_dir = f"{self.replay_dir}/{state.run}/{state.policy_record.run_name}"
        
        # Match SimTool's evaluate_policy call exactly
        eval_results = evaluate_policy(
            policy_record=state.policy_record,
            simulations=list(state.simulations),  # Pass all simulations at once
            device=state.device,
            vectorization=state.system.vectorization,
            stats_dir=self.stats_dir,  # Matches SimTool's stats_dir
            replay_dir=replay_dir,
            export_stats_db_uri=state.stats_db_uri,
            stats_epoch_id=None,
            wandb_policy_name=None,
            eval_task_id=eval_task_id,
            policy_store=state.policy_store,
            stats_client=state.stats_client,
            logger=logger,
        )
        
        # Process wandb metrics if configured  
        if self.push_metrics_to_wandb and state.policy_record:
            try:
                process_policy_evaluator_stats(state.policy_record, eval_results)
            except Exception as e:
                logger.error(f"Error logging evaluation results to wandb: {e}")
        
        # Store the eval results in the same format as SimTool
        state.eval_results = eval_results
        return state

    def _save_results(self, state: SimulationState) -> SimulationState:
        """Save evaluation results."""
        # Save to eval database if configured
        if state.stats_db_uri:
            logger.info(f"Saving results to: {state.stats_db_uri}")
            # The actual export happens in evaluate_policy via export_stats_db_uri
            state.eval_db_path = state.stats_db_uri

        # Log to wandb if configured
        if state.wandb and hasattr(state.wandb, "project"):
            logger.info("Logging results to WandB")
            # In a real implementation, this would log to wandb
            # The actual wandb logging happens in evaluate_policy

        return state

    def _finalize(self, state: SimulationState) -> dict:
        """Finalize and return results."""
        logger.info(f"All simulations complete: {state.run}")
        
        # Build results in same format as SimTool
        if state.eval_results:
            # Extract scores and replay URLs from eval_results
            result = {
                "run": state.run,
                "num_simulations": len(state.simulations),
                "simulations": [sim.name for sim in state.simulations],
                "policy_uri": state.policy_uri,
                "metrics": {
                    "reward_avg": state.eval_results.scores.avg_simulation_score,
                    "reward_avg_category_normalized": state.eval_results.scores.avg_category_score,
                    "detailed": state.eval_results.scores.to_wandb_metrics_format(),
                },
                "replay_url": state.eval_results.replay_urls,
                "status": "complete",
            }
        else:
            result = {
                "run": state.run,
                "num_simulations": len(state.simulations),
                "simulations": [sim.name for sim in state.simulations],
                "policy_uri": state.policy_uri,
                "status": "complete",
            }
        
        return result

