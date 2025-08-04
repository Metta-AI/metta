"""Train and eval pipeline following the canonical sequential_sweep pattern.

This demonstrates the lean Axiom pattern:
1. Pipeline builder class that accepts factories
2. Clear I/O vs stage distinction
3. Simple Tool wrapper for orchestration
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Callable, Optional

import torch

from metta.agent.agent_config import AgentConfig
from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.axiom import Ctx, Pipeline
from metta.tools.sim_pipeline import SimJobPipeline
from metta.tools.train_pipeline import TrainJobPipeline


class TrainAndEvalPipeline:
    """Pipeline builder for train and eval workflow.
    
    Following the canonical pattern from sequential_sweep:
    - Accepts factories (not configs) as arguments
    - Has clear initialization if needed
    - Builds pipelines with explicit I/O vs stage distinction
    - Handles state flow through typed returns
    """
    
    def __init__(
        self,
        train_tool_factory: Callable[[], TrainJobPipeline],
        eval_tool_factory: Callable[[str], SimJobPipeline],
        run_name: Optional[str] = None,
    ):
        """Initialize with factories.
        
        Args:
            train_tool_factory: Factory that creates configured TrainJobPipeline
            eval_tool_factory: Factory that creates SimJobPipeline given checkpoint path
            run_name: Optional run name (will be auto-generated if not provided)
        """
        self.train_tool_factory = train_tool_factory
        self.eval_tool_factory = eval_tool_factory
        self.run_name = run_name or f"train_eval_{uuid.uuid4().hex[:8]}"
        
        # State that accumulates during pipeline
        self.train_tool = None
        self.sim_tool = None
        self.start_time = None
    
    def build_pipeline(self) -> Pipeline:
        """Build the train and eval pipeline."""
        return (
            Pipeline()
            .stage("initialize", self._initialize)
            .stage("train", self._train_model)
            .stage("evaluate", self._evaluate_model)
            .io("save_manifest", self._save_manifest)
        )
    
    # ============= Pipeline Stages =============
    
    def _initialize(self):
        """Initialize the pipeline state."""
        self.start_time = time.time()
        return {
            "run_name": self.run_name,
            "start_timestamp": datetime.utcnow().isoformat(),
        }
    
    def _train_model(self, init_state):
        """Stage: Execute training.
        
        Training is deterministic given config and seed, so it's a stage.
        """
        # Create train tool from factory
        self.train_tool = self.train_tool_factory()
        
        # Set run name if not already set
        if not self.train_tool.run:
            self.train_tool.run = self.run_name
        
        # Train
        train_start = time.time()
        result = self.train_tool.invoke({}, [])
        train_time = time.time() - train_start
        
        if result != 0 and result is not None:
            raise RuntimeError(f"Training failed with code {result}")
        
        checkpoint_dir = self.train_tool.trainer.checkpoint.checkpoint_dir
        
        return {
            **init_state,
            "train_status": "complete",
            "checkpoint_dir": checkpoint_dir,
            "train_time": train_time,
        }
    
    def _evaluate_model(self, train_state):
        """Stage: Run evaluation simulations."""
        if not self.train_tool:
            return {**train_state, "eval_status": "skipped"}
        
        checkpoint_dir = train_state["checkpoint_dir"]
        
        # Create sim tool from factory
        self.sim_tool = self.eval_tool_factory(f"file://{checkpoint_dir}")
        
        # Evaluate
        eval_start = time.time()
        result = self.sim_tool.invoke({}, [])
        eval_time = time.time() - eval_start
        
        if result != 0 and result is not None:
            raise RuntimeError(f"Evaluation failed with code {result}")
        
        return {
            **train_state,
            "eval_status": "complete",
            "eval_time": eval_time,
            "num_simulations": len(self.sim_tool.simulations) if self.sim_tool else 0,
        }
    
    def _save_manifest(self, final_state):
        """I/O: Save experiment manifest."""
        duration = time.time() - self.start_time
        
        # Build manifest
        manifest = {
            "experiment": {
                "run_name": self.run_name,
                "pipeline": "train_and_eval",
            },
            "timing": {
                "start_time": final_state["start_timestamp"],
                "end_time": datetime.utcnow().isoformat(),
                "duration_seconds": duration,
                "train_time": final_state.get("train_time", 0),
                "eval_time": final_state.get("eval_time", 0),
            },
            "results": {
                "training": {
                    "status": final_state.get("train_status", "skipped"),
                    "checkpoint_dir": final_state.get("checkpoint_dir"),
                },
                "evaluation": {
                    "status": final_state.get("eval_status", "skipped"),
                    "num_simulations": final_state.get("num_simulations", 0),
                },
            },
        }
        
        # Add system info if available
        if self.train_tool and hasattr(self.train_tool, "system"):
            manifest["system"] = {
                "device": self.train_tool.system.device,
                "seed": self.train_tool.system.seed,
                "torch_deterministic": self.train_tool.system.torch_deterministic,
            }
        
        # Save manifest
        run_dir = os.path.dirname(final_state.get("checkpoint_dir", "./"))
        manifest_path = os.path.join(run_dir, f"{self.run_name}_manifest.json")
        os.makedirs(run_dir, exist_ok=True)
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Manifest saved to {manifest_path}")
        return {**final_state, "manifest_path": manifest_path, "manifest": manifest}


class TrainAndEvalTool(Tool):
    """Tool wrapper for train and eval pipeline.
    
    This follows the canonical pattern from SweepTool:
    - Thin wrapper that handles orchestration
    - Creates pipeline builder
    - Runs pipeline with context
    """
    
    # Configuration
    trainer: TrainerConfig = TrainerConfig()
    agent: AgentConfig = AgentConfig()
    wandb: WandbConfig = WandbConfig.Unconfigured()
    system: SystemConfig = SystemConfig()
    
    # Evaluation simulations
    simulations: list[SimulationConfig] = []
    
    # Run configuration
    run: Optional[str] = None
    run_dir: Optional[str] = None
    
    # Skip training and use existing policy
    policy_path: Optional[str] = None
    
    consumed_args: list[str] = ["run"]
    
    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Execute the train and eval pipeline."""
        # Process run name from args if provided
        if "run" in args:
            self.run = args["run"]
        
        # Auto-generate run name if needed
        if not self.run:
            self.run = f"train_eval_{uuid.uuid4().hex[:8]}"
        
        # Create factories
        def train_factory() -> TrainJobPipeline:
            """Create configured TrainJobPipeline."""
            if self.policy_path:
                return None  # Skip training
            
            return TrainJobPipeline(
                run=self.run,
                run_dir=self.run_dir,
                trainer=self.trainer,
                policy_architecture=self.agent,
                wandb=self.wandb,
                system=self.system,
            )
        
        def eval_factory(checkpoint_path: str) -> SimJobPipeline:
            """Create configured SimJobPipeline."""
            if not self.simulations:
                return None  # Skip evaluation
            
            return SimJobPipeline(
                run=f"{self.run}_eval",
                simulations=self.simulations,
                policy_uris=self.policy_path or checkpoint_path,
                system=self.system,
                agent=self.agent,
                wandb=self.wandb,
            )
        
        # Create pipeline builder
        pipeline_builder = TrainAndEvalPipeline(
            train_tool_factory=train_factory if not self.policy_path else lambda: None,
            eval_tool_factory=eval_factory,
            run_name=self.run,
        )
        
        # Build and run pipeline
        pipeline = pipeline_builder.build_pipeline()
        ctx = Ctx()
        ctx.metadata["run"] = self.run
        
        try:
            result = pipeline.run(ctx)
            if result and "manifest" in result:
                print(f"Pipeline complete: {result['manifest']['experiment']['run_name']}")
            return 0
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return 1


# ============= Factory Functions =============

def create_quick_train_eval(
    name: str = "quick_test",
    total_timesteps: int = 10000,
    num_eval_episodes: int = 5,
) -> TrainAndEvalTool:
    """Create a quick train and eval tool for testing.
    
    This shows the pattern of creating configured tools.
    """
    from metta.mettagrid.config import envs as eb
    
    # Create base environment
    env = eb.make_arena(num_agents=4)
    
    # Configure trainer
    trainer = TrainerConfig()
    trainer.total_timesteps = total_timesteps
    trainer.checkpoint.checkpoint_interval = 100
    
    # Configure evaluation
    simulations = [
        SimulationConfig(
            name="arena/basic",
            env=env,
            num_episodes=num_eval_episodes,
            max_time_s=60,
        )
    ]
    
    # Create tool
    return TrainAndEvalTool(
        run=name,
        trainer=trainer,
        simulations=simulations,
    )