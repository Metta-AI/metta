"""Clean MVP implementation of Train and Eval experiments."""

import os
import uuid
from typing import Optional

import torch
from pydantic import Field

from metta.agent.agent_config import AgentConfig
from metta.cogworks.curriculum import bucketed, env_curriculum
from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
import metta.mettagrid.config.envs as eb
from metta.rl.trainer_config import TrainerConfig
from metta.rl.system_config import SystemConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.axiom.experiment import AxiomExperiment
from metta.sweep.axiom.experiment_spec import AxiomControls, ExperimentSpec
from metta.tools.sim_pipeline import SimJobPipeline
from metta.tools.train_pipeline import TrainJobPipeline


class TrainAndEvalSpec(ExperimentSpec):
    """Specification for train and eval experiments.

    This spec defines exactly what's needed to run a training
    followed by evaluation experiment.
    """

    # System configuration
    system_config: SystemConfig = Field(
        default_factory=SystemConfig, description="System configuration (device, data_dir, etc.)"
    )

    # Agent configuration
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig, description="Agent/policy architecture configuration"
    )

    # WandB configuration
    wandb_config: WandbConfig = Field(
        default_factory=lambda: WandbConfig.Unconfigured(), description="Weights & Biases configuration"
    )

    # Typed configurations using existing classes from the codebase
    trainer_config: TrainerConfig = Field(default_factory=TrainerConfig, description="Training configuration")

    eval_configs: list[SimulationConfig] = Field(default_factory=list, description="List of evaluation configurations")

    # Policy path for evaluation (if not training fresh)
    policy_path: Optional[str] = Field(default=None, description="Path to existing policy (skip training if provided)")


class TrainAndEvalExperiment(AxiomExperiment):
    """Train and eval experiment implementation.

    Takes a TrainAndEvalSpec and executes:
    1. Training (unless policy_path is provided)
    2. Evaluation on trained/provided policy
    """

    def __init__(self, spec: TrainAndEvalSpec):
        """Initialize from spec.

        Args:
            spec: TrainAndEvalSpec with all configuration
        """
        # Initialize AxiomExperiment base
        super().__init__(spec=spec)
        
        # Type hint for IDE
        self.spec: TrainAndEvalSpec = spec

        # Create tools from configs
        self._create_tools()

    def _create_tools(self):
        """Create TrainJobPipeline and SimJobPipeline instances from spec configs."""
        # Generate run name
        self.run_name = f"{self.spec.name}_{uuid.uuid4().hex[:8]}"

        # Create TrainJobPipeline if we're training
        if not self.spec.policy_path:
            self.run_dir = os.path.join(self.spec.run_dir, self.run_name)
            print(f"DEBUG: Creating TrainJobPipeline with total_timesteps={self.spec.trainer_config.total_timesteps}")
            self.train_tool = TrainJobPipeline(
                run=self.run_name,
                run_dir=self.run_dir,  # Set run_dir explicitly
                trainer=self.spec.trainer_config,
                policy_architecture=self.spec.agent_config,
                wandb=self.spec.wandb_config,
                system=self.spec.system_config,  # System config from Tool base class
            )
            print(f"DEBUG: TrainJobPipeline.trainer.total_timesteps={self.train_tool.trainer.total_timesteps}")
            # Checkpoint dir will be auto-set by TrainJobPipeline if not already set
            if not self.train_tool.trainer.checkpoint.checkpoint_dir:
                self.train_tool.trainer.checkpoint.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        else:
            self.train_tool = None
            self.run_dir = self.spec.run_dir

        # Create a single SimJobPipeline with all eval configs
        # Determine policy URI once
        if self.spec.policy_path:
            policy_uri = self.spec.policy_path
        else:
            policy_uri = f"file://{self.train_tool.trainer.checkpoint.checkpoint_dir}"
        
        # Create single simulation tool with all configs
        if self.spec.eval_configs:
            self.sim_tool = SimJobPipeline(
                run=f"{self.run_name}_eval",
                simulations=self.spec.eval_configs,  # Pass all simulations
                policy_uris=policy_uri,  # Note: policy_uris (with 's')
                system=self.spec.system_config,  # System config from Tool base class
                agent=self.spec.agent_config,
                wandb=self.spec.wandb_config,
            )
        else:
            self.sim_tool = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Execute the train and eval experiment.
        
        This runs the training and evaluation tools in sequence,
        using regular imperative code flow instead of pipeline composition.
        """
        import time
        from datetime import datetime
        
        start_time = time.time()
        start_timestamp = datetime.utcnow().isoformat()
        results = {}
        
        # Capture seed information from system config
        seed_info = {
            'seed': self.spec.system_config.seed,
            'torch_deterministic': self.spec.system_config.torch_deterministic,
            'device': self.spec.system_config.device,
            'vectorization': self.spec.system_config.vectorization,
        }
        
        # Run training if configured
        if self.train_tool:
            print(f"Training for {self.spec.trainer_config.total_timesteps} timesteps")
            
            # Run training - pass empty args since tool is already configured
            # The tool was created with all necessary config in __init__
            train_result = self.train_tool.invoke({}, [])
            if train_result != 0 and train_result is not None:
                print(f"Training failed with code {train_result}")
                return train_result
                
            results['training'] = {
                'status': 'complete',
                'checkpoint_dir': self.train_tool.trainer.checkpoint.checkpoint_dir
            }
            print(f"Training complete. Checkpoints saved to {self.train_tool.trainer.checkpoint.checkpoint_dir}")
        else:
            print(f"Skipping training - using policy from {self.spec.policy_path}")
            results['training'] = {
                'status': 'skipped',
                'policy_path': self.spec.policy_path
            }
        
        # Run evaluation if configured
        if self.sim_tool:
            print(f"Running {len(self.spec.eval_configs)} evaluation simulations")
            
            # Run evaluation - pass empty args since tool is already configured
            sim_result = self.sim_tool.invoke({}, [])
            if sim_result != 0 and sim_result is not None:
                print(f"Evaluation failed with code {sim_result}")
                return sim_result
                
            results['evaluation'] = {
                'status': 'complete',
                'num_simulations': len(self.spec.eval_configs)
            }
            print(f"Evaluation complete")
        else:
            print("No evaluation configurations provided")
            results['evaluation'] = {'status': 'skipped'}
        
        # Save results manifest with more details
        manifest_path = os.path.join(self.spec.run_dir, f"{self.run_name}_manifest.json")
        
        end_time = time.time()
        duration_seconds = end_time - start_time
        
        # Save individual config files for reproducibility
        config_dir = os.path.join(self.run_dir, 'configs')
        os.makedirs(config_dir, exist_ok=True)
        
        # Save training config
        training_config = None
        if self.train_tool:
            training_config_path = os.path.join(config_dir, 'training.json')
            with open(training_config_path, 'w') as f:
                import json
                json.dump(self.train_tool.model_dump(), f, indent=2)
            training_config = training_config_path
        
        # Save evaluation config
        eval_config = None
        if self.sim_tool:
            eval_config_path = os.path.join(config_dir, 'evaluation.json')
            with open(eval_config_path, 'w') as f:
                import json
                json.dump(self.sim_tool.model_dump(), f, indent=2)
            eval_config = eval_config_path
        
        # Save experiment spec
        spec_config_path = os.path.join(config_dir, 'experiment_spec.json')
        with open(spec_config_path, 'w') as f:
            import json
            json.dump(self.spec.model_dump(), f, indent=2)
        
        # Add experiment metadata to results
        full_manifest = {
            'experiment': {
                'name': self.spec.name,
                'run_name': self.run_name,
                'total_timesteps': self.spec.trainer_config.total_timesteps if not self.spec.policy_path else 0,
                'num_agents': self.spec.agent_config.num_agents if hasattr(self.spec.agent_config, 'num_agents') else 'default',
                'num_eval_episodes': self.spec.eval_configs[0].num_episodes if self.spec.eval_configs else 0,
            },
            'timing': {
                'start_time': start_timestamp,
                'end_time': datetime.utcnow().isoformat(),
                'duration_seconds': duration_seconds,
            },
            'reproducibility': seed_info,
            'config_files': {
                'training': training_config,
                'evaluation': eval_config,
                'experiment_spec': spec_config_path,
            },
            'results': results,
            'paths': {
                'run_dir': self.run_dir,
                'checkpoints': self.train_tool.trainer.checkpoint.checkpoint_dir if self.train_tool else None,
                'manifest': manifest_path,
                'configs': config_dir,
            }
        }
        
        with open(manifest_path, 'w') as f:
            import json
            json.dump(full_manifest, f, indent=2)
        print(f"Results saved to {manifest_path}")
        print(f"Configs saved to {config_dir}")
        
        return 0  # Success


# Factory functions for common configurations


def create_experiment(
    name: str = "experiment",
    total_timesteps: int = 10000,
    num_agents: int = 4,
    rollout_workers: int = None,
    batch_size: int = None,
    minibatch_size: int = None,
    learning_rate: float = None,
    num_eval_episodes: int = 5,
    policy_path: Optional[str] = None,
    use_curriculum: bool = True,
    enable_combat: bool = False,
) -> TrainAndEvalSpec:
    """Create a train and eval experiment spec.

    This single factory function handles all experiment creation needs.

    Args:
        name: Experiment name
        total_timesteps: Total training timesteps (0 for eval-only)
        num_agents: Number of agents in environment
        rollout_workers: Number of parallel rollout workers (None = use default)
        batch_size: PPO batch size (None = use default)
        minibatch_size: PPO minibatch size (None = use default)
        learning_rate: Optimizer learning rate (None = use default)
        num_eval_episodes: Episodes per evaluation
        policy_path: Path to existing policy (for eval-only)
        use_curriculum: Whether to use curriculum learning
        enable_combat: Whether to enable combat in evaluation

    Returns:
        Configured TrainAndEvalSpec
    """
    # Create base environment
    base_env = eb.make_arena(num_agents=num_agents)

    # Start with default trainer config and modify as needed
    trainer = TrainerConfig()
    if not policy_path:
        # Always set total timesteps
        trainer.total_timesteps = total_timesteps
        
        # Only override if provided by user
        if rollout_workers is not None:
            trainer.rollout_workers = rollout_workers
        if batch_size is not None:
            trainer.batch_size = batch_size
        if minibatch_size is not None:
            trainer.minibatch_size = minibatch_size
        if learning_rate is not None:
            trainer.optimizer.learning_rate = learning_rate

        # Set up curriculum if requested
        if use_curriculum and num_agents > 4:
            # Use bucketed curriculum for larger experiments
            curriculum_tasks = bucketed(base_env)
            for item in ["ore_red", "battery_red", "laser", "armor"]:
                curriculum_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 1.0])
            trainer.curriculum = curriculum_tasks.to_curriculum()
        else:
            # Simple environment curriculum
            trainer.curriculum = env_curriculum(base_env)

    # Configure evaluation environments
    eval_configs = []

    # Basic evaluation (no combat)
    basic_env = base_env.model_copy()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100  # Disable combat
    eval_configs.append(
        SimulationConfig(
            name="arena/basic",
            env=basic_env,
            num_episodes=num_eval_episodes,
            max_time_s=60 if num_eval_episodes <= 10 else 300,
        )
    )

    # Combat evaluation (if enabled)
    if enable_combat:
        combat_env = base_env.model_copy()
        combat_env.game.actions.attack.consumed_resources["laser"] = 1
        eval_configs.append(
            SimulationConfig(
                name="arena/combat",
                env=combat_env,
                num_episodes=num_eval_episodes,
                max_time_s=60 if num_eval_episodes <= 10 else 300,
            )
        )

    # Configure system with proper seed
    system = SystemConfig()
    system.device = "cuda" if torch.cuda.is_available() else "cpu"
    system.seed = 42  # Default seed for reproducibility
    system.torch_deterministic = True  # Enable deterministic mode for PyTorch

    # Build spec
    return TrainAndEvalSpec(
        name=name,
        description=f"{'Eval-only' if policy_path else 'Train and eval'} experiment",
        trainer_config=trainer,
        eval_configs=eval_configs,
        policy_path=policy_path,
        controls=AxiomControls(
            seed=42,
            enforce_determinism=True,
        ),
        system_config=system,
        agent_config=AgentConfig(),  # Use default agent architecture
        wandb_config=WandbConfig.Unconfigured(),  # User can override if needed
    )


# Example usage
def main():
    """Example of running train and eval experiment."""
    print("Train and Eval Experiment")
    print("=" * 50)

    # Quick test example
    print("\n--- Quick Test Example ---")
    spec = create_experiment(name="quick_test", total_timesteps=10000, num_agents=4, num_eval_episodes=5)

    print(f"Experiment: {spec.name}")
    print(f"Training timesteps: {spec.trainer_config.total_timesteps}")
    print(f"Eval configs: {len(spec.eval_configs)} environments")

    # Full training example
    print("\n--- Full Training Example ---")
    full_spec = create_experiment(
        name="full_training",
        total_timesteps=1000000,
        num_agents=24,
        rollout_workers=8,
        batch_size=512,
        minibatch_size=64,
        num_eval_episodes=100,
        enable_combat=True,
    )
    print(f"Created full training spec: {full_spec.name}")

    # Eval-only example
    print("\n--- Eval-Only Example ---")
    eval_spec = create_experiment(
        name="eval_only",
        policy_path="file://./checkpoints/trained_policy",
        num_agents=24,
        num_eval_episodes=100,
        enable_combat=True,
    )
    print(f"Created eval-only spec: {eval_spec.name}")

    # Run the quick test
    print("\n--- Running Quick Test ---")
    experiment = TrainAndEvalExperiment(spec)
    experiment.prepare()

    try:
        result = experiment.run()
        manifest = result.manifest()
        print(f"Status: {manifest.get('pipeline_result', {}).get('status', 'unknown')}")
    except Exception as e:
        print(f"Note: Example run requires actual training environment. Error: {e}")


if __name__ == "__main__":
    main()

