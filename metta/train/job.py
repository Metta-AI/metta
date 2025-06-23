"""Training job interface - replaces train_job.yaml with programmatic API."""

from dataclasses import dataclass
from typing import Optional, Union

from metta.agent import BaseAgent, create_agent
from metta.agent.policy_store import PolicyStore
from metta.env.factory import create_env
from metta.rl.trainer import MettaTrainer
from metta.runtime import RuntimeConfig, get_runtime
from metta.sim.registry import get_simulation_suite
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.train.config import AgentConfig, TrainerConfig, TrainingConfig, WandbConfig
from metta.util.wandb.wandb_context import WandbContext


@dataclass
class TrainingJob:
    """Complete training job specification.

    This replaces the train_job.yaml file with a programmatic interface
    that's easier to use as a library.
    """

    # Agent configuration
    agent: Union[str, BaseAgent, AgentConfig] = "simple_cnn"

    # Training configuration
    trainer: Optional[TrainerConfig] = None

    # Evaluation configuration
    evaluations: Union[str, SimulationSuiteConfig] = "quick"

    # Runtime configuration
    runtime: Optional[RuntimeConfig] = None

    # WandB configuration
    wandb: Optional[WandbConfig] = None

    # Additional options
    map_preview_uri: Optional[str] = None
    seed: int = 1

    def __post_init__(self):
        """Initialize with defaults."""
        if self.runtime is None:
            self.runtime = get_runtime()

        if self.trainer is None:
            self.trainer = TrainerConfig()

        if self.wandb is None:
            self.wandb = WandbConfig(
                enabled=True,
                project="metta",
                name=self.runtime.run_name,
            )

    def to_training_config(self) -> TrainingConfig:
        """Convert to TrainingConfig format."""
        # Handle agent configuration
        if isinstance(self.agent, str):
            agent_cfg = AgentConfig(name=self.agent)
        elif isinstance(self.agent, BaseAgent):
            # Extract config from instantiated agent
            agent_cfg = AgentConfig(
                name=self.agent.__class__.__name__.lower().replace("agent", ""),
                hidden_size=getattr(self.agent, "hidden_size", 128),
                lstm_layers=getattr(self.agent, "lstm_layers", 2),
            )
        else:
            agent_cfg = self.agent

        return TrainingConfig(
            run_name=self.runtime.run_name,
            run_dir=str(self.runtime.run_dir),
            data_dir=str(self.runtime.data_dir),
            trainer=self.trainer,
            agent=agent_cfg,
            wandb=self.wandb,
        )

    def get_evaluation_suite(self) -> SimulationSuiteConfig:
        """Get the evaluation suite configuration."""
        if isinstance(self.evaluations, str):
            return get_simulation_suite(self.evaluations)
        return self.evaluations

    def run(self) -> BaseAgent:
        """Run the training job and return the trained agent."""
        # Set runtime configuration
        from metta.runtime import set_runtime

        set_runtime(self.runtime)

        # Create environment
        env = create_env()  # Uses defaults, can be customized

        # Create or get agent
        if isinstance(self.agent, BaseAgent):
            agent = self.agent
        else:
            config = self.to_training_config()
            agent = create_agent(
                agent_name=config.agent.name,
                obs_space=env.observation_space,
                action_space=env.action_space,
                obs_width=env.obs_width,
                obs_height=env.obs_height,
                feature_normalizations=env.feature_normalizations,
                device=str(self.runtime.device),
                **config.agent.kwargs,
            )

        # Create trainer
        config = self.to_training_config()
        config_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__

        # Add runtime config
        config_dict.update(self.runtime.to_dict())

        # Create policy store
        policy_store = PolicyStore(config_dict, None)

        # Create trainer with wandb if enabled
        wandb_context = None
        if self.wandb.enabled:
            wandb_context = WandbContext(self.wandb.__dict__, config_dict)

        with wandb_context as wandb_run:
            trainer = MettaTrainer(
                cfg=config_dict,
                wandb_run=wandb_run,
                policy_store=policy_store,
                sim_suite_config=self.get_evaluation_suite(),
                stats_client=None,
            )

            # Train
            trainer.train()

            return trainer.policy


def quick_train(
    agent: Union[str, BaseAgent] = "simple_cnn",
    total_timesteps: int = 1_000_000,
    evaluations: str = "quick",
    run_name: Optional[str] = None,
    **kwargs,
) -> BaseAgent:
    """Quick training interface for common use cases.

    Args:
        agent: Agent name or instance
        total_timesteps: Total training steps
        evaluations: Evaluation suite name
        run_name: Name for this run
        **kwargs: Additional trainer parameters

    Returns:
        Trained agent

    Example:
        >>> agent = quick_train("simple_cnn", total_timesteps=5_000_000)
        >>> agent = quick_train(MyCustomAgent(), evaluations="navigation")
    """
    if run_name is None:
        agent_name = agent if isinstance(agent, str) else agent.__class__.__name__
        run_name = f"{agent_name}_{total_timesteps}"

    job = TrainingJob(
        agent=agent,
        trainer=TrainerConfig(total_timesteps=total_timesteps, **kwargs),
        evaluations=evaluations,
        runtime=RuntimeConfig(run_name=run_name),
    )

    return job.run()


# Convenience builders for common patterns
class JobBuilder:
    """Fluent interface for building training jobs."""

    def __init__(self):
        self._job = TrainingJob()

    def with_agent(self, agent: Union[str, BaseAgent, AgentConfig]) -> "JobBuilder":
        """Set the agent."""
        self._job.agent = agent
        return self

    def with_timesteps(self, timesteps: int) -> "JobBuilder":
        """Set total timesteps."""
        if self._job.trainer is None:
            self._job.trainer = TrainerConfig()
        self._job.trainer.total_timesteps = timesteps
        return self

    def with_batch_size(self, batch_size: int) -> "JobBuilder":
        """Set batch size."""
        if self._job.trainer is None:
            self._job.trainer = TrainerConfig()
        self._job.trainer.batch_size = batch_size
        return self

    def with_evaluations(self, evaluations: Union[str, SimulationSuiteConfig]) -> "JobBuilder":
        """Set evaluations."""
        self._job.evaluations = evaluations
        return self

    def with_wandb(self, project: str, entity: Optional[str] = None) -> "JobBuilder":
        """Configure WandB."""
        self._job.wandb = WandbConfig(
            enabled=True,
            project=project,
            entity=entity,
        )
        return self

    def without_wandb(self) -> "JobBuilder":
        """Disable WandB."""
        self._job.wandb = WandbConfig(enabled=False)
        return self

    def with_device(self, device: str) -> "JobBuilder":
        """Set device."""
        if self._job.runtime is None:
            self._job.runtime = RuntimeConfig()
        self._job.runtime.device = device
        return self

    def build(self) -> TrainingJob:
        """Build the training job."""
        return self._job

    def run(self) -> BaseAgent:
        """Build and run the training job."""
        return self._job.run()
