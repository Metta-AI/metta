"""Functional trainer with explicit, modifiable training loop."""

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

from metta.rl.checkpointer import TrainingCheckpointer
from metta.rl.collectors import RolloutCollector
from metta.rl.configs import PPOConfig, TrainerConfig
from metta.rl.evaluator import PolicyEvaluator
from metta.rl.experience import Experience
from metta.rl.optimizers import PPOOptimizer
from metta.rl.stats_logger import StatsLogger
from metta.rl.vecenv import make_vecenv

if TYPE_CHECKING:
    from metta.agent import BaseAgent
    from metta.agent.policy_store import PolicyStore
    from mettagrid.curriculum import Curriculum

logger = logging.getLogger(__name__)


class TrainingState:
    """Mutable state for the training loop."""

    def __init__(self):
        self.epoch = 0
        self.agent_steps = 0
        self.should_stop = False
        self.evals = {}


def create_training_components(
    config: TrainerConfig,
    ppo_config: PPOConfig,
    policy: "BaseAgent",
    vecenv,
    experience: Experience,
    policy_store: "PolicyStore",
    wandb_run=None,
) -> Dict:
    """Create all training components from configs."""
    device = torch.device(config.device)

    # Optimizer
    if config.optimizer_type == "adam":
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    else:
        from heavyball import ForeachMuon

        optimizer = ForeachMuon(policy.parameters(), lr=config.learning_rate)

    # Components
    components = {
        "collector": RolloutCollector(vecenv, policy, experience, device),
        "ppo": PPOOptimizer(policy, optimizer, device, **ppo_config.__dict__),
        "checkpointer": TrainingCheckpointer(config.checkpoint_dir, policy_store, wandb_run),
        "stats_logger": StatsLogger(wandb_run),
        "optimizer": optimizer,
    }

    # Optional evaluator
    if config.evaluate_interval > 0:
        components["evaluator"] = PolicyEvaluator(
            sim_suite_config=None,  # Would come from config
            policy_store=policy_store,
            device=device,
        )

    return components


def default_training_step(
    state: TrainingState,
    components: Dict,
    config: TrainerConfig,
    experience: Experience,
    custom_losses: Optional[List[Callable]] = None,
) -> Dict[str, float]:
    """Single training iteration with support for custom losses.

    Args:
        state: Mutable training state
        components: Dict of training components
        config: Training configuration
        experience: Experience buffer
        custom_losses: Optional list of callables that compute additional losses

    Returns:
        Dict of metrics from this step
    """
    metrics = {}

    # 1. Collect rollouts
    rollout_stats, steps = components["collector"].collect()
    state.agent_steps = components["collector"].agent_steps
    components["stats_logger"].add_stats(rollout_stats)

    # 2. PPO update with optional custom losses
    if custom_losses:
        # Allow injection of custom losses into PPO update
        # This is a simplified version - could be made more flexible
        loss_stats = components["ppo"].update(
            experience=experience,
            update_epochs=4,  # From config
            custom_loss_fns=custom_losses,
        )
    else:
        loss_stats = components["ppo"].update(experience=experience, update_epochs=4)

    metrics.update(loss_stats)

    # 3. Optional evaluation
    if config.evaluate_interval > 0 and state.epoch % config.evaluate_interval == 0:
        if "evaluator" in components:
            eval_metrics = components["evaluator"].evaluate(
                policy_record=None,  # Would come from checkpointer
            )
            state.evals.update(eval_metrics)
            metrics.update(eval_metrics)

    # 4. Checkpointing
    if state.epoch % config.checkpoint_interval == 0:
        components["checkpointer"].save_trainer_state(
            run_dir=".",
            agent_step=state.agent_steps,
            epoch=state.epoch,
            optimizer_state_dict=components["optimizer"].state_dict(),
        )

    # 5. Logging
    components["stats_logger"].process_and_log(
        agent_step=state.agent_steps,
        epoch=state.epoch,
        timer=None,  # Simplified for now
        losses=components["ppo"].losses,
        experience=experience,
        policy=components["collector"].policy,
        system_monitor=None,  # Simplified
        evals=state.evals,
        trainer_config=config,
    )

    state.epoch += 1
    return metrics


def functional_training_loop(
    config: TrainerConfig,
    ppo_config: PPOConfig,
    policy: "BaseAgent",
    curriculum: "Curriculum",
    policy_store: "PolicyStore",
    step_fn: Optional[Callable] = None,
    custom_losses: Optional[List[Callable]] = None,
    wandb_run=None,
) -> TrainingState:
    """Functional training loop that's easy to modify.

    Args:
        config: Training configuration
        ppo_config: PPO configuration
        policy: Policy to train
        curriculum: Environment curriculum
        policy_store: Policy storage
        step_fn: Optional custom training step function
        custom_losses: Optional list of custom loss functions
        wandb_run: Optional wandb run for logging

    Returns:
        Final training state
    """
    # Setup
    device = torch.device(config.device)
    policy = policy.to(device)

    # Create environment
    num_agents = curriculum.get_task().env_cfg().game.num_agents
    batch_size = config.batch_size
    num_envs = batch_size * config.async_factor

    vecenv = make_vecenv(
        curriculum,
        "serial",  # From config
        num_envs=num_envs,
        batch_size=batch_size,
        num_workers=config.num_workers,
    )

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=config.batch_size,
        bptt_horizon=config.bptt_horizon,
        minibatch_size=config.minibatch_size,
        obs_space=vecenv.single_observation_space,
        atn_space=vecenv.single_action_space,
        device=device,
        hidden_size=256,  # From policy
        cpu_offload=config.cpu_offload,
    )

    # Create components
    components = create_training_components(config, ppo_config, policy, vecenv, experience, policy_store, wandb_run)

    # Training state
    state = TrainingState()

    # Use custom step function if provided
    if step_fn is None:
        step_fn = default_training_step

    # Main training loop - explicit and modifiable
    logger.info("Starting functional training loop")

    while state.agent_steps < config.total_timesteps and not state.should_stop:
        # Single training step
        metrics = step_fn(state, components, config, experience, custom_losses)

        # Log progress
        if state.epoch % 10 == 0:
            logger.info(
                f"Epoch {state.epoch}: {state.agent_steps}/{config.total_timesteps} steps, "
                f"loss={metrics.get('policy_loss', 0):.4f}"
            )

    logger.info("Training complete!")
    vecenv.close()

    return state


# Example of custom training step
def custom_training_step_example(
    state: TrainingState,
    components: Dict,
    config: TrainerConfig,
    experience: Experience,
    custom_losses: Optional[List[Callable]] = None,
) -> Dict[str, float]:
    """Example of a custom training step with additional logic."""
    # Call default step
    metrics = default_training_step(state, components, config, experience, custom_losses)

    # Add custom logic
    if state.epoch % 100 == 0:
        logger.info(f"Custom checkpoint at epoch {state.epoch}")
        # Could add custom checkpointing, metric computation, etc.

    # Early stopping based on custom criteria
    if metrics.get("value_loss", float("inf")) < 0.01:
        logger.info("Early stopping: value loss threshold reached")
        state.should_stop = True

    return metrics


# Example of custom loss function
def curiosity_loss(policy, obs, actions, next_obs) -> torch.Tensor:
    """Example custom loss for curiosity-driven exploration."""
    # Simplified curiosity loss
    # In practice, this would use a forward model to predict next_obs
    # and compute prediction error as intrinsic reward
    return torch.tensor(0.0)  # Placeholder
