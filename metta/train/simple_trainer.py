"""Simplified trainer that doesn't rely on Hydra."""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from metta.agent import BaseAgent, create_agent
from metta.rl.trainer import MettaTrainer
from metta.train.config import TrainingConfig
from metta.util.wandb.wandb_context import WandbContext
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger(__name__)


class SimpleTrainer(MettaTrainer):
    """Simplified trainer that creates agents directly without Hydra."""

    def __init__(self, config: TrainingConfig, env: MettaGridEnv, wandb_run: Optional[WandbContext] = None, **kwargs):
        self.config = config
        self.env = env

        # Create the agent directly
        agent = self._create_agent(env)

        # Convert our dataclass config to the dict format expected by MettaTrainer
        cfg_dict = self._config_to_dict(config)

        # Initialize the base trainer
        super().__init__(
            cfg=cfg_dict,
            wandb_run=wandb_run,
            policy_store=None,  # We'll manage policies directly
            sim_suite_config=None,  # Simplified - no simulation suite
            stats_client=None,
            **kwargs,
        )

        # Override the policy with our directly created agent
        self.policy = agent
        self.uncompiled_policy = agent
        self._initial_agent = agent
        self.last_agent = agent

    def _create_agent(self, env: MettaGridEnv) -> BaseAgent:
        """Create an agent using our new architecture."""
        agent_config = self.config.agent

        agent = create_agent(
            agent_name=agent_config.name,
            obs_space=env.observation_space,
            action_space=env.action_space,
            obs_width=env.obs_width,
            obs_height=env.obs_height,
            feature_normalizations=env.feature_normalizations,
            device=self.config.trainer.device,
            hidden_size=agent_config.hidden_size,
            lstm_layers=agent_config.lstm_layers,
            **agent_config.kwargs,
        )

        # Activate actions
        agent.activate_actions(env.action_names, env.max_action_args, torch.device(self.config.trainer.device))

        return agent

    def _config_to_dict(self, config: TrainingConfig) -> dict:
        """Convert our dataclass config to dict format expected by MettaTrainer."""
        return {
            "run": config.run_name,
            "run_dir": config.run_dir,
            "data_dir": config.data_dir,
            "device": config.trainer.device,
            "vectorization": config.vectorization,
            "trainer": {
                "total_timesteps": config.trainer.total_timesteps,
                "batch_size": config.trainer.batch_size,
                "minibatch_size": config.trainer.minibatch_size,
                "num_workers": config.trainer.num_workers,
                "checkpoint_dir": config.trainer.checkpoint_dir,
                "checkpoint_interval": config.trainer.checkpoint_interval,
                "wandb_checkpoint_interval": config.trainer.wandb_checkpoint_interval,
                "evaluate_interval": config.trainer.evaluate_interval,
                "replay_interval": config.trainer.replay_interval,
                "curriculum": config.trainer.curriculum,
                "env_overrides": config.trainer.env_overrides,
                "compile": config.trainer.compile,
                "compile_mode": config.trainer.compile_mode,
                "forward_pass_minibatch_target_size": config.trainer.forward_pass_minibatch_target_size,
                "bptt_horizon": config.trainer.bptt_horizon,
                "clip_param": config.trainer.clip_param,
                "value_coef": config.trainer.value_coef,
                "entropy_coef": config.trainer.entropy_coef,
                "max_grad_norm": config.trainer.max_grad_norm,
                "average_reward": config.trainer.average_reward,
                "average_reward_alpha": config.trainer.average_reward_alpha,
                "use_rnn": config.trainer.use_rnn,
                "cpu_offload": config.trainer.cpu_offload,
                "require_contiguous_env_ids": config.trainer.require_contiguous_env_ids,
                "async_factor": config.trainer.async_factor,
                "profiler_interval_epochs": config.trainer.profiler_interval_epochs,
                "optimizer": {
                    "type": config.trainer.optimizer.type,
                    "learning_rate": config.trainer.optimizer.learning_rate,
                    "beta1": config.trainer.optimizer.beta1,
                    "beta2": config.trainer.optimizer.beta2,
                    "eps": config.trainer.optimizer.eps,
                    "weight_decay": config.trainer.optimizer.weight_decay,
                },
                "initial_policy": {
                    "uri": config.trainer.initial_policy_uri,
                },
            },
            "agent": {
                "l2_norm_coeff": config.agent.l2_norm_coeff,
                "l2_init_coeff": config.agent.l2_init_coeff,
                "l2_init_weight_update_interval": config.agent.l2_init_weight_update_interval,
                "analyze_weights_interval": config.agent.analyze_weights_interval,
            },
            "wandb": {
                "enabled": config.wandb.enabled,
                "project": config.wandb.project,
                "entity": config.wandb.entity,
                "name": config.wandb.name,
                "tags": config.wandb.tags,
            },
        }


def train_agent(
    agent_name: str = "simple_cnn",
    total_timesteps: int = 10_000_000,
    device: str = "cuda",
    wandb_enabled: bool = True,
    **kwargs,
) -> BaseAgent:
    """Convenience function to train an agent with minimal configuration.

    Args:
        agent_name: Name of the agent to train
        total_timesteps: Total training timesteps
        device: Device to train on
        wandb_enabled: Whether to use Weights & Biases
        **kwargs: Additional arguments passed to agent constructor

    Returns:
        Trained agent
    """
    # Create configuration
    config = TrainingConfig(
        run_name=f"train_{agent_name}",
        agent=AgentConfig(name=agent_name, kwargs=kwargs),
        trainer=TrainerConfig(
            total_timesteps=total_timesteps,
            device=device,
        ),
        wandb=WandbConfig(enabled=wandb_enabled),
    )

    # Create environment
    # Note: This is simplified - in practice you'd want to configure the environment
    from mettagrid import MettaGridEnv

    env = MettaGridEnv()

    # Create trainer
    wandb_context = WandbContext(config.wandb, config) if wandb_enabled else None

    try:
        with wandb_context as wandb_run:
            trainer = SimpleTrainer(config, env, wandb_run)
            trainer.train()
            return trainer.policy
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
