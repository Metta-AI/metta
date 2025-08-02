"""Integration layer for Muesli with existing trainer infrastructure."""

import logging
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig

from metta.rl.muesli.agent import MuesliAgent
from metta.rl.muesli.config import MuesliConfig
from metta.rl.muesli.replay_buffer import MuesliReplayBuffer
from metta.rl.muesli.trainer import muesli_train
from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class MuesliPolicyWrapper(nn.Module):
    """Wrapper to make MuesliAgent compatible with existing policy interface."""
    
    def __init__(
        self,
        muesli_agent: MuesliAgent,
        action_names: list[str],
        max_action_args: list[int],
    ):
        super().__init__()
        self.muesli_agent = muesli_agent
        self.action_names = action_names
        self.max_action_args = max_action_args
        
    def forward(
        self, 
        obs: torch.Tensor,
        lstm_state: Any = None,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass compatible with existing interface.
        
        Returns:
            action, log_prob, entropy, value, full_logprobs
        """
        # Get Muesli output
        output = self.muesli_agent(obs, lstm_state, action)
        
        # Extract required values
        if action is None:
            action = output['action']
            
        log_prob = output['log_prob']
        entropy = output['entropy']
        value = output['value']
        policy_logits = output['policy_logits']
        
        # Normalize logits to get log probabilities
        full_logprobs = torch.log_softmax(policy_logits, dim=-1)
        
        return action, log_prob, entropy, value, full_logprobs
        
    def get_value(self, obs: torch.Tensor, lstm_state: Any = None) -> torch.Tensor:
        """Get value prediction."""
        output = self.muesli_agent(obs, lstm_state)
        return output['value']
        
    def get_action_and_value(
        self, 
        obs: torch.Tensor,
        lstm_state: Any = None,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value (simplified interface)."""
        output = self.muesli_agent(obs, lstm_state, action)
        return output['action'], output['log_prob'], output['value']


def create_muesli_agent(
    env_cfg: Any,
    agent_cfg: DictConfig,
    device: torch.device,
    trainer_cfg: TrainerConfig
) -> Tuple[MuesliPolicyWrapper, MuesliConfig]:
    """Create a Muesli agent from configuration.
    
    Args:
        env_cfg: Environment configuration
        agent_cfg: Agent configuration (unused for Muesli)
        device: Device to use
        trainer_cfg: Trainer configuration containing Muesli config
        
    Returns:
        Wrapped Muesli agent and configuration
    """
    # Extract Muesli config from trainer config
    if not hasattr(trainer_cfg, 'muesli'):
        raise ValueError("Trainer config must have 'muesli' section for Muesli algorithm")
        
    muesli_cfg = MuesliConfig(**trainer_cfg.muesli)
    
    # Get observation and action spaces from environment
    # This is simplified - in practice you'd need proper env access
    obs_shape = (env_cfg.game.observation_size,)  # Adjust based on actual env
    
    # Create dummy action space for now
    # In practice, you'd get this from the environment
    class DummyActionSpace:
        def __init__(self, n):
            self.n = n
            
    action_space = DummyActionSpace(env_cfg.game.num_actions if hasattr(env_cfg.game, 'num_actions') else 10)
    
    # Create Muesli agent
    muesli_agent = MuesliAgent(
        obs_shape=obs_shape,
        action_space=action_space,
        config=muesli_cfg,
        device=device
    )
    
    # Get action names from env
    action_names = getattr(env_cfg.game, 'action_names', [f"action_{i}" for i in range(action_space.n)])
    max_action_args = getattr(env_cfg.game, 'max_action_args', [0] * action_space.n)
    
    # Wrap in compatibility layer
    wrapped_agent = MuesliPolicyWrapper(
        muesli_agent,
        action_names,
        max_action_args
    )
    
    return wrapped_agent, muesli_cfg


def create_muesli_trainer(
    policy: MuesliPolicyWrapper,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    trainer_cfg: TrainerConfig,
    muesli_cfg: MuesliConfig
) -> Dict[str, Any]:
    """Create Muesli training components.
    
    Returns:
        Dictionary with training components
    """
    # Create replay buffer
    replay_buffer = MuesliReplayBuffer(
        capacity=muesli_cfg.replay.capacity,
        unroll_length=muesli_cfg.model_learning.unroll_steps,
        gamma=trainer_cfg.ppo.gamma,  # Use gamma from trainer config
        device=device
    )
    
    # Training function that integrates with existing loop
    def train_fn(
        experience: Experience,
        losses: Losses,
        agent_step: int,
        epoch: int,
        envs: Any,
        wandb_run: Optional[Any] = None
    ) -> Dict[str, float]:
        """Muesli training function compatible with existing trainer."""
        # Get number of training steps based on experience size
        num_minibatches = experience.num_minibatches * trainer_cfg.update_epochs
        
        # Run Muesli training
        metrics = muesli_train(
            agent=policy.muesli_agent,
            envs=envs,
            config=muesli_cfg,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            experience=experience,
            losses=losses,
            training_steps=num_minibatches,
            agent_step=agent_step,
            epoch=epoch,
            device=device,
            wandb_run=wandb_run
        )
        
        return metrics
        
    return {
        'replay_buffer': replay_buffer,
        'train_fn': train_fn,
        'muesli_cfg': muesli_cfg
    }


# Monkey patch to integrate with existing trainer
def patch_trainer_for_muesli():
    """Patch the existing trainer to support Muesli algorithm."""
    import metta.rl.trainer as trainer_module
    
    # Store original train function
    original_train = trainer_module.train
    
    def muesli_aware_train(
        run_dir: str,
        run: str,
        env_cfg: Any,
        agent_cfg: DictConfig,
        device: torch.device,
        trainer_cfg: TrainerConfig,
        wandb_run: Any,
        policy_store: Any,
        sim_suite_config: Any,
        stats_client: Any,
    ) -> None:
        """Modified train function that supports Muesli."""
        
        # Check if using Muesli algorithm
        if hasattr(trainer_cfg, 'algorithm') and trainer_cfg.algorithm == 'muesli':
            logger.info("Using Muesli algorithm")
            
            # Create Muesli components
            from metta.rl.muesli.integration import create_muesli_agent, create_muesli_trainer
            
            # Import necessary modules
            import torch.distributed as dist
            from metta.rl.kickstarter import Kickstarter
            from metta.rl.checkpoint_manager import CheckpointManager
            from metta.rl.hyperparameter_scheduler import HyperparameterScheduler
            from metta.utils.timer import Timer
            
            # Initialize distributed if needed
            world_size = 1
            global_rank = 0
            if dist.is_initialized():
                world_size = dist.get_world_size()
                global_rank = dist.get_rank()
                
            # Create environments
            from metta.env.vec_env import VecEnv
            from metta.rl.curriculum import create_curriculum
            
            curriculum = create_curriculum(trainer_cfg.curriculum, env_cfg)
            
            # Create vectorized environment
            num_envs = trainer_cfg.num_workers or 64
            vecenv = VecEnv(
                env_cfg=env_cfg,
                num_envs=num_envs,
                device=device,
                global_rank=global_rank,
                world_size=world_size,
                cpu_offload=trainer_cfg.cpu_offload,
                bptt_horizon=trainer_cfg.bptt_horizon,
                zero_copy=trainer_cfg.zero_copy,
                require_contiguous_env_ids=trainer_cfg.require_contiguous_env_ids,
            )
            
            # Create Muesli agent
            policy, muesli_cfg = create_muesli_agent(
                env_cfg, agent_cfg, device, trainer_cfg
            )
            
            # Create optimizer
            from metta.rl.optimizer import create_optimizer
            optimizer, scheduler = create_optimizer(
                policy.parameters(),
                trainer_cfg.optimizer,
                trainer_cfg.hyperparameter_scheduler,
                trainer_cfg.total_timesteps,
                trainer_cfg.batch_size,
            )
            
            # Create training components
            muesli_components = create_muesli_trainer(
                policy, optimizer, device, trainer_cfg, muesli_cfg
            )
            
            # Create experience buffer
            experience = Experience(
                total_agents=vecenv.num_agents,
                batch_size=trainer_cfg.batch_size,
                bptt_horizon=trainer_cfg.bptt_horizon,
                minibatch_size=trainer_cfg.minibatch_size,
                max_minibatch_size=trainer_cfg.forward_pass_minibatch_target_size,
                obs_space=vecenv.single_observation_space,
                atn_space=vecenv.single_action_space,
                device=device,
                hidden_size=policy.muesli_agent.hidden_size,
                cpu_offload=trainer_cfg.cpu_offload,
                num_lstm_layers=muesli_cfg.network.num_lstm_layers,
            )
            
            # Initialize other components
            timer = Timer()
            losses = Losses()
            checkpoint_manager = CheckpointManager(
                trainer_cfg.checkpoint,
                wandb_run,
                policy_store,
                stats_client,
            )
            
            # Dummy kickstarter for compatibility
            kickstarter = type('DummyKickstarter', (), {
                'enabled': False,
                'loss': lambda *args, **kwargs: (torch.tensor(0.0), torch.tensor(0.0))
            })()
            
            # Training loop
            agent_step = 0
            epoch = 0
            
            while agent_step < trainer_cfg.total_timesteps:
                # Run Muesli training
                metrics = muesli_components['train_fn'](
                    experience=experience,
                    losses=losses,
                    agent_step=agent_step,
                    epoch=epoch,
                    envs=vecenv,
                    wandb_run=wandb_run
                )
                
                # Update counters
                agent_step += trainer_cfg.batch_size
                epoch += 1
                
                # Checkpointing
                if epoch % trainer_cfg.checkpoint.checkpoint_interval == 0:
                    checkpoint_manager.save_checkpoint(
                        agent_step=agent_step,
                        epoch=epoch,
                        optimizer=optimizer,
                        policy_path=f"{run_dir}/policy_{epoch}.pt",
                        timer=timer,
                        run_dir=run_dir,
                        kickstarter=kickstarter,
                        force=False,
                    )
                    
                # Logging
                if wandb_run and epoch % 10 == 0:
                    wandb_run.log({
                        'epoch': epoch,
                        'agent_step': agent_step,
                        **metrics
                    })
                    
            logger.info("Muesli training completed")
            
        else:
            # Use original training function for non-Muesli algorithms
            original_train(
                run_dir, run, env_cfg, agent_cfg, device, 
                trainer_cfg, wandb_run, policy_store, 
                sim_suite_config, stats_client
            )
            
    # Replace the train function
    trainer_module.train = muesli_aware_train
    
    logger.info("Trainer patched for Muesli support")