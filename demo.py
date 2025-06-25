#!/usr/bin/env python3
"""
Functional Training Demo for Metta

This demo showcases the new functional training approach in Metta, which provides
a simpler, more explicit alternative to the existing Hydra-based training system.

=== What is Functional Training? ===

Instead of hiding the training loop inside a Trainer class with complex
configuration magic, functional training:

1. Creates all objects explicitly (no Hydra instantiation)
2. Uses simple Pydantic models for configuration
3. Exposes the training loop as a simple while loop
4. Calls two main functions: rollout() and train_ppo()
5. Gives you full control over every aspect of training

=== Key Benefits ===

- Transparency: See exactly what's happening in your training loop
- Flexibility: Easy to add custom losses, logging, or modify any part
- Simplicity: No framework magic, just regular Python code
- Performance: Same optimized C++/CUDA kernels as the original

=== How It Works ===

The core pattern is simple:

    while agent_step < total_timesteps:
        # Collect experience from environment
        agent_step, stats = rollout(policy, vecenv, experience, device, agent_step)

        # Update policy using PPO
        epoch = train_ppo(policy, optimizer, experience, device, losses, epoch, ...)

        # Your custom logic (logging, checkpointing, etc.)

This demo shows both:
1. A conceptual example (to understand the pattern)
2. A complete working example (with real environments)

Let's start!
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import yaml
from pydantic import BaseModel

# Add metta to path if running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import DictConfig

# =============================================================================
# PART 1: Pydantic Configuration Models
# =============================================================================

# Instead of complex YAML hierarchies, we use simple Pydantic models
# These provide type safety, validation, and easy programmatic modification


class PPOConfig(BaseModel):
    """PPO algorithm parameters"""

    gamma: float = 0.977  # Discount factor
    gae_lambda: float = 0.916  # GAE lambda
    clip_coef: float = 0.1  # PPO clipping coefficient
    ent_coef: float = 0.0021  # Entropy coefficient
    vf_coef: float = 0.44  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    norm_adv: bool = True  # Normalize advantages
    clip_vloss: bool = True  # Clip value loss
    vf_clip_coef: float = 0.1  # Value function clipping
    target_kl: Optional[float] = None  # Early stopping based on KL
    vtrace_rho_clip: float = 1.0  # V-trace clipping
    vtrace_c_clip: float = 1.0  # V-trace clipping


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""

    type: str = "adam"  # "adam" or "muon"
    learning_rate: float = 0.0004573146765703167
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-12
    weight_decay: float = 0.0


class TrainingConfig(BaseModel):
    """Training loop configuration"""

    total_timesteps: int = 1_000_000
    batch_size: int = 6144
    minibatch_size: int = 256
    bptt_horizon: int = 64  # Backprop through time horizon
    update_epochs: int = 1
    checkpoint_interval: int = 10
    evaluate_interval: int = 100
    cpu_offload: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    zero_copy: bool = True
    l2_reg_loss_coef: float = 0.0
    l2_init_loss_coef: float = 0.0


class EnvironmentConfig(BaseModel):
    """Environment configuration"""

    curriculum: str = "env/mettagrid/curriculum/simple"
    num_workers: int = 1
    async_factor: int = 2
    forward_pass_minibatch_target_size: int = 32
    seed: Optional[int] = 42


class PrioritizedReplayConfig(BaseModel):
    """Prioritized experience replay configuration"""

    prio_alpha: float = 0.0
    prio_beta0: float = 0.6


def load_yaml_config(path: str, config_class):
    """Load YAML config into Pydantic model"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return config_class(**data)


# =============================================================================
# PART 2: Simple Conceptual Example
# =============================================================================


def conceptual_example():
    """
    This example shows the STRUCTURE of functional training without
    the complexity of real environment setup. Use this to understand
    the core pattern.
    """

    print("\n" + "=" * 80)
    print("PART 1: Conceptual Example (Understanding the Pattern)")
    print("=" * 80 + "\n")

    # 1. Create configurations using Pydantic
    ppo_config = PPOConfig()
    optimizer_config = OptimizerConfig()
    training_config = TrainingConfig(
        total_timesteps=1000,  # Small for demo
        checkpoint_interval=100,
    )

    print("1. Created Pydantic configurations:")
    print(f"   - PPO: gamma={ppo_config.gamma}, clip_coef={ppo_config.clip_coef}")
    print(f"   - Optimizer: {optimizer_config.type}, lr={optimizer_config.learning_rate}")
    print(f"   - Training: {training_config.total_timesteps} steps\n")

    # 2. Create objects directly (no Hydra magic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock policy for demonstration
    class MockPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)

        def forward(self, obs, state=None, action=None):
            # Mock forward pass
            batch_size = obs.shape[0]
            actions = torch.randint(0, 10, (batch_size, 2))
            logprobs = torch.randn(batch_size)
            entropy = torch.randn(batch_size)
            values = torch.randn(batch_size)
            full_logprobs = torch.randn(batch_size, 10)
            return actions, logprobs, entropy, values, full_logprobs

    policy = MockPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=optimizer_config.learning_rate)

    print("2. Created objects directly:")
    print(f"   - Policy: {policy.__class__.__name__}")
    print(f"   - Optimizer: {optimizer.__class__.__name__}")
    print(f"   - Device: {device}\n")

    # 3. Main training loop - the key pattern!
    agent_step = 0
    epoch = 0

    print("3. Starting functional training loop...\n")

    while agent_step < training_config.total_timesteps:
        # ========== ROLLOUT PHASE ==========
        print(f"Epoch {epoch}: Rollout phase")

        # In real code:
        # agent_step, stats = rollout(
        #     policy, vecenv, experience, device, agent_step, timer
        # )

        # Mock rollout
        rollout_steps = 128
        agent_step += rollout_steps

        # ========== TRAINING PHASE ==========
        print(f"Epoch {epoch}: Training phase")

        # In real code:
        # epoch = train_ppo(
        #     policy=policy,
        #     optimizer=optimizer,
        #     experience=experience,
        #     device=device,
        #     losses=losses,
        #     epoch=epoch,
        #     cfg=cfg,
        #     **ppo_config.dict(),
        # )

        # Mock training
        for _ in range(training_config.update_epochs):
            loss = torch.randn(1, requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_config.max_grad_norm)
            optimizer.step()

        epoch += 1

        # ========== YOUR CUSTOM LOGIC ==========
        print(f"   - Steps: {agent_step}/{training_config.total_timesteps}")
        print(f"   - Mock SPS: {np.random.randint(1000, 2000)}\n")

        if epoch % training_config.checkpoint_interval == 0:
            print(f"💾 [Your code] Save checkpoint at epoch {epoch}\n")

    print("✅ Conceptual example complete!\n")
    print("Key insights:")
    print("- The training loop is just a while loop")
    print("- You call rollout() to collect experience")
    print("- You call train_ppo() to update the policy")
    print("- Everything else is up to you!")


# =============================================================================
# PART 3: Complete Working Example
# =============================================================================


def complete_example():
    """
    This is a more complete example that shows how to set up real
    environments and run actual training. Due to environment setup
    complexity, this uses Hydra temporarily just for loading the
    curriculum configuration.
    """

    print("\n" + "=" * 80)
    print("PART 2: Complete Example (With Real Environments)")
    print("=" * 80 + "\n")

    try:
        # Import required modules
        from metta.agent.policy_store import PolicyStore
        from metta.common.stopwatch import Stopwatch
        from metta.mettagrid.mettagrid_env import MettaGridEnv
        from metta.rl.experience import Experience
        from metta.rl.functional_trainer import rollout, train_ppo
        from metta.rl.losses import Losses
        from metta.rl.trainer_checkpoint import TrainerCheckpoint
        from metta.rl.vecenv import make_vecenv

        # Pufferlib C extension for advantage computation
        try:
            from pufferlib import _C  # noqa: F401
        except ImportError:
            raise ImportError("Failed to import C/CUDA advantage kernel. Try installing with --no-build-isolation")

        # 1. Create configurations
        ppo_config = PPOConfig()
        optimizer_config = OptimizerConfig()
        training_config = TrainingConfig(
            total_timesteps=10_000,  # Short demo
            checkpoint_interval=5,
        )
        env_config = EnvironmentConfig()
        replay_config = PrioritizedReplayConfig()

        # 2. Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")

        run_dir = "./demo_run"
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 3. Create curriculum (uses Hydra temporarily for config loading)
        from hydra import initialize_config_dir

        from metta.util.resolvers import register_resolvers

        register_resolvers()
        config_path = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_path, version_base=None):
            from metta.mettagrid.curriculum.util import curriculum_from_config_path

            curriculum = curriculum_from_config_path(env_config.curriculum, DictConfig({}))

        # 4. Create vectorized environment
        num_agents = curriculum.get_task().env_cfg().game.num_agents
        target_batch_size = env_config.forward_pass_minibatch_target_size // num_agents
        target_batch_size = max(2, target_batch_size)

        env_batch_size = (target_batch_size // env_config.num_workers) * env_config.num_workers
        env_batch_size = max(env_config.num_workers, env_batch_size)
        num_envs = env_batch_size * env_config.async_factor

        if env_config.num_workers == 1:
            env_batch_size = num_envs

        print(f"Creating {num_envs} environments...")
        vecenv = make_vecenv(
            curriculum,
            "serial",
            num_envs=num_envs,
            batch_size=env_batch_size,
            num_workers=env_config.num_workers,
            zero_copy=training_config.zero_copy,
        )

        seed = env_config.seed or np.random.randint(0, 1000000)
        vecenv.async_reset(seed)

        metta_grid_env: MettaGridEnv = vecenv.driver_env

        # 5. Create or load policy
        print("Creating policy...")
        simple_cfg = DictConfig({"device": str(device), "data_dir": run_dir})
        policy_store = PolicyStore(simple_cfg, wandb_run=None)

        checkpoint = TrainerCheckpoint.load(run_dir)
        if checkpoint.policy_path:
            policy_record = policy_store.load_from_uri(checkpoint.policy_path)
            print("Loaded existing policy from checkpoint")
        else:
            policy_record = policy_store.create(metta_grid_env)
            print("Created new policy")

        policy = policy_record.policy().to(device)

        # Activate actions
        policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, device)

        uncompiled_policy = policy
        if training_config.compile:
            print("Compiling policy...")
            policy = torch.compile(policy, mode=training_config.compile_mode)

        # 6. Create experience buffer
        experience = Experience(
            total_agents=vecenv.num_agents,
            batch_size=training_config.batch_size,
            bptt_horizon=training_config.bptt_horizon,
            minibatch_size=training_config.minibatch_size,
            max_minibatch_size=training_config.minibatch_size,
            obs_space=vecenv.single_observation_space,
            atn_space=vecenv.single_action_space,
            device=device,
            hidden_size=getattr(policy, "hidden_size", 256),
            cpu_offload=training_config.cpu_offload,
            num_lstm_layers=2,
            agents_per_batch=getattr(vecenv, "agents_per_batch", None),
        )

        # 7. Create optimizer
        if optimizer_config.type == "adam":
            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=optimizer_config.learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
            )
        elif optimizer_config.type == "muon":
            from heavyball import ForeachMuon

            optimizer = ForeachMuon(
                policy.parameters(),
                lr=optimizer_config.learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
            )

        # 8. Initialize training state
        agent_step = checkpoint.agent_step
        epoch = checkpoint.epoch
        losses = Losses()
        timer = Stopwatch(None)
        timer.start()

        agent_cfg = DictConfig({"agent": {"clip_range": 0}})

        print(f"\nStarting training from epoch {epoch}, step {agent_step}")
        print("=" * 50)

        # 9. Main training loop - this is the key!
        while agent_step < training_config.total_timesteps:
            steps_before = agent_step

            # Collect experience from environment
            with timer("rollout"):
                agent_step, stats = rollout(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device,
                    agent_step=agent_step,
                    timer=timer,
                )

            # Update policy using PPO
            with timer("train"):
                epoch = train_ppo(
                    policy=policy,
                    optimizer=optimizer,
                    experience=experience,
                    device=device,
                    losses=losses,
                    epoch=epoch,
                    cfg=agent_cfg,
                    lr_scheduler=None,
                    timer=timer,
                    # PPO parameters
                    gamma=ppo_config.gamma,
                    gae_lambda=ppo_config.gae_lambda,
                    clip_coef=ppo_config.clip_coef,
                    ent_coef=ppo_config.ent_coef,
                    vf_coef=ppo_config.vf_coef,
                    max_grad_norm=ppo_config.max_grad_norm,
                    norm_adv=ppo_config.norm_adv,
                    clip_vloss=ppo_config.clip_vloss,
                    vf_clip_coef=ppo_config.vf_clip_coef,
                    update_epochs=training_config.update_epochs,
                    target_kl=ppo_config.target_kl,
                    kickstarter=None,
                    agent_step=agent_step,
                    l2_reg_loss_coef=training_config.l2_reg_loss_coef,
                    l2_init_loss_coef=training_config.l2_init_loss_coef,
                    clip_range=0,
                    # Prioritized replay
                    prio_alpha=replay_config.prio_alpha,
                    prio_beta0=replay_config.prio_beta0,
                    total_timesteps=training_config.total_timesteps,
                    batch_size=training_config.batch_size,
                    # V-trace
                    vtrace_rho_clip=ppo_config.vtrace_rho_clip,
                    vtrace_c_clip=ppo_config.vtrace_c_clip,
                )

            # Calculate metrics
            rollout_time = timer.get_last_elapsed("rollout")
            train_time = timer.get_last_elapsed("train")
            total_time = train_time + rollout_time
            steps_calculated = agent_step - steps_before
            steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

            loss_stats = losses.stats()

            print(
                f"Epoch {epoch:4d} | "
                f"Steps: {agent_step:6d}/{training_config.total_timesteps} | "
                f"SPS: {steps_per_sec:5.0f} | "
                f"Loss: {loss_stats.get('policy_loss', 0):.4f}"
            )

            # Checkpoint
            if epoch % training_config.checkpoint_interval == 0:
                print(f"Saving checkpoint at epoch {epoch}...")
                name = policy_store.make_model_name(epoch)
                path = os.path.join(checkpoint_dir, name)
                pr = policy_store.save(
                    name=name, path=path, policy=uncompiled_policy, metadata={"epoch": epoch, "agent_step": agent_step}
                )

                checkpoint = TrainerCheckpoint(
                    agent_step=agent_step,
                    epoch=epoch,
                    total_agent_step=agent_step,
                    optimizer_state_dict=optimizer.state_dict(),
                    policy_path=pr.uri if pr else None,
                    extra_args={},
                )
                checkpoint.save(run_dir)

        # Final checkpoint
        print("\nTraining complete! Saving final checkpoint...")
        name = policy_store.make_model_name(epoch)
        path = os.path.join(checkpoint_dir, name)
        pr = policy_store.save(
            name=name,
            path=path,
            policy=uncompiled_policy,
            metadata={"epoch": epoch, "agent_step": agent_step, "final": True},
        )

        vecenv.close()
        print(f"✅ Complete example done! Model saved to: {path}")

    except Exception as e:
        print(f"\n⚠️  Complete example failed: {e}")
        print("This is likely due to environment setup complexity.")
        print("The conceptual example above shows the key pattern!")


# =============================================================================
# PART 4: Custom Training Examples
# =============================================================================


def custom_training_patterns():
    """Shows various custom training patterns you can implement"""

    print("\n" + "=" * 80)
    print("PART 3: Custom Training Patterns")
    print("=" * 80 + "\n")

    print("With functional training, you have full control to implement:\n")

    print("1. Custom Losses:")
    print("""
    def curiosity_loss(policy, obs, actions):
        # Your custom loss computation
        return loss_tensor

    # In train_ppo, add your loss:
    total_loss = ppo_loss + curiosity_loss(policy, obs, actions)
    """)

    print("2. Custom Advantage Functions:")
    print("""
    def my_advantage_function(rewards, values, gamma):
        # Your custom advantage computation
        return advantages

    # Use instead of default advantage computation
    advantages = my_advantage_function(rewards, values, gamma)
    """)

    print("3. Custom Logging:")
    print("""
    # Log to TensorBoard
    writer.add_scalar('custom/metric', value, step)

    # Log to your own system
    my_logger.log({
        'step': agent_step,
        'custom_metric': compute_custom_metric(policy)
    })
    """)

    print("4. Dynamic Hyperparameter Scheduling:")
    print("""
    # Adjust learning rate
    if agent_step > warmup_steps:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.99

    # Adjust PPO clip range
    ppo_config.clip_coef = 0.2 * (1 - agent_step / total_steps)
    """)

    print("5. Custom Evaluation:")
    print("""
    if epoch % eval_interval == 0:
        eval_reward = evaluate_on_test_env(policy)
        if eval_reward > best_reward:
            save_best_model(policy)
    """)


# =============================================================================
# PART 5: Migration Guide
# =============================================================================


def migration_guide():
    """Guide for migrating from the old Hydra-based approach"""

    print("\n" + "=" * 80)
    print("PART 4: Migrating from Hydra-based Training")
    print("=" * 80 + "\n")

    print("If you're currently using the old approach:\n")

    print("OLD WAY (Hydra + MettaTrainer):")
    print("""
    # Configuration hidden in YAML files
    @hydra.main(config_path="configs", config_name="train")
    def main(cfg):
        trainer = MettaTrainer(cfg, wandb_run, policy_store)
        trainer.train()  # Training loop hidden inside
    """)

    print("\nNEW WAY (Functional):")
    print("""
    # Configuration in code
    ppo_config = PPOConfig(gamma=0.99, clip_coef=0.2)
    training_config = TrainingConfig(total_timesteps=1_000_000)

    # Training loop visible
    while agent_step < training_config.total_timesteps:
        agent_step, stats = rollout(...)
        epoch = train_ppo(...)
        # Your custom logic here
    """)

    print("\nMigration Steps:")
    print("1. Keep using tools/train.py for production")
    print("2. Use functional approach for research/experimentation")
    print("3. Gradually move custom logic to functional style")
    print("4. Eventually replace Hydra configs with Pydantic models")


# =============================================================================
# PART 6: Tips and Best Practices
# =============================================================================


def tips_and_practices():
    """Best practices for functional training"""

    print("\n" + "=" * 80)
    print("PART 5: Tips and Best Practices")
    print("=" * 80 + "\n")

    print("✅ DO:")
    print("- Start with the conceptual example to understand the pattern")
    print("- Use Pydantic configs for type safety and validation")
    print("- Keep your training loop simple and readable")
    print("- Add logging/metrics incrementally")
    print("- Test with small experiments before scaling up")

    print("\n❌ DON'T:")
    print("- Try to replicate all Hydra features at once")
    print("- Make the training loop too complex")
    print("- Forget to save checkpoints regularly")
    print("- Skip validation of your custom losses")

    print("\n🚀 Performance Tips:")
    print("- Use compile=True for 10-30% speedup")
    print("- Enable zero_copy=True for efficiency")
    print("- Use appropriate batch sizes for your GPU")
    print("- Profile with torch.profiler if needed")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("""
    🚀 Metta Functional Training Demo
    ================================

    This demo shows how to use Metta's new functional training approach,
    which gives you full control over the training process without the
    complexity of Hydra configuration management.
    """)

    # Run all demo sections
    conceptual_example()  # Simple pattern demonstration
    complete_example()  # Real environment example
    custom_training_patterns()  # Custom training ideas
    migration_guide()  # How to migrate from old approach
    tips_and_practices()  # Best practices

    print("\n" + "=" * 80)
    print("🎯 Next Steps")
    print("=" * 80)
    print("""
    1. Explore metta/rl/functional_trainer.py to see the implementation
    2. Modify this demo for your own experiments
    3. Check tools/train.py to see the original approach
    4. Join the community to share your custom training loops!

    Remember: The power of functional training is that YOU control everything.
    No magic, no hidden complexity, just clean Python code.

    Happy training! 🎮
    """)
