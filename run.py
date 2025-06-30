"""Example of using Metta as a library without Hydra configuration."""

import torch

from metta.api import (
    # Configuration classes
    AgentModelConfig,
    CheckpointConfig,
    EnvConfig,
    ExperienceConfig,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    # Functions
    compute_advantages,
    eval_policy,
    make_agent,
    make_curriculum,
    make_environment,
    make_experience_manager,
    make_optimizer,
    rollout,
    save_checkpoint,
    train_ppo,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environment with typed config
env_config = EnvConfig(
    game={
        "max_steps": 1000,
        "num_agents": 4,
        "width": 32,
        "height": 32,
    }
)
env = make_environment(env_config)

# Create agent with typed config
agent_config = AgentModelConfig(
    hidden_dim=512,
    lstm_layers=1,
    bptt_horizon=8,
)
agent = make_agent(
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    global_features=[],
    device=device,
    config=agent_config,
)

# Create optimizer with typed config
optimizer_config = OptimizerConfig(
    type="adam",
    learning_rate=3e-4,
)
optimizer = make_optimizer(agent, config=optimizer_config)

# Create curriculum
curriculum = make_curriculum("/env/mettagrid/simple")

# Create experience manager with typed config
experience_config = ExperienceConfig(
    batch_size=8192,
    minibatch_size=512,
    bptt_horizon=8,
)
experience = make_experience_manager(env, agent, config=experience_config)

# Training configuration
ppo_config = PPOConfig(
    clip_coef=0.1,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
)

checkpoint_config = CheckpointConfig(
    checkpoint_interval=100,
    checkpoint_dir="./checkpoints",
)

simulation_config = SimulationConfig(
    evaluate_interval=500,
)

# Training loop
total_steps = 0
epoch = 0
target_steps = 100_000

print(f"Starting training for {target_steps} steps...")

while total_steps < target_steps:
    # Collect experience
    print(f"Epoch {epoch}: Collecting rollouts...")
    batch_info = rollout(experience, agent)
    total_steps += batch_info.total_env_steps

    # Compute advantages
    advantages = compute_advantages(experience, gamma=ppo_config.gamma, gae_lambda=ppo_config.gae_lambda)

    # Train
    print(f"Epoch {epoch}: Training PPO...")
    train_stats = train_ppo(
        agent,
        optimizer,
        experience,
        ppo_config=ppo_config,
        update_epochs=4,
    )

    # Print stats
    print(f"Epoch {epoch}: Steps={total_steps}, Stats={train_stats}")

    # Save checkpoint
    if epoch % checkpoint_config.checkpoint_interval == 0:
        save_checkpoint(agent, checkpoint_config.checkpoint_dir, epoch)
        print(f"Saved checkpoint at epoch {epoch}")

    # Evaluate
    if epoch % simulation_config.evaluate_interval == 0 and epoch > 0:
        print("Evaluating policy...")
        eval_stats = eval_policy(agent, env, num_episodes=5)
        print(f"Evaluation stats: {eval_stats}")

    epoch += 1

print("Training complete!")

# Final evaluation
print("Running final evaluation...")
final_stats = eval_policy(agent, env, num_episodes=10)
print(f"Final evaluation results: {final_stats}")
