"""Example of functional training with custom losses and modifications."""

import torch

from metta.agent import create_agent
from metta.rl.configs import PPOConfig, TrainerConfig
from metta.rl.functional_trainer import (
    functional_training_loop,
)
from mettagrid import create_env
from mettagrid.curriculum import SimpleCurriculum


# Custom loss function example
def curiosity_driven_loss(policy, obs, actions, rewards, values, advantages):
    """Intrinsic curiosity reward based on state prediction error."""
    # This is a simplified example - real implementation would:
    # 1. Have a forward model that predicts next state
    # 2. Compute prediction error as intrinsic reward
    # 3. Add intrinsic reward to the loss

    # For now, just return a small regularization
    return 0.01 * torch.mean(torch.abs(values))


def diversity_loss(policy, obs, actions, rewards, values, advantages):
    """Encourage diverse actions through entropy bonus."""
    # Get action logits
    lstm_state = None  # Simplified
    logits, _, _, _, _ = policy(obs, lstm_state)

    # Compute entropy of action distribution
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

    # Negative loss to encourage high entropy
    return -0.1 * entropy.mean()


def custom_training_step(state, components, config, experience, custom_losses=None):
    """Custom training step with additional logic."""
    metrics = {}

    # 1. Collect rollouts
    rollout_stats, steps = components["collector"].collect()
    state.agent_steps = components["collector"].agent_steps
    components["stats_logger"].add_stats(rollout_stats)

    # 2. Dynamic learning rate adjustment
    if state.epoch > 0 and state.epoch % 100 == 0:
        # Decay learning rate
        for param_group in components["optimizer"].param_groups:
            param_group["lr"] *= 0.9
        print(f"Adjusted learning rate to {param_group['lr']}")

    # 3. PPO update with custom losses
    loss_stats = components["ppo"].update(
        experience=experience,
        update_epochs=4,
        custom_loss_fns=custom_losses,
    )
    metrics.update(loss_stats)

    # 4. Custom evaluation logic
    if state.epoch % 50 == 0:
        # Could add custom evaluation here
        print(f"Epoch {state.epoch}: Policy loss = {loss_stats.get('policy_loss', 0):.4f}")

    # 5. Early stopping based on performance
    recent_rewards = rollout_stats.get("episode/reward", [])
    if recent_rewards and sum(recent_rewards) / len(recent_rewards) > 0.95:
        print("Reached target performance, stopping training")
        state.should_stop = True

    state.epoch += 1
    return metrics


def main():
    """Example of functional training with customizations."""

    # 1. Create configs
    trainer_config = TrainerConfig(
        total_timesteps=100_000,
        batch_size=2048,
        minibatch_size=256,
        checkpoint_interval=100,
        evaluate_interval=0,  # Disabled for simplicity
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ppo_config = PPOConfig(
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=4,
    )

    # 2. Create environment and agent
    env = create_env(width=15, height=15, max_steps=500)

    agent = create_agent(
        agent_name="simple_cnn",
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        device=trainer_config.device,
    )

    # 3. Create curriculum
    curriculum = SimpleCurriculum(base_env_config={"width": 15, "height": 15})

    # 4. Create policy store (simplified)
    from metta.agent.policy_store import MemoryPolicyStore

    policy_store = MemoryPolicyStore()

    # 5. Run training with custom losses and step function
    custom_losses = [curiosity_driven_loss, diversity_loss]

    final_state = functional_training_loop(
        config=trainer_config,
        ppo_config=ppo_config,
        policy=agent,
        curriculum=curriculum,
        policy_store=policy_store,
        step_fn=custom_training_step,
        custom_losses=custom_losses,
    )

    print(f"Training completed after {final_state.epoch} epochs")
    print(f"Total steps: {final_state.agent_steps}")


# Example of fully custom training loop
def fully_custom_training():
    """Example of completely custom training loop using components."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = create_env(width=15, height=15)

    # Create agent
    agent = create_agent("simple_cnn", env.observation_space, env.action_space, device=device)

    # Create optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    # Custom training loop
    for epoch in range(1000):
        # Collect experience (simplified)
        obs = torch.randn(32, *env.observation_space.shape).to(device)

        # Forward pass
        with torch.no_grad():
            actions, logprobs, _, values, _ = agent(obs, None)

        # Simulate rewards
        rewards = torch.randn(32).to(device)

        # Custom loss computation
        policy_loss = -torch.mean(logprobs * rewards)
        value_loss = torch.mean((values - rewards) ** 2)

        # Add any custom losses
        curiosity_bonus = 0.01 * torch.mean(torch.abs(values))

        total_loss = policy_loss + 0.5 * value_loss + curiosity_bonus

        # Optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")


if __name__ == "__main__":
    # Run functional training example
    main()

    # Or run fully custom training
    # fully_custom_training()
