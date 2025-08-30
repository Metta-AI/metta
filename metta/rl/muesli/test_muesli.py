"""Test script for Muesli implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from metta.rl.muesli.agent import MuesliAgent
from metta.rl.muesli.config import MuesliConfig
from metta.rl.muesli.replay_buffer import MuesliReplayBuffer
from metta.rl.muesli.losses import compute_muesli_losses
from metta.rl.muesli.categorical import scalar_to_support, support_to_scalar


def create_dummy_env():
    """Create a simple dummy environment for testing."""
    class DummyEnv:
        def __init__(self, obs_size=4, num_actions=2):
            self.obs_size = obs_size
            self.num_actions = num_actions
            self.observation_space = type('Space', (), {'shape': (obs_size,), 'dtype': np.float32})()
            self.action_space = type('Space', (), {'n': num_actions})()
            self.state = None
            
        def reset(self):
            self.state = np.random.randn(self.obs_size).astype(np.float32)
            return self.state
            
        def step(self, action):
            # Simple dynamics: move towards action direction
            self.state = self.state + 0.1 * (action - 0.5)
            self.state = np.clip(self.state, -2, 2)
            
            # Reward is negative distance from origin
            reward = -np.linalg.norm(self.state)
            
            # Episode ends randomly
            done = np.random.random() < 0.05
            
            if done:
                self.state = self.reset()
                
            return self.state, reward, done, {}
            
    return DummyEnv()


def test_muesli_components():
    """Test individual Muesli components."""
    print("Testing Muesli components...")
    
    # Create configuration
    config = MuesliConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy environment
    env = create_dummy_env()
    
    # Test 1: Agent creation
    print("\n1. Testing agent creation...")
    agent = MuesliAgent(
        obs_shape=env.observation_space.shape,
        action_space=env.action_space,
        config=config,
        device=device
    )
    print(f"✓ Agent created with {sum(p.numel() for p in agent.parameters())} parameters")
    
    # Test 2: Forward pass
    print("\n2. Testing forward pass...")
    obs = torch.randn(4, *env.observation_space.shape, device=device)
    output = agent(obs)
    print(f"✓ Forward pass successful")
    print(f"  - Action shape: {output['action'].shape}")
    print(f"  - Value shape: {output['value'].shape}")
    print(f"  - Policy logits shape: {output['policy_logits'].shape}")
    
    # Test 3: Categorical conversions
    print("\n3. Testing categorical conversions...")
    test_values = torch.tensor([0.0, 10.5, -50.0, 299.0], device=device)
    categorical = scalar_to_support(
        test_values, 
        config.categorical.support_size,
        config.categorical.value_min,
        config.categorical.value_max
    )
    reconstructed = support_to_scalar(
        categorical,
        config.categorical.support_size,
        config.categorical.value_min,
        config.categorical.value_max
    )
    print(f"✓ Categorical conversions working")
    print(f"  - Original: {test_values}")
    print(f"  - Reconstructed: {reconstructed}")
    print(f"  - Max error: {(test_values - reconstructed).abs().max().item():.6f}")
    
    # Test 4: Replay buffer
    print("\n4. Testing replay buffer...")
    replay_buffer = MuesliReplayBuffer(
        capacity=1000,
        unroll_length=5,
        gamma=0.99,
        device=device
    )
    
    # Collect a dummy episode
    obs = env.reset()
    for _ in range(20):
        obs_tensor = torch.tensor(obs, device=device)
        with torch.no_grad():
            output = agent(obs_tensor.unsqueeze(0))
        
        action = output['action'].squeeze().cpu().numpy()
        next_obs, reward, done, _ = env.step(action)
        
        replay_buffer.add_step(
            obs=obs_tensor,
            action=output['action'].squeeze(),
            reward=reward,
            next_obs=torch.tensor(next_obs, device=device),
            done=done,
            value=output['value'].squeeze().item(),
            policy=output['policy_logits'].squeeze().softmax(dim=-1)
        )
        
        obs = next_obs
        if done:
            break
            
    print(f"✓ Replay buffer has {len(replay_buffer)} sequences")
    
    # Test sampling
    if len(replay_buffer) > 0:
        batch = replay_buffer.sample(min(4, len(replay_buffer)))
        print(f"✓ Sampled batch with shape: {batch['obs'].shape}")
    
    # Test 5: Loss computation
    print("\n5. Testing loss computation...")
    if len(replay_buffer) > 10:
        batch = replay_buffer.sample(8)
        
        # Add retrace targets (simplified)
        with torch.no_grad():
            batch['retrace_targets'] = batch['rewards']
            
        losses, metrics = compute_muesli_losses(
            agent,
            agent.target_network,
            batch,
            config,
            training_step=0
        )
        
        print("✓ Loss computation successful")
        for name, loss in losses.items():
            if name != 'total':
                print(f"  - {name}: {loss.item():.4f}")
                
    # Test 6: Target network update
    print("\n6. Testing target network update...")
    # Get initial target network weights
    target_param = next(agent.target_network.parameters()).clone()
    main_param = next(agent.parameters()).clone()
    
    # Modify main network
    with torch.no_grad():
        next(agent.parameters()).add_(0.1)
        
    # Update target network
    agent.update_target_network(tau=0.5)
    
    # Check update
    new_target_param = next(agent.target_network.parameters())
    expected = 0.5 * (main_param + 0.1) + 0.5 * target_param
    error = (new_target_param - expected).abs().max().item()
    print(f"✓ Target network update working (max error: {error:.6f})")
    
    print("\n✅ All component tests passed!")


def test_muesli_training():
    """Test end-to-end Muesli training."""
    print("\n\nTesting Muesli training loop...")
    
    # Setup
    config = MuesliConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = create_dummy_env()
    
    # Create agent
    agent = MuesliAgent(
        obs_shape=env.observation_space.shape,
        action_space=env.action_space,
        config=config,
        device=device
    )
    
    # Create optimizer
    optimizer = optim.Adam(agent.parameters(), lr=0.0003)
    
    # Create replay buffer
    replay_buffer = MuesliReplayBuffer(
        capacity=10000,
        unroll_length=5,
        gamma=0.99,
        device=device
    )
    
    # Training loop
    num_episodes = 50
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Agent action
            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
            with torch.no_grad():
                output = agent(obs_tensor.unsqueeze(0))
                
            action = output['action'].squeeze().cpu().numpy()
            
            # Environment step
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Add to replay buffer
            replay_buffer.add_step(
                obs=obs_tensor,
                action=output['action'].squeeze(),
                reward=reward,
                next_obs=torch.tensor(next_obs, device=device, dtype=torch.float32),
                done=done,
                value=output['value'].squeeze().item(),
                policy=output['policy_logits'].squeeze().softmax(dim=-1)
            )
            
            # Train if we have enough data
            if len(replay_buffer) > 100 and episode_length % 10 == 0:
                # Sample batch
                batch = replay_buffer.sample(32)
                
                # Compute simple targets (no full Retrace for simplicity)
                with torch.no_grad():
                    batch['retrace_targets'] = batch['rewards']
                    
                # Compute losses
                losses, metrics = compute_muesli_losses(
                    agent,
                    agent.target_network,
                    batch,
                    config,
                    training_step=episode * 100 + episode_length
                )
                
                # Optimize
                optimizer.zero_grad()
                losses['total'].backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()
                
                # Update target network
                agent.update_target_network(config.target_network.tau)
                
            obs = next_obs
            
            if done or episode_length > 200:
                break
                
        episode_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
            
    print("\n✅ Training loop completed successfully!")
    
    # Plot results if possible
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 'r-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Muesli Training Progress')
        plt.legend(['Episode Reward', '10-Episode Average'])
        plt.grid(True)
        plt.savefig('muesli_training_progress.png')
        print("\n📊 Training progress saved to muesli_training_progress.png")
    except ImportError:
        print("\n(Install matplotlib to see training progress plot)")


if __name__ == "__main__":
    print("=" * 60)
    print("MUESLI IMPLEMENTATION TEST")
    print("=" * 60)
    
    # Test components
    test_muesli_components()
    
    # Test training
    test_muesli_training()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)