"""
Integration tests for MettaAgent using the modern configuration system.
These tests use real environments and configurations without Hydra/YAML.
"""

import pytest
import torch
from tensordict import TensorDict

import metta.mettagrid.config.envs as eb
from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import MettaAgent
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.system_config import SystemConfig


@pytest.fixture
def create_env_and_agent():
    """Create a real environment and agent using modern configuration."""
    # Create a small test environment
    env_config = eb.make_navigation(num_agents=1)
    env_config.game.max_steps = 100
    env_config.game.map_builder.width = 8
    env_config.game.map_builder.height = 8
    
    # Create the actual environment
    env = MettaGridEnv(env_config, num_envs=2)  # 2 parallel environments for testing
    
    # Create system and agent configs
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = AgentConfig(name="fast")
    
    # Create the agent
    agent = MettaAgent(
        env=env,
        system_cfg=system_cfg,
        policy_architecture_cfg=agent_cfg,
    )
    
    # Initialize agent to environment
    features = env.get_observation_features()
    agent.initialize_to_environment(
        features,
        env.action_names,
        env.max_action_args,
        device="cpu"
    )
    
    return env, agent


def test_full_forward_pass(create_env_and_agent):
    """Test a complete forward pass with a real environment."""
    env, agent = create_env_and_agent
    
    # Reset environment to get initial observations
    obs, info = env.reset()
    
    # obs is already a TensorDict from MettagridEnv
    assert isinstance(obs, TensorDict)
    assert "env_obs" in obs
    
    # Forward pass through agent
    output = agent(obs)
    
    # Check output structure
    assert isinstance(output, TensorDict)
    assert "actions" in output
    assert "values" in output
    assert "act_log_prob" in output
    
    # Check action shape matches environment expectations
    actions = output["actions"]
    assert actions.shape[0] == 2  # batch size (num_envs)
    assert actions.shape[1] == 2  # action dimensions (action_type, action_param)


def test_action_sampling_and_stepping(create_env_and_agent):
    """Test that sampled actions can be used to step the environment."""
    env, agent = create_env_and_agent
    
    # Reset and get initial observation
    obs, info = env.reset()
    
    # Get actions from agent
    output = agent(obs)
    actions = output["actions"]
    
    # Step environment with sampled actions
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Check that step returns valid data
    assert isinstance(next_obs, TensorDict)
    assert rewards.shape == (2,)  # One reward per environment
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)


def test_memory_handling_with_episodes(create_env_and_agent):
    """Test memory handling across episode boundaries."""
    env, agent = create_env_and_agent
    
    # Reset memory at start
    agent.reset_memory()
    initial_memory = agent.get_memory()
    assert isinstance(initial_memory, dict)
    
    # Run a few steps
    obs, _ = env.reset()
    for _ in range(3):
        output = agent(obs)
        actions = output["actions"]
        obs, rewards, terminated, truncated, _ = env.step(actions)
        
        # If any environment terminates, reset its memory
        if terminated.any() or truncated.any():
            agent.reset_memory()
            break
    
    # Memory should be reset after termination
    memory_after_reset = agent.get_memory()
    assert isinstance(memory_after_reset, dict)


def test_action_distribution_properties(create_env_and_agent):
    """Test properties of action distributions."""
    env, agent = create_env_and_agent
    
    # Get observation
    obs, _ = env.reset()
    
    # Forward pass
    output = agent(obs)
    
    # Check that we have log probabilities
    assert "act_log_prob" in output
    log_probs = output["act_log_prob"]
    
    # Log probabilities should be negative (or zero for deterministic actions)
    assert torch.all(log_probs <= 0)
    
    # Check that we have full log probs for all actions
    if "full_log_probs" in output:
        full_log_probs = output["full_log_probs"]
        # Should have probabilities for all possible action combinations
        assert full_log_probs.dim() == 2


def test_value_estimation(create_env_and_agent):
    """Test that value estimates are produced."""
    env, agent = create_env_and_agent
    
    # Get observation
    obs, _ = env.reset()
    
    # Forward pass
    output = agent(obs)
    
    # Check value estimates
    assert "values" in output
    values = output["values"]
    
    # Values should be scalar per environment
    assert values.shape == (2,)  # One value per environment
    assert values.dtype == torch.float32


def test_batch_processing(create_env_and_agent):
    """Test that agent can handle different batch sizes."""
    env, agent = create_env_and_agent
    
    # Test with different numbers of environments
    for num_envs in [1, 4, 8]:
        # Create environment with specific batch size
        env_config = eb.make_navigation(num_agents=1)
        env_config.game.max_steps = 100
        batch_env = MettagridEnv(env_config, num_envs=num_envs)
        
        # Get observations
        obs, _ = batch_env.reset()
        
        # Forward pass
        output = agent(obs)
        
        # Check batch dimensions match
        assert output["actions"].shape[0] == num_envs
        assert output["values"].shape[0] == num_envs


def test_training_mode_vs_inference(create_env_and_agent):
    """Test differences between training and inference modes."""
    env, agent = create_env_and_agent
    
    # Get observation
    obs, _ = env.reset()
    
    # Test in training mode
    agent.train()
    output_train = agent(obs)
    
    # Test in evaluation mode
    agent.eval()
    output_eval = agent(obs)
    
    # Both should produce valid outputs
    assert "actions" in output_train
    assert "actions" in output_eval
    assert "values" in output_train
    assert "values" in output_eval


def test_checkpoint_compatibility(create_env_and_agent):
    """Test that agent state can be saved and loaded."""
    env, agent = create_env_and_agent
    
    # Get initial output
    obs, _ = env.reset()
    output_before = agent(obs)
    
    # Save state
    state_dict = agent.state_dict()
    
    # Create new agent and load state
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = AgentConfig(name="fast")
    new_agent = MettaAgent(
        env=env,
        system_cfg=system_cfg,
        policy_architecture_cfg=agent_cfg,
    )
    
    # Initialize and load state
    features = env.get_observation_features()
    new_agent.initialize_to_environment(
        features,
        env.action_names,
        env.max_action_args,
        device="cpu"
    )
    new_agent.load_state_dict(state_dict)
    
    # Outputs should be similar (not identical due to sampling)
    new_agent.eval()
    agent.eval()
    output_after = new_agent(obs)
    
    # Values should be identical in eval mode
    torch.testing.assert_close(output_before["values"], output_after["values"])


def test_multi_agent_environment(create_env_and_agent):
    """Test agent with multi-agent environments."""
    # Create multi-agent environment
    env_config = eb.make_arena(num_agents=4)
    env_config.game.max_steps = 100
    env_config.game.map_builder.width = 16
    env_config.game.map_builder.height = 16
    
    multi_env = MettagridEnv(env_config, num_envs=2)
    
    # Create agent
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = AgentConfig(name="latent_attn_tiny")  # Use attention model for multi-agent
    
    agent = MettaAgent(
        env=multi_env,
        system_cfg=system_cfg,
        policy_architecture_cfg=agent_cfg,
    )
    
    # Initialize
    features = multi_env.get_observation_features()
    agent.initialize_to_environment(
        features,
        multi_env.action_names,
        multi_env.max_action_args,
        device="cpu"
    )
    
    # Reset and step
    obs, _ = multi_env.reset()
    output = agent(obs)
    
    # Should handle multiple agents (4 agents × 2 envs = 8 total)
    assert output["actions"].shape[0] == 8
    assert output["values"].shape[0] == 8


def test_different_agent_architectures():
    """Test that different agent architectures work correctly."""
    architectures = ["fast", "latent_attn_tiny", "latent_attn_small"]
    
    for arch_name in architectures:
        # Create environment
        env_config = eb.make_navigation(num_agents=1)
        env = MettagridEnv(env_config, num_envs=2)
        
        # Create agent with specific architecture
        system_cfg = SystemConfig(device="cpu")
        agent_cfg = AgentConfig(name=arch_name)
        
        agent = MettaAgent(
            env=env,
            system_cfg=system_cfg,
            policy_architecture_cfg=agent_cfg,
        )
        
        # Initialize
        features = env.get_observation_features()
        agent.initialize_to_environment(
            features,
            env.action_names,
            env.max_action_args,
            device="cpu"
        )
        
        # Test forward pass
        obs, _ = env.reset()
        output = agent(obs)
        
        # All architectures should produce valid outputs
        assert "actions" in output
        assert "values" in output
        assert output["actions"].shape == (2, 2)  # (batch, action_dims)


def test_pytorch_vs_component_policies():
    """Test both PyTorch and ComponentPolicy implementations."""
    # Test ComponentPolicy version
    env_config = eb.make_navigation(num_agents=1)
    env = MettagridEnv(env_config, num_envs=2)
    
    system_cfg = SystemConfig(device="cpu")
    
    # ComponentPolicy version
    component_cfg = AgentConfig(name="fast")
    component_agent = MettaAgent(env=env, system_cfg=system_cfg, policy_architecture_cfg=component_cfg)
    
    # PyTorch version
    pytorch_cfg = AgentConfig(name="pytorch/fast")
    pytorch_agent = MettaAgent(env=env, system_cfg=system_cfg, policy_architecture_cfg=pytorch_cfg)
    
    # Initialize both
    features = env.get_observation_features()
    for agent in [component_agent, pytorch_agent]:
        agent.initialize_to_environment(
            features,
            env.action_names,
            env.max_action_args,
            device="cpu"
        )
    
    # Both should work
    obs, _ = env.reset()
    
    component_output = component_agent(obs)
    pytorch_output = pytorch_agent(obs)
    
    # Both should produce similar structure
    assert "actions" in component_output
    assert "actions" in pytorch_output
    assert component_output["actions"].shape == pytorch_output["actions"].shape