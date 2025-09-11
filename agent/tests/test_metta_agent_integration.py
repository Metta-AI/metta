import numpy as np
import pytest
import torch
from tensordict import TensorDict

import metta.mettagrid.builder.envs as eb
from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import MettaAgent
from metta.agent.utils import obs_to_td
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.system_config import SystemConfig


@pytest.fixture
def create_env_and_agent():
    """Create a real environment and agent using modern configuration."""
    # Create a small test environment
    mg_config = eb.make_navigation(num_agents=1)
    mg_config.game.max_steps = 100
    mg_config.game.map_builder.width = 8
    mg_config.game.map_builder.height = 8

    # Create a single environment (vectorization handled separately if needed)
    env = MettaGridEnv(mg_config, render_mode=None)

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
    agent.initialize_to_environment(features, env.action_names, env.max_action_args, device="cpu")

    return env, agent


def test_full_forward_pass(create_env_and_agent):
    """Test a complete forward pass with a real environment."""
    env, agent = create_env_and_agent

    # Reset environment to get initial observations
    obs, info = env.reset()

    # MettaGridEnv returns tokenized observations as numpy arrays
    assert isinstance(obs, np.ndarray)

    # Forward pass with automatic TensorDict conversion
    output = agent(obs_to_td(obs))

    # Check output structure
    assert isinstance(output, TensorDict)
    assert "actions" in output
    assert "values" in output
    assert "act_log_prob" in output

    # Check action shape matches environment expectations
    actions = output["actions"]
    assert actions.shape[0] == obs.shape[0]  # batch size matches number of agents
    assert actions.shape[1] == 2  # action dimensions (action_type, action_param)


def test_action_sampling_and_stepping(create_env_and_agent):
    """Test that sampled actions can be used to step the environment."""
    env, agent = create_env_and_agent

    # Reset and get initial observation
    obs, info = env.reset()

    # Get actions from agent
    output = agent(obs_to_td(obs))
    actions = output["actions"]

    # Convert actions to numpy for environment
    actions_np = actions.numpy()  # Keep shape as (num_agents, 2)

    # Step environment with sampled actions
    next_obs, rewards, terminated, truncated, info = env.step(actions_np)

    # Check that step returns valid data
    assert isinstance(next_obs, np.ndarray)  # Tokenized observations
    assert isinstance(rewards, (int, float, np.ndarray))  # Single reward or array
    assert isinstance(terminated, (bool, np.ndarray))
    assert isinstance(truncated, (bool, np.ndarray))


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
        output = agent(obs_to_td(obs))
        actions = output["actions"].numpy()
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

    # Forward pass with automatic conversion
    output = agent(obs_to_td(obs))

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

    # Forward pass with automatic conversion
    output = agent(obs_to_td(obs))

    # Check value estimates
    assert "values" in output
    values = output["values"]

    # Values should be scalar per agent
    assert values.shape == (obs.shape[0],)  # One value per agent
    assert values.dtype == torch.float32


def test_batch_processing(create_env_and_agent):
    """Test that agent processes the natural batch size from environment."""
    env, agent = create_env_and_agent

    # Get observation from environment
    obs, _ = env.reset()

    # Forward pass with automatic conversion
    output = agent(obs_to_td(obs))

    # Check batch dimensions match observation batch size
    assert output["actions"].shape[0] == obs.shape[0]
    assert output["values"].shape[0] == obs.shape[0]

    # The actual batch processing at scale is handled by VecEnv wrapper,
    # not by manually replicating observations


def test_training_mode_vs_inference(create_env_and_agent):
    """Test differences between training and inference modes."""
    env, agent = create_env_and_agent

    # Get observation
    obs, _ = env.reset()

    # Test in training mode
    agent.train()
    output_train = agent(obs_to_td(obs))

    # Test in evaluation mode
    agent.eval()
    output_eval = agent(obs_to_td(obs))

    # Both should produce valid outputs
    assert "actions" in output_train
    assert "actions" in output_eval
    assert "values" in output_train
    assert "values" in output_eval


def test_multi_agent_environment(create_env_and_agent):
    """Test agent with multi-agent environments."""
    # Create multi-agent environment
    mg_config = eb.make_arena(num_agents=6)
    mg_config.game.max_steps = 100
    mg_config.game.map_builder.width = 16
    mg_config.game.map_builder.height = 16

    multi_env = MettaGridEnv(mg_config, render_mode=None)

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
    agent.initialize_to_environment(features, multi_env.action_names, multi_env.max_action_args, device="cpu")

    # Reset and step
    obs, _ = multi_env.reset()

    # For multi-agent, MettaGridEnv returns tokenized observations for all agents
    output = agent(obs_to_td(obs))  # Automatically handles batch_size from obs.shape[0]

    # Should handle multiple agents (4 agents in single env)
    assert output["actions"].shape[0] == 6
    assert output["values"].shape[0] == 6


def test_different_agent_architectures():
    """Test that different agent architectures work correctly."""
    architectures = ["fast", "latent_attn_tiny", "latent_attn_small"]

    for arch_name in architectures:
        # Create environment
        mg_config = eb.make_navigation(num_agents=1)
        env = MettaGridEnv(mg_config, render_mode=None)

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
        agent.initialize_to_environment(features, env.action_names, env.max_action_args, device="cpu")

        # Test forward pass
        obs, _ = env.reset()
        output = agent(obs_to_td(obs))

        # All architectures should produce valid outputs
        assert "actions" in output
        assert "values" in output
        assert output["actions"].shape == (obs.shape[0], 2)  # (batch, action_dims)


@pytest.mark.skip(reason="PyTorch agents need different initialization")
def test_pytorch_vs_component_policies():
    """Test both PyTorch and ComponentPolicy implementations."""
    # Test ComponentPolicy version
    mg_config = eb.make_navigation(num_agents=1)
    env = MettaGridEnv(mg_config, render_mode=None)

    system_cfg = SystemConfig(device="cpu")

    # ComponentPolicy version (latent_attn_tiny)
    component_cfg = AgentConfig(name="latent_attn_tiny")
    component_agent = MettaAgent(env=env, system_cfg=system_cfg, policy_architecture_cfg=component_cfg)

    # PyTorch version (pytorch/latent_attn_tiny)
    pytorch_cfg = AgentConfig(name="pytorch/latent_attn_tiny")
    pytorch_agent = MettaAgent(env=env, system_cfg=system_cfg, policy_architecture_cfg=pytorch_cfg)

    # Initialize both
    features = env.get_observation_features()
    for agent in [component_agent, pytorch_agent]:
        agent.initialize_to_environment(features, env.action_names, env.max_action_args, device="cpu")

    # Both should work
    obs, _ = env.reset()
    # Note: This test needs unsqueeze for batch dimension compatibility
    obs_single = obs.reshape(1, *obs.shape[1:])  # Add batch dimension

    component_output = component_agent(obs_to_td(obs_single))
    pytorch_output = pytorch_agent(obs_to_td(obs_single))

    # Both should produce similar structure
    assert "actions" in component_output
    assert "actions" in pytorch_output
    assert component_output["actions"].shape == pytorch_output["actions"].shape
