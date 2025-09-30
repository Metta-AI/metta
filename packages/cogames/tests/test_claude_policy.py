"""Tests for ClaudePolicy."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from cogames.policy.claude import ClaudePolicy


@pytest.fixture
def mock_env():
    """Create a mock environment."""
    env = Mock()
    env.num_agents = 2
    env.action_names = ["move", "interact"]
    env.resource_names = ["wood", "stone"]

    # Mock action space
    action_space = Mock()
    action_space.nvec = np.array([2, 4])
    action_space.sample = lambda: np.array([0, 1], dtype=np.int32)
    env.single_action_space = action_space

    return env


def test_claude_policy_init_with_api_key(mock_env):
    """Test initializing ClaudePolicy with explicit API key."""
    policy = ClaudePolicy(mock_env, api_key="test-key")
    assert policy._api_key == "test-key"


def test_claude_policy_init_from_env(mock_env, monkeypatch):
    """Test initializing ClaudePolicy from environment variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    policy = ClaudePolicy(mock_env)
    assert policy._api_key == "env-key"


def test_claude_policy_init_no_api_key(mock_env, monkeypatch):
    """Test that missing API key raises error."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Anthropic API key not provided"):
        ClaudePolicy(mock_env)


def test_claude_policy_default_prompt(mock_env):
    """Test that default prompt is generated correctly."""
    policy = ClaudePolicy(mock_env, api_key="test-key")
    assert "cooperative multi-agent game" in policy._prompt
    assert "move" in policy._prompt
    assert "interact" in policy._prompt


def test_claude_policy_custom_prompt(mock_env):
    """Test using custom prompt."""
    custom_prompt = "Custom AI behavior"
    policy = ClaudePolicy(mock_env, api_key="test-key", prompt=custom_prompt)
    assert policy._prompt == custom_prompt


def test_claude_policy_agent_policy_creation(mock_env):
    """Test creating per-agent policies."""
    # Mock anthropic module before creating policy
    with patch.dict("sys.modules", {"anthropic": MagicMock()}):
        policy = ClaudePolicy(mock_env, api_key="test-key")
        agent_policy = policy.agent_policy(0)
        assert agent_policy is not None
        assert agent_policy._agent_id == 0


def test_claude_policy_load_policy_data(mock_env, tmp_path):
    """Test loading policy configuration from YAML."""
    config_file = tmp_path / "claude_settings.yaml"
    config_file.write_text(
        """
api_key: yaml-key
prompt: Custom prompt from YAML
model: claude-sonnet-4-5-20250929
"""
    )

    policy = ClaudePolicy(mock_env, api_key="test-key")
    policy.load_policy_data(str(config_file))

    assert policy._api_key == "yaml-key"
    assert policy._prompt == "Custom prompt from YAML"
    assert policy._model == "claude-sonnet-4-5-20250929"


def test_claude_policy_save_policy_data(mock_env, tmp_path):
    """Test saving policy configuration to YAML."""
    config_file = tmp_path / "claude_settings.yaml"

    policy = ClaudePolicy(mock_env, api_key="test-key", prompt="Test prompt")
    policy.save_policy_data(str(config_file))

    assert config_file.exists()
    content = config_file.read_text()
    assert "Test prompt" in content
    assert "claude-sonnet-4-5-20250929" in content
    # API key should not be saved
    assert "test-key" not in content


def test_claude_agent_policy_step(mock_env):
    """Test agent policy step with mocked Anthropic API."""
    # Mock the API response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="action: 1, argument: 2")]
    mock_client.messages.create.return_value = mock_response

    # Create mock anthropic module
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    # Mock anthropic module before creating policy
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        policy = ClaudePolicy(mock_env, api_key="test-key")
        agent_policy = policy.agent_policy(0)

        obs = np.zeros((10, 10), dtype=np.uint8)
        action = agent_policy.step(obs)

        assert isinstance(action, np.ndarray)
        assert len(action) == 2


def test_claude_agent_policy_maintains_history(mock_env):
    """Test that agent policy maintains conversation history across steps."""
    # Mock the API response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="action: 0, argument: 1")]
    mock_client.messages.create.return_value = mock_response

    # Create mock anthropic module
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    # Mock anthropic module before creating policy
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        policy = ClaudePolicy(mock_env, api_key="test-key")
        agent_policy = policy.agent_policy(0)

        # Initially, history should be empty
        assert agent_policy.get_history_length() == 0
        assert agent_policy.get_step_count() == 0

        # Take first step
        obs = np.zeros((10, 10), dtype=np.uint8)
        agent_policy.step(obs)

        # History should now have 2 messages (user + assistant)
        assert agent_policy.get_history_length() == 2
        assert agent_policy.get_step_count() == 1

        # Take second step
        agent_policy.step(obs)

        # History should now have 4 messages
        assert agent_policy.get_history_length() == 4
        assert agent_policy.get_step_count() == 2

        # Verify that messages were passed to API with growing history
        calls = mock_client.messages.create.call_args_list
        assert len(calls) == 2
        # First call should have 1 message (just the user message)
        first_call_messages = calls[0].kwargs["messages"]
        assert len(first_call_messages) == 1
        assert first_call_messages[0]["role"] == "user"
        # Second call should have 3 messages (user, assistant, user)
        second_call_messages = calls[1].kwargs["messages"]
        assert len(second_call_messages) == 3
        assert second_call_messages[0]["role"] == "user"
        assert second_call_messages[1]["role"] == "assistant"
        assert second_call_messages[2]["role"] == "user"


def test_claude_agent_policy_reset_history(mock_env):
    """Test resetting conversation history."""
    # Mock the API response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="action: 0, argument: 1")]
    mock_client.messages.create.return_value = mock_response

    # Create mock anthropic module
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    # Mock anthropic module before creating policy
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        policy = ClaudePolicy(mock_env, api_key="test-key")
        agent_policy = policy.agent_policy(0)

        # Take a few steps
        obs = np.zeros((10, 10), dtype=np.uint8)
        agent_policy.step(obs)
        agent_policy.step(obs)

        # Verify history is populated
        assert agent_policy.get_history_length() == 4
        assert agent_policy.get_step_count() == 2

        # Reset history
        agent_policy.reset_history()

        # Verify history is cleared
        assert agent_policy.get_history_length() == 0
        assert agent_policy.get_step_count() == 0


def test_claude_agent_policy_step_includes_step_count(mock_env):
    """Test that step messages include the step count."""
    # Mock the API response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="action: 0, argument: 1")]
    mock_client.messages.create.return_value = mock_response

    # Create mock anthropic module
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    # Mock anthropic module before creating policy
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        policy = ClaudePolicy(mock_env, api_key="test-key")
        agent_policy = policy.agent_policy(0)

        # Take a step
        obs = np.zeros((10, 10), dtype=np.uint8)
        agent_policy.step(obs)

        # Verify the message includes step count
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]["messages"]
        assert "Step 1" in messages[0]["content"]
