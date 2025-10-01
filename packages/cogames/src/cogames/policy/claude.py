"""Claude AI agent policy for CoGames using Anthropic API."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from cogames.policy.policy import AgentPolicy, Policy
from mettagrid import MettaGridEnv

logger = logging.getLogger("cogames.policy.claude")


class ClaudeAgentPolicyImpl(AgentPolicy):
    """Per-agent policy that uses Claude AI to make decisions."""

    def __init__(
        self,
        env: MettaGridEnv,
        agent_id: int,
        api_key: str,
        prompt: str,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        """Initialize Claude agent policy.

        Args:
            env: The game environment
            agent_id: ID of this agent
            api_key: Anthropic API key
            prompt: System prompt for Claude
            model: Claude model to use
        """
        self._env = env
        self._agent_id = agent_id
        self._api_key = api_key
        self._prompt = prompt
        self._model = model
        self._action_space = env.single_action_space
        self._conversation_history: list[dict[str, str]] = []
        self._step_count = 0

        # Try to import anthropic
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "anthropic package not found. Install with: pip install anthropic or uv add anthropic"
            ) from e

    def _format_observation(self, obs: Any) -> str:
        """Format observation into a string for Claude."""
        # obs is typically a numpy array representing the observation
        # Format it in a human-readable way
        obs_shape = obs.shape if hasattr(obs, "shape") else len(obs)
        obs_summary = f"Observation shape: {obs_shape}\n"

        # Add environment state information if available
        action_names = getattr(self._env, "action_names", None)
        resource_names = getattr(self._env, "resource_names", None)

        if action_names:
            obs_summary += f"\nAvailable actions: {', '.join(action_names)}\n"

        if resource_names:
            obs_summary += f"Available resources: {', '.join(resource_names)}\n"

        # Convert observation to a readable format
        if isinstance(obs, np.ndarray):
            # For large observations, provide a summary rather than full array
            if obs.size > 100:
                obs_summary += "\nObservation statistics:\n"
                obs_summary += f"  Min: {obs.min()}, Max: {obs.max()}, Mean: {obs.mean():.2f}\n"
                obs_summary += f"  Shape: {obs.shape}\n"
            else:
                obs_summary += f"\nObservation data:\n{obs}\n"

        return obs_summary

    def _parse_action(self, response: str) -> Any:
        """Parse Claude's response into an action.

        Expects response to contain action in format like "action: 0, argument: 1"
        or just numbers that can be parsed.
        """
        try:
            # Try to extract numbers from the response
            import re

            numbers = re.findall(r"\d+", response)
            if len(numbers) >= 2:
                action = np.array([int(numbers[0]), int(numbers[1])], dtype=np.int32)
                # Validate action is within bounds
                if action[0] < self._action_space.nvec[0] and action[1] < self._action_space.nvec[1]:
                    return action

            # If parsing fails, return a random action
            logger.warning(f"Could not parse action from Claude response: {response}. Using random action.")
            return self._action_space.sample()

        except Exception as e:
            logger.error(f"Error parsing action: {e}. Using random action.")
            return self._action_space.sample()

    def step(self, obs: Any) -> Any:
        """Get action from Claude given an observation.

        Maintains conversation history to provide context across multiple steps.
        """
        obs_text = self._format_observation(obs)
        self._step_count += 1

        # Build the message to send to Claude
        user_message = f"Step {self._step_count} - Agent {self._agent_id} observation:\n{obs_text}\n\n"
        user_message += f"Action space: {self._action_space.nvec}\n"
        user_message += "Respond with an action in the format: action: X, argument: Y"

        # Add user message to conversation history
        self._conversation_history.append({"role": "user", "content": user_message})

        try:
            # Call Claude API with full conversation history (pass a copy to avoid mutation issues)
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=self._prompt,
                messages=list(self._conversation_history),
            )

            # Extract text from response
            response_text = response.content[0].text
            logger.debug(f"Claude response: {response_text}")

            # Add assistant response to conversation history
            self._conversation_history.append({"role": "assistant", "content": response_text})

            # Parse the action
            action = self._parse_action(response_text)
            return action

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}. Using random action.")
            # Remove the failed user message from history
            if self._conversation_history and self._conversation_history[-1]["role"] == "user":
                self._conversation_history.pop()
            return self._action_space.sample()

    def reset_history(self) -> None:
        """Reset conversation history and step count.

        Useful when starting a new episode or game.
        """
        self._conversation_history.clear()
        self._step_count = 0
        logger.debug(f"Reset conversation history for agent {self._agent_id}")

    def get_history_length(self) -> int:
        """Get the number of messages in the conversation history."""
        return len(self._conversation_history)

    def get_step_count(self) -> int:
        """Get the current step count."""
        return self._step_count


class ClaudePolicy(Policy):
    """Claude AI policy that creates per-agent Claude policies."""

    def __init__(
        self,
        env: MettaGridEnv,
        device: Optional[Any] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        """Initialize Claude policy.

        Args:
            env: The game environment
            device: Device (ignored for Claude policy)
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            prompt: System prompt for Claude (if None, uses default)
            model: Claude model to use
        """
        self._env = env
        self._model = model

        # Get API key from env if not provided
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter, or use --policy-data with a config file."
            )

        # Use default prompt if not provided
        self._prompt = prompt or self._default_prompt()

    def _default_prompt(self) -> str:
        """Generate default system prompt for Claude."""
        action_names = getattr(self._env, "action_names", [])
        resource_names = getattr(self._env, "resource_names", [])

        prompt = "You are an AI agent playing a cooperative multi-agent game.\n\n"
        prompt += "Your goal is to cooperate with other agents to maximize rewards.\n\n"

        if action_names:
            prompt += f"Available actions: {', '.join(action_names)}\n"

        if resource_names:
            prompt += f"Available resources: {', '.join(resource_names)}\n"

        prompt += "\nWhen responding, provide an action in the format: action: X, argument: Y\n"
        prompt += "where X and Y are integers representing the action type and argument."

        return prompt

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create a Claude AgentPolicy instance for a specific agent."""
        return ClaudeAgentPolicyImpl(
            env=self._env,
            agent_id=agent_id,
            api_key=self._api_key,
            prompt=self._prompt,
            model=self._model,
        )

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load policy configuration from YAML file.

        Expected YAML format:
        ```yaml
        api_key: "sk-ant-..."
        prompt: "Custom system prompt"
        model: "claude-sonnet-4-5-20250929"  # optional
        ```
        """
        path = Path(policy_data_path)
        if not path.exists():
            raise ValueError(f"Policy data file not found: {policy_data_path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Update configuration
        if "api_key" in config:
            self._api_key = config["api_key"]

        if "prompt" in config:
            self._prompt = config["prompt"]

        if "model" in config:
            self._model = config["model"]

        logger.info(f"Loaded Claude policy configuration from {policy_data_path}")

    def save_policy_data(self, policy_data_path: str) -> None:
        """Save policy configuration to YAML file."""
        config = {
            "model": self._model,
            "prompt": self._prompt,
            # Don't save API key for security
        }

        path = Path(policy_data_path)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved Claude policy configuration to {policy_data_path}")
