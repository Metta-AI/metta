"""LLM-based policy for MettaGrid using GPT or Claude."""

import json
import logging
import os
import random
import subprocess
from typing import Literal

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

logger = logging.getLogger(__name__)


def check_ollama_available() -> bool:
    """Check if Ollama server is running.

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # Try to list models as a health check
        client.models.list()
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """List available Ollama models.

    Returns:
        List of model names
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: skip header line, extract model names
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return []


def ensure_ollama_model(model: str | None = None) -> str:
    """Ensure an Ollama model is available, pulling if necessary.

    Args:
        model: Model name to check/pull, or None to use default

    Returns:
        The model name that is available

    Raises:
        RuntimeError: If Ollama is not available or model pull fails
    """
    if not check_ollama_available():
        raise RuntimeError(
            "Ollama server is not running. Please start it with 'ollama serve' "
            "or install from https://ollama.ai"
        )

    available_models = list_ollama_models()

    # If no model specified, try to use an available one
    if model is None:
        if available_models:
            model = available_models[0]
            logger.info(f"Using available Ollama model: {model}")
            return model
        else:
            # Pull default model
            model = "llama3.2"
            logger.info(f"No models found. Pulling default model: {model}")

    # Check if model is already available
    if any(model in m for m in available_models):
        return model

    # Try to pull the model
    logger.info(f"Pulling Ollama model: {model}...")
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
            capture_output=False  # Show progress
        )
        logger.info(f"Successfully pulled model: {model}")
        return model
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}") from e

GAME_RULES_PROMPT = """You are playing MettaGrid, a multi-agent gridworld game.

OBJECTS YOU MIGHT SEE:
- Altar: Use energy here to gain rewards (costs energy, has cooldown)
- Converter: Convert resources to energy (no energy cost, has cooldown)
- Generator: Harvest resources from here (has cooldown)
- Wall: Impassable barrier
- Agent: Other players in the game

AVAILABLE ACTIONS:
- move: Move forward or backward relative to your orientation
- rotate: Change facing direction (down, left, right, up)
- attack: Attack agents or objects in your range (costs energy, freezes target, steals resources)
- shield: Toggle shield protection (consumes energy while active, protects from attacks)
- use: Interact with nearby objects (altar/converter/generator)
- noop: Do nothing this turn

GAME MECHANICS:
- Energy is required for most actions
- Harvest resources from generators
- Convert resources to energy at converters
- Use altars to gain rewards (this is your main goal)
- Attacks freeze targets and steal their resources
- Shield protects you but drains energy

STRATEGY TIPS:
- Prioritize energy management
- Gather resources when energy is low
- Convert resources to energy at converters
- Use altars when you have enough energy
- Be aware of other agents (potential threats or allies)

Your goal is to maximize rewards by using the altar efficiently while managing your resources and energy.
"""


def observation_to_json(obs: AgentObservation, policy_env_info: PolicyEnvInterface) -> dict:
    """Convert observation tokens to structured JSON for LLM consumption.

    Args:
        obs: Agent observation containing tokens
        policy_env_info: Policy environment interface with feature specs

    Returns:
        Dictionary with structured observation data
    """
    tokens_list = []

    for token in obs.tokens:
        token_dict = {
            "feature": token.feature.name,
            "location": {
                "x": token.col(),
                "y": token.row()
            },
            "value": token.value
        }
        tokens_list.append(token_dict)

    return {
        "agent_id": obs.agent_id,
        "visible_objects": tokens_list,
        "available_actions": policy_env_info.action_names,
        "num_visible_objects": len(tokens_list)
    }


class LLMAgentPolicy(AgentPolicy):
    """Per-agent LLM policy that queries GPT or Claude for action selection."""

    # Class-level tracking for all instances
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7
    ):
        """Initialize LLM agent policy.

    Args:
        policy_env_info: Policy environment interface
        provider: LLM provider ("openai", "anthropic", or "ollama")
        model: Model name (defaults: gpt-4o-mini, claude-3-5-sonnet, or llama3.2 for ollama)
        temperature: Sampling temperature for LLM
    """
        super().__init__(policy_env_info)
        self.provider = provider
        self.temperature = temperature

        # Initialize LLM client
        if self.provider == "openai":
            from openai import OpenAI
            self.client: OpenAI | None = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.anthropic_client = None
            self.ollama_client = None
            self.model = model or "gpt-4o-mini"
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = None
            self.anthropic_client: Anthropic | None = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.ollama_client = None
            self.model = model or "claude-3-5-sonnet-20241022"
        elif self.provider == "ollama":
            from openai import OpenAI
            self.client = None
            self.anthropic_client = None

            # Ensure Ollama is available and model is pulled
            try:
                self.model = ensure_ollama_model(model)
            except RuntimeError as e:
                logger.error(f"Ollama setup failed: {e}")
                raise

            self.ollama_client: OpenAI | None = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Ollama doesn't need a real API key
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def step(self, obs: AgentObservation) -> Action:
        """Get action from LLM given observation.

        Args:
            obs: Agent observation

        Returns:
            Action to take
        """
        # Convert observation to JSON
        obs_json = observation_to_json(obs, self.policy_env_info)

        # Create user prompt
        user_prompt = f"""Current game state:
{json.dumps(obs_json, indent=2)}

Based on the visible objects and game rules, choose the BEST action to maximize your rewards.

Respond with ONLY the action name from the available actions list. No explanation, just the action name.
"""

        # Query LLM
        try:
            action_name = "noop"  # Default fallback

            if self.provider == "openai":
                assert self.client is not None
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": GAME_RULES_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=50
                )
                action_name = response.choices[0].message.content
                if action_name is None:
                    action_name = "noop"
                action_name = action_name.strip()

                # Track usage and cost
                usage = response.usage
                if usage:
                    LLMAgentPolicy.total_calls += 1
                    LLMAgentPolicy.total_input_tokens += usage.prompt_tokens
                    LLMAgentPolicy.total_output_tokens += usage.completion_tokens

                    # Cost calculation for gpt-4o-mini: $0.150/1M input, $0.600/1M output
                    input_cost = (usage.prompt_tokens / 1_000_000) * 0.150
                    output_cost = (usage.completion_tokens / 1_000_000) * 0.600
                    call_cost = input_cost + output_cost
                    LLMAgentPolicy.total_cost += call_cost

                    logger.debug(
                        f"OpenAI response: '{action_name}' | "
                        f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out | "
                        f"Cost: ${call_cost:.6f}"
                    )

            elif self.provider == "ollama":
                assert self.ollama_client is not None
                response = self.ollama_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": GAME_RULES_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=50
                )
                action_name = response.choices[0].message.content
                if action_name is None:
                    action_name = "noop"
                action_name = action_name.strip()

                # Track usage (Ollama is free/local)
                usage = response.usage
                if usage:
                    LLMAgentPolicy.total_calls += 1
                    LLMAgentPolicy.total_input_tokens += usage.prompt_tokens
                    LLMAgentPolicy.total_output_tokens += usage.completion_tokens
                    # No cost for local Ollama

                    logger.debug(
                        f"Ollama response: '{action_name}' | "
                        f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out | "
                        f"Cost: $0.00 (local)"
                    )

            elif self.provider == "anthropic":
                assert self.anthropic_client is not None
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    system=GAME_RULES_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self.temperature,
                    max_tokens=50
                )
                # Extract text from response content blocks
                from anthropic.types import TextBlock
                action_name = "noop"
                for block in response.content:
                    if isinstance(block, TextBlock):
                        action_name = block.text.strip()
                        break

                # Track usage and cost
                usage = response.usage
                LLMAgentPolicy.total_calls += 1
                LLMAgentPolicy.total_input_tokens += usage.input_tokens
                LLMAgentPolicy.total_output_tokens += usage.output_tokens

                # Cost calculation for claude-3-5-sonnet: $3.00/1M input, $15.00/1M output
                input_cost = (usage.input_tokens / 1_000_000) * 3.00
                output_cost = (usage.output_tokens / 1_000_000) * 15.00
                call_cost = input_cost + output_cost
                LLMAgentPolicy.total_cost += call_cost

                logger.debug(
                    f"Anthropic response: '{action_name}' | "
                    f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out | "
                    f"Cost: ${call_cost:.6f}"
                )

            # Parse and return action
            parsed_action = self._parse_action(action_name)
            logger.info(
                f"Agent {obs.agent_id}: LLM chose '{action_name}' -> Action: {parsed_action.name} | "
                f"Obs tokens: {len(obs.tokens)}"
            )
            logger.debug(f"Full action object: {parsed_action}")
            return parsed_action

        except Exception as e:
            logger.error(f"LLM API error: {e}. Falling back to random action.")
            return random.choice(self.policy_env_info.actions.actions())


    def _parse_action(self, action_name: str) -> Action:
        """Parse LLM response and return valid Action.

        Args:
            action_name: Action name from LLM response

        Returns:
            Valid Action object
        """
        # Clean up response
        action_name = action_name.strip().strip('"\'').lower()

        # Find matching action (case-insensitive)
        for action in self.policy_env_info.actions.actions():
            if action.name.lower() == action_name:
                return action

        # If no match, try partial match
        for action in self.policy_env_info.actions.actions():
            if action_name in action.name.lower() or action.name.lower() in action_name:
                return action

        # Fallback to random action if parsing fails
        logger.warning(f"Could not parse action '{action_name}'. Using random action.")
        return random.choice(self.policy_env_info.actions.actions())

    @classmethod
    def get_cost_summary(cls) -> dict:
        """Get summary of API usage and costs.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_calls": cls.total_calls,
            "total_input_tokens": cls.total_input_tokens,
            "total_output_tokens": cls.total_output_tokens,
            "total_tokens": cls.total_input_tokens + cls.total_output_tokens,
            "total_cost": cls.total_cost,
        }

    @classmethod
    def reset_cost_tracking(cls) -> None:
        """Reset cost tracking counters."""
        cls.total_calls = 0
        cls.total_input_tokens = 0
        cls.total_output_tokens = 0
        cls.total_cost = 0.0


class LLMMultiAgentPolicy(MultiAgentPolicy):
    """LLM-based multi-agent policy for MettaGrid."""

    short_names = ["llm"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7
    ):
        """Initialize LLM multi-agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature for LLM
        """
        super().__init__(policy_env_info)
        self.provider: Literal["openai", "anthropic", "ollama"] = provider
        self.model = model
        self.temperature = temperature

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Create an LLM agent policy for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            LLMAgentPolicy instance
        """
        return LLMAgentPolicy(
            self.policy_env_info,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature
        )


class LLMGPTMultiAgentPolicy(LLMMultiAgentPolicy):
    """OpenAI GPT-based policy for MettaGrid."""

    short_names = ["llm-gpt", "llm-openai"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7
    ):
        super().__init__(policy_env_info, provider="openai", model=model, temperature=temperature)


class LLMClaudeMultiAgentPolicy(LLMMultiAgentPolicy):
    """Anthropic Claude-based policy for MettaGrid."""

    short_names = ["llm-claude", "llm-anthropic"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7
    ):
        super().__init__(policy_env_info, provider="anthropic", model=model, temperature=temperature)


class LLMOllamaMultiAgentPolicy(LLMMultiAgentPolicy):
    """Ollama local LLM-based policy for MettaGrid."""

    short_names = ["llm-ollama", "llm-local"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7
    ):
        super().__init__(policy_env_info, provider="ollama", model=model, temperature=temperature)
