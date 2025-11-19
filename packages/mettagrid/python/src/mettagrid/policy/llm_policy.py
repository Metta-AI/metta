"""LLM-based policy for MettaGrid using GPT or Claude."""

import json
import os
import random
from typing import Literal

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

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

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str | None = None,
        temperature: float = 0.7
    ):
        """Initialize LLM agent policy.

    Args:
        policy_env_info: Policy environment interface
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (defaults to gpt-4o-mini or claude-3-5-sonnet)
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
            self.model = model or "gpt-4o-mini"
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = None
            self.anthropic_client: Anthropic | None = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-3-5-sonnet-20241022"
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

            # Parse and return action
            return self._parse_action(action_name)

        except Exception as e:
            print(f"LLM API error: {e}. Falling back to random action.")
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
        print(f"Warning: Could not parse action '{action_name}'. Using random action.")
        return random.choice(self.policy_env_info.actions.actions())


class LLMMultiAgentPolicy(MultiAgentPolicy):
    """LLM-based multi-agent policy for MettaGrid."""

    short_names = ["llm", "llm-gpt", "llm-claude"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str | None = None,
        temperature: float = 0.7
    ):
        """Initialize LLM multi-agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai" or "anthropic")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature for LLM
        """
        super().__init__(policy_env_info)
        self.provider: Literal["openai", "anthropic"] = provider
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
