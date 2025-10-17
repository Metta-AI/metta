"""Claude API integration for playing MettaGrid.

Uses Anthropic's Claude API to select actions based on game observations.
"""

import os
import re
from pathlib import Path
from typing import Optional

import anthropic


def load_env_file():
    """Load environment variables from .env file.

    Checks in order:
    1. Repository root (.env)
    2. Current directory (.env)
    3. Script directory (.env)
    """
    # Try to find repo root by looking for .git directory
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            repo_root = current
            break
        current = current.parent
    else:
        # Fallback: assume we're in packages/mettagrid/examples/claude_client
        repo_root = Path(__file__).parent.parent.parent.parent.parent

    # Check multiple locations in order of preference
    env_locations = [
        repo_root / ".env",  # Repo root first
        Path.cwd() / ".env",  # Current directory
        Path(__file__).parent / ".env",  # Script directory
    ]

    for env_path in env_locations:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, _, value = line.partition("=")
                        # Only set if not already in environment
                        env_key = key.strip()
                        if env_key not in os.environ:
                            os.environ[env_key] = value.strip().strip('"').strip("'")
            break  # Stop after loading first .env file found


# Try to load .env file on import
load_env_file()


class ObservationDecoder:
    """Decodes raw observation bytes into human-readable descriptions."""

    @staticmethod
    def decode_observations(
        obs_bytes: bytes,
        num_agents: int,
        tokens_per_agent: int,
        components_per_token: int = 3,
    ):
        """Decode observation buffer into structured data."""
        import numpy as np

        # Convert bytes to numpy array
        obs_array = np.frombuffer(obs_bytes, dtype=np.uint8)

        # Reshape to [num_agents, tokens_per_agent, components_per_token]
        expected_size = num_agents * tokens_per_agent * components_per_token
        if len(obs_array) != expected_size:
            print(f"Warning: Expected {expected_size} bytes, got {len(obs_array)}")
            return None

        obs_array = obs_array.reshape(
            (num_agents, tokens_per_agent, components_per_token)
        )

        return obs_array

    @staticmethod
    def describe_observation(obs_array, agent_idx: int = 0) -> str:
        """Create a text description of what the agent observes."""
        if obs_array is None:
            return "No observation data available"

        agent_obs = obs_array[agent_idx]

        # Count different types of cells in observation
        # Each token is [type_id, feature1, feature2]
        type_ids = agent_obs[:, 0]

        empty_cells = int((type_ids == 0).sum())
        wall_cells = int((type_ids == 2).sum())
        agent_cells = int((type_ids == 1).sum())
        total_cells = len(type_ids)

        description = f"""Observation Window ({total_cells} cells):
- Empty spaces: {empty_cells}
- Walls: {wall_cells}
- Agents: {agent_cells}

Interpretation:
"""

        if wall_cells > total_cells * 0.7:
            description += "- You are surrounded by walls or very close to walls\n"
        elif wall_cells > total_cells * 0.4:
            description += "- There are walls nearby\n"
        else:
            description += "- You have open space around you\n"

        if agent_cells > 1:
            description += f"- You see {agent_cells - 1} other agent(s) nearby\n"

        return description


class ClaudePlayer:
    """Uses Claude to play MettaGrid by analyzing observations and selecting actions."""

    # Action mapping for better descriptions
    ACTION_MAP = {
        0: ("noop", "Do nothing, stay in place"),
        1: ("move_forward", "Move one step forward in your current direction"),
        2: ("move_backward", "Move one step backward from your current direction"),
        3: ("move_left", "Move one step to the left"),
        4: ("move_right", "Move one step to the right"),
        5: ("rotate_left", "Turn 90° left (counterclockwise) without moving"),
        6: ("rotate_right", "Turn 90° right (clockwise) without moving"),
    }

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.decoder = ObservationDecoder()

        # Build action description
        action_list = "\n".join(
            [
                f"- {idx}: {name} - {desc}"
                for idx, (name, desc) in self.ACTION_MAP.items()
            ]
        )

        # System message with comprehensive game rules
        self.system_message = f"""You are playing MettaGrid, a grid-based navigation and exploration game.

GAME WORLD:
- You control an agent on a 2D grid world
- The world contains walls (impassable) and open spaces
- You have a limited observation window showing nearby cells
- The grid wraps around at the edges (toroidal topology)

YOUR AGENT:
- Faces a direction (North, South, East, or West)
- Can move forward, backward, left, or right
- Can rotate to change facing direction
- Has limited vision (5x5 observation window)

GOALS:
- Explore the environment efficiently
- Avoid running into walls
- Collect positive rewards where possible
- Complete the episode within the step limit

AVAILABLE ACTIONS:
{action_list}

STRATEGY TIPS:
1. If you see many walls, try rotating or moving sideways to find open space
2. Moving forward in your facing direction is usually more efficient
3. Rotating is free (no movement), use it to reorient
4. If you're stuck or near walls, rotate to find a better path
5. Don't waste actions with noop unless waiting is strategic

RESPONSE FORMAT:
Respond with ONLY the action index number (0-6).
Do not include any explanation or reasoning in your response."""

    def choose_action(
        self,
        step_num: int,
        total_steps: int,
        obs_bytes: bytes,
        num_agents: int,
        tokens_per_agent: int,
        last_reward: float = 0.0,
        agent_idx: int = 0,
    ) -> int:
        """Ask Claude to choose an action based on the current game state."""

        # Decode observations
        obs_array = self.decoder.decode_observations(
            obs_bytes, num_agents, tokens_per_agent
        )
        obs_description = self.decoder.describe_observation(obs_array, agent_idx)

        # Calculate progress percentage
        progress_pct = (step_num / total_steps) * 100

        # Create detailed prompt
        prompt = f"""CURRENT GAME STATE:

Step: {step_num}/{total_steps} ({progress_pct:.1f}% complete)
Last Reward: {last_reward:+.3f}

{obs_description}

What action should you take? Respond with just the number (0-6)."""

        try:
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                system=self.system_message,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            # Extract action from response
            response_text = message.content[0].text.strip()

            # Try to parse the action number
            match = re.search(r"\b([0-6])\b", response_text)
            if match:
                action = int(match.group(1))
                action_name = self.ACTION_MAP[action][0]
                print(f"  Claude chose: {action} ({action_name})")
                return action
            else:
                print(
                    f"Warning: Could not parse action from Claude response: '{response_text}'"
                )
                print("Defaulting to noop (0)")
                return 0

        except Exception as e:
            print(f"Error calling Claude API: {e}")
            print("Defaulting to noop (0)")
            return 0
