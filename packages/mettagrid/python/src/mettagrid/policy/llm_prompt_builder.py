"""Dynamic prompt builder for LLM policy with context window management.

This module provides intelligent prompt building that:
1. Sends basic game rules only once per N steps (context window)
2. Sends only observable changes at each step
3. Manages context windows of configurable size
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


@dataclass
class VisibleElements:
    """Tracks what elements are currently visible in an observation."""

    tags: set[int]  # Object type IDs visible
    features: set[str]  # Feature names visible

    def __eq__(self, other: object) -> bool:
        """Check if visible elements are the same."""
        if not isinstance(other, VisibleElements):
            return False
        return self.tags == other.tags and self.features == other.features


class LLMPromptBuilder:
    """Builds dynamic, context-aware prompts for LLM policy.

    Strategy:
    - N=1 or N%context_window_size==1: Send basic_info + observable
    - N=2 to N=context_window_size: Send only observable (changes)
    - N=context_window_size+1: Resend basic_info + observable (window reset)

    This ensures the LLM always has game rules in context without repeating them every step.
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        context_window_size: int = 20,
    ):
        """Initialize prompt builder.

        Args:
            policy_env_info: Policy environment interface with feature/tag/action specs
            context_window_size: Number of steps before resending basic info (default: 20)
        """
        self._policy_env_info = policy_env_info
        self._context_window_size = context_window_size
        self._step_counter = 0
        self._last_visible: VisibleElements | None = None

        # Pre-build static content (game rules, coordinate system)
        self._basic_info_cache = self._build_basic_info()

    def _build_basic_info(self) -> str:
        """Build basic game information (rules, coordinate system, action reference).

        This is sent:
        - At step 1
        - Every N steps (when context window resets)

        Returns:
            Static game rules and mechanics
        """
        obs_width = self._policy_env_info.obs_width
        obs_height = self._policy_env_info.obs_height
        agent_x = obs_width // 2
        agent_y = obs_height // 2

        # Build action reference
        action_docs = []
        for action_id, action_name in enumerate(self._policy_env_info.action_names):
            action_docs.append(f"  {action_id}: '{action_name}'")

        return f"""You are playing MettaGrid, a multi-agent gridworld game.

=== COORDINATE SYSTEM ===

- Observation window is {obs_width}x{obs_height} grid
- YOU (the agent) are always at the CENTER: x={agent_x}, y={agent_y}
- Coordinates are EGOCENTRIC (relative to you)
- x=0 is West edge, x={obs_width - 1} is East edge
- y=0 is North edge, y={obs_height - 1} is South edge

CARDINAL DIRECTIONS FROM YOUR POSITION:
- North: x={agent_x}, y={agent_y - 1}
- South: x={agent_x}, y={agent_y + 1}
- East: x={agent_x + 1}, y={agent_y}
- West: x={agent_x - 1}, y={agent_y}

=== OBSERVATION FORMAT ===

You receive observations as a list of tokens. Each token has:
- "feature": Feature name (what property it describes)
- "location": {{"x": col, "y": row}} coordinates
- "value": Feature value

UNDERSTANDING TOKENS:
1. Tokens at YOUR location (x={agent_x}, y={agent_y}) describe YOUR state (inventory, frozen status, etc.)
2. Tokens at OTHER locations describe objects/agents you can see
3. Multiple tokens at the SAME location = same object with multiple properties
4. "tag" feature tells you what type of object it is

=== CORE GAME MECHANICS ===

MOVEMENT LOGIC (CRITICAL):
- A tile is WALKABLE if it has NO tokens at that location
- A tile is BLOCKED if it has ANY tokens (wall, object, agent, etc.)
- ALWAYS check target location for tokens before moving
- If you see ANY token at target coordinates → DON'T move there
- If you see NO tokens at target coordinates → SAFE to move

OBJECT INTERACTION:
- Move adjacent to objects to interact with them
- Different object types have different uses (extractors, assemblers, chests, etc.)
- Check object's features to understand its state (cooldown, remaining uses, etc.)

AVAILABLE ACTIONS:
{chr(10).join(action_docs)}

=== STRATEGY TIPS ===

1. ALWAYS check target location for tokens before moving
2. Empty locations (no tokens) = walkable
3. Any tokens at location = blocked/occupied
4. Prioritize energy and resource management
5. Use objects strategically based on their cooldowns and requirements
"""

    def basic_info_prompt(self) -> str:
        """Get the basic game information prompt.

        Returns:
            Static game rules and mechanics
        """
        return self._basic_info_cache

    def observable_prompt(self, obs: AgentObservation, include_actions: bool = False) -> str:
        """Build prompt with ONLY currently observable elements.

        This is the "dynamic" part sent at each step, describing:
        - Visible object types (tags)
        - Visible features and their meanings
        - Current observation data

        Args:
            obs: Current agent observation
            include_actions: Whether to include available_actions list (only on first/reset steps)

        Returns:
            Prompt describing only visible elements
        """
        # Extract what's visible
        visible = self._extract_visible_elements(obs)

        sections = []

        # 1. Describe visible object types
        if visible.tags:
            sections.append(self._build_visible_tags_section(visible.tags))

        # 2. Describe visible features
        if visible.features:
            sections.append(self._build_visible_features_section(visible.features))

        # 3. Current observation data
        obs_json = self._observation_to_json(obs, include_actions=include_actions)
        sections.append(f"""=== CURRENT OBSERVATION ===

{json.dumps(obs_json, indent=2)}
""")

        return "\n\n".join(sections)

    def full_prompt(self, obs: AgentObservation) -> str:
        """Build full prompt (basic_info + observable).

        This is used for:
        - Step 1
        - Every N steps (context window reset)
        - Backward compatibility with old static prompt approach

        Args:
            obs: Current agent observation

        Returns:
            Complete prompt with game rules and current observation
        """
        return f"""{self.basic_info_prompt()}

{self.observable_prompt(obs, include_actions=True)}

Based on the visible objects and game rules, choose the BEST action to maximize your rewards.

CRITICAL: Your response must be EXACTLY ONE action name, nothing else.
Format: action_name
Do NOT explain your reasoning.
Do NOT say what actions you won't take.
Do NOT use sentences.

Example valid responses:
move_east
use
noop

Example INVALID responses:
I should not move_north, so I'll move_east (WRONG - contains multiple actions)
The best action is move_east (WRONG - contains extra words)
"""

    def context_prompt(
        self,
        obs: AgentObservation,
        force_basic_info: bool = False,
    ) -> tuple[str, bool]:
        """Build prompt with smart context window management.

        Strategy:
        - First step or every N steps: Send basic_info + observable
        - Other steps: Send only observable (if it changed)

        Args:
            obs: Current agent observation
            force_basic_info: Force sending basic info even if not at window boundary

        Returns:
            Tuple of (prompt, includes_basic_info)
            - prompt: The built prompt
            - includes_basic_info: True if basic_info was included
        """
        self._step_counter += 1

        # Check if we need to resend basic info
        is_first_step = self._step_counter == 1
        is_window_reset = self._step_counter % self._context_window_size == 1
        should_send_basic = is_first_step or is_window_reset or force_basic_info

        if should_send_basic:
            # Reset context window - send full prompt
            prompt = self.full_prompt(obs)
            includes_basic = True
        else:
            # Within context window - send only observable
            prompt = f"""{self.observable_prompt(obs)}

Choose the BEST action. Reply with ONLY the action name, nothing else.
"""
            includes_basic = False

        # Track what we saw this step
        self._last_visible = self._extract_visible_elements(obs)

        return prompt, includes_basic

    def reset_context(self) -> None:
        """Reset the context window counter.

        Call this at the start of a new episode.
        """
        self._step_counter = 0
        self._last_visible = None

    def _extract_visible_elements(self, obs: AgentObservation) -> VisibleElements:
        """Extract unique tags and features from observation.

        Args:
            obs: Agent observation

        Returns:
            VisibleElements with sets of visible tags and features
        """
        visible_tags = set()
        visible_features = set()

        for token in obs.tokens:
            visible_features.add(token.feature.name)
            if token.feature.name == "tag":
                visible_tags.add(token.value)

        return VisibleElements(tags=visible_tags, features=visible_features)

    def _build_visible_tags_section(self, tags: set[int]) -> str:
        """Build section describing visible object types.

        Args:
            tags: Set of visible tag IDs

        Returns:
            Formatted description of visible objects
        """
        lines = [f"=== OBJECTS YOU CAN SEE ({len(tags)} types) ===\n"]

        for tag_id in sorted(tags):
            tag_name = self._policy_env_info.tags[tag_id]
            description = self._get_tag_description(tag_name)
            lines.append(f"Tag {tag_id}: {tag_name}")
            lines.append(f"  → {description}")

        return "\n".join(lines)

    def _build_visible_features_section(self, features: set[str]) -> str:
        """Build section describing visible features.

        Args:
            features: Set of visible feature names

        Returns:
            Formatted description of visible features
        """
        lines = [f"=== FEATURES YOU CAN SEE ({len(features)} features) ===\n"]

        # Group features by type for better organization
        inventory = [f for f in features if f.startswith("inv:")]
        protocol_input = [f for f in features if f.startswith("protocol_input:")]
        protocol_output = [f for f in features if f.startswith("protocol_output:")]
        agent_features = [f for f in features if f.startswith("agent:")]
        other = [
            f for f in features
            if not any(f.startswith(prefix) for prefix in ["inv:", "protocol_input:", "protocol_output:", "agent:"])
        ]

        if inventory:
            lines.append("Your Inventory:")
            for feature in sorted(inventory):
                resource = feature[4:]  # Remove "inv:" prefix
                lines.append(f"  • {feature}: Amount of {resource} you're carrying (0-255)")

        if protocol_input:
            lines.append("\nObject Requirements (what it needs):")
            for feature in sorted(protocol_input):
                resource = feature[15:]  # Remove "protocol_input:" prefix
                lines.append(f"  • {feature}: {resource} required to use this object")

        if protocol_output:
            lines.append("\nObject Outputs (what it produces):")
            for feature in sorted(protocol_output):
                resource = feature[16:]  # Remove "protocol_output:" prefix
                lines.append(f"  • {feature}: {resource} produced when using this object")

        if agent_features:
            lines.append("\nAgent State:")
            for feature in sorted(agent_features):
                description = self._get_feature_description(feature)
                lines.append(f"  • {feature}: {description}")

        if other:
            lines.append("\nOther Features:")
            for feature in sorted(other):
                description = self._get_feature_description(feature)
                lines.append(f"  • {feature}: {description}")

        return "\n".join(lines)

    def _get_tag_description(self, tag_name: str) -> str:
        """Get brief description for an object tag.

        Args:
            tag_name: Name of the tag

        Returns:
            Human-readable description
        """
        descriptions = {
            "agent": "Another agent (check agent:group to see if ally/enemy)",
            "assembler": "Crafts items using protocols. Vibe determines which recipe activates.",
            "carbon_extractor": "Harvests carbon. Check cooldown_remaining to see if ready.",
            "oxygen_extractor": "Harvests oxygen. Check cooldown_remaining to see if ready.",
            "germanium_extractor": "Harvests germanium. Check cooldown_remaining to see if ready.",
            "silicon_extractor": "Harvests silicon. Check cooldown_remaining to see if ready.",
            "charger": "Restores energy. Move into it to charge.",
            "chest": "Stores resources. Vibe determines deposit (positive) vs withdraw (negative).",
            "wall": "Impassable obstacle. Cannot move through walls.",
            "altar": "Use energy here to gain rewards (costs energy, has cooldown)",
            "converter": "Convert resources to energy (no energy cost, has cooldown)",
            "generator": "Harvest resources from here (has cooldown)",
        }
        return descriptions.get(tag_name, f"Unknown object type: {tag_name}")

    def _get_feature_description(self, feature_name: str) -> str:
        """Get brief description for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Human-readable description
        """
        descriptions = {
            "tag": "Object type ID (see OBJECTS YOU CAN SEE above)",
            "cooldown_remaining": "Steps until object is ready (0 = ready now)",
            "remaining_uses": "Times object can still be used (0 = depleted)",
            "clipped": "Whether object is clipped (0 = normal, 1 = clipped)",
            "agent:group": "Team ID (same as yours = ally, different = enemy)",
            "agent:frozen": "Whether agent can act (0 = can act, 1 = frozen)",
            "vibe": "Current interaction mode (affects object interactions)",
            "agent:compass": "Direction to objective (0=N, 1=E, 2=S, 3=W)",
            "last_action": "Your previous action ID",
            "last_reward": "Reward from last step (positive = good)",
            "episode_completion_pct": "Progress through episode (0-255)",
        }
        return descriptions.get(feature_name, f"{feature_name} feature")

    def _observation_to_json(self, obs: AgentObservation, include_actions: bool = False) -> dict:
        """Convert observation to JSON format for LLM.

        Args:
            obs: Agent observation
            include_actions: Whether to include available_actions list (only on first/reset steps)

        Returns:
            Dictionary with structured observation data
        """
        tokens_list = []
        for token in obs.tokens:
            token_dict = {
                "feature": token.feature.name,
                "location": {"x": token.col(), "y": token.row()},
                "value": token.value,
            }
            tokens_list.append(token_dict)

        result = {
            "agent_id": obs.agent_id,
            "visible_objects": tokens_list,
            "num_visible_objects": len(tokens_list),
        }

        # Only include actions list on first step or context window reset
        if include_actions:
            result["available_actions"] = self._policy_env_info.action_names

        return result

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_counter

    @property
    def context_window_size(self) -> int:
        """Get context window size."""
        return self._context_window_size
