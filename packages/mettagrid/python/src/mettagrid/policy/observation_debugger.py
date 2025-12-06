"""Human-readable observation debugger for LLM policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface
    from mettagrid.simulator.interface import AgentObservation


class ObservationDebugger:
    """Converts agent observations to human-readable debug information."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        """Initialize observation debugger.

        Args:
            policy_env_info: Policy environment interface with feature specs
        """
        self.policy_env_info = policy_env_info
        self.obs_width = policy_env_info.obs_width
        self.obs_height = policy_env_info.obs_height
        # Agent SHOULD be at center of observation grid in egocentric view
        self.expected_agent_x = self.obs_width // 2
        self.expected_agent_y = self.obs_height // 2

    def debug_observation(self, obs: AgentObservation, last_action: str | None = None) -> str:
        """Convert observation to human-readable debug string.

        Args:
            obs: Agent observation
            last_action: Last action taken (optional)

        Returns:
            Human-readable debug string
        """
        lines = ["=" * 70, f"AGENT {obs.agent_id} OBSERVATION DEBUG", "=" * 70]

        # Find actual agent position in observation
        agent_x, agent_y = self._find_agent_position(obs)

        # Parse observation tokens
        spatial_grid = self._build_spatial_grid(obs)
        agent_state = self._extract_agent_state(obs, agent_x, agent_y)
        directional_info = self._analyze_directions( spatial_grid, agent_x, agent_y)

        # Agent state section
        lines.append("\n📊 AGENT STATE:")
        if last_action:
            lines.append(f"  Last Action: {last_action}")
        for key, value in agent_state.items():
            lines.append(f"  {key}: {value}")

        # Directional awareness section
        lines.append("\n🧭 DIRECTIONAL AWARENESS:")
        for direction, info in directional_info.items():
            lines.append(f"  {direction}: {info}")

        # Spatial grid visualization
        lines.append("\n🗺️  SPATIAL GRID (what agent sees):")
        lines.append(self._visualize_grid(spatial_grid, agent_x, agent_y))

        # Inventory section
        inventory = self._extract_inventory(obs)
        if inventory:
            lines.append("\n🎲 INVENTORY:")
            for resource, amount in inventory.items():
                lines.append(f"  {resource}: {amount}")

        # Raw token count
        lines.append(f"\n📝 Total observation tokens: {len(obs.tokens)}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _build_spatial_grid(self, obs: AgentObservation) -> dict[tuple[int, int], list[dict]]:
        """Build spatial grid from observation tokens.

        Args:
            obs: Agent observation

        Returns:
            Dictionary mapping (x, y) to list of token info at that location
        """
        grid: dict[tuple[int, int], list[dict]] = {}

        for token in obs.tokens:
            x, y = token.row(), token.col()
            if (x, y) not in grid:
                grid[(x, y)] = []

            # Get tag name if this is a tag feature
            tag_name = None
            if token.feature.name == "tag" and token.value < len(self.policy_env_info.tags):
                tag_name = self.policy_env_info.tags[token.value]

            grid[(x, y)].append({
                "feature": token.feature.name,
                "value": token.value,
                "tag": tag_name,
            })

        return grid

    def _find_agent_position(self, obs: AgentObservation) -> tuple[int, int]:
        """Find the agent's actual position in the observation.

        The agent should be at the center in egocentric view. We verify
        by looking for the "agent" tag token.

        Args:
            obs: Agent observation

        Returns:
            Tuple of (x, y) coordinates of agent position
        """
        # Look for "agent" tag in observation
        for token in obs.tokens:
            if token.feature.name == "tag":
                tag_name = self.policy_env_info.tags[token.value] if token.value < len(self.policy_env_info.tags) else None
                if tag_name == "agent":
                    return token.row(), token.col()

        # If no agent tag found, assume center (egocentric view default)
        return self.expected_agent_x, self.expected_agent_y

    def _extract_agent_state(self, obs: AgentObservation, agent_x: int, agent_y: int) -> dict[str, str]:
        """Extract agent state information from observation.

        Args:
            obs: Agent observation
            agent_x: Agent's X coordinate
            agent_y: Agent's Y coordinate

        Returns:
            Dictionary of agent state information
        """
        state = {}

        for token in obs.tokens:
            # Look for agent-specific features at agent's location
            if token.row() == agent_x and token.col() == agent_y:
                if token.feature.name == "agent:frozen":
                    state["Frozen"] = "Yes" if token.value > 0 else "No"
                elif token.feature.name == "agent:group":
                    state["Group"] = str(token.value)
                elif token.feature.name == "vibe":
                    state["Vibe"] = str(token.value)
                elif token.feature.name == "agent:compass":
                    compass_dirs = ["North", "East", "South", "West"]
                    if 0 <= token.value < len(compass_dirs):
                        state["Compass Direction"] = compass_dirs[token.value]

            # Global features (not location-specific)
            if token.feature.name == "episode_completion_pct":
                state["Episode Progress"] = f"{token.value}%"
            elif token.feature.name == "last_action":
                if token.value < len(self.policy_env_info.action_names):
                    state["Last Action ID"] = self.policy_env_info.action_names[token.value]
            elif token.feature.name == "last_reward":
                state["Last Reward"] = str(token.value)

        return state

    def _analyze_directions(self,  spatial_grid: dict[tuple[int, int], list[dict]], agent_x: int, agent_y: int) -> dict[str, str]:
        """Analyze what's in each cardinal direction.

        Args:
            spatial_grid: Spatial grid of tokens
            agent_x: Agent's X coordinate
            agent_y: Agent's Y coordinate

        Returns:
            Dictionary mapping directions to descriptions
        """
        directions = {}

        # Check cardinal directions (North, South, East, West)
        checks = [
            ("North", 0, -1),
            ("South", 0, 1),
            ("East", 1, 0),
            ("West", -1, 0),
        ]

        for dir_name, dx, dy in checks:
            target_x = agent_x + dx
            target_y = agent_y + dy

            if (target_x, target_y) in spatial_grid:
                tokens_at_pos = spatial_grid[(target_x, target_y)]
                # Look for tag to identify object
                for token_info in tokens_at_pos:
                    if token_info["tag"]:
                        directions[dir_name] = f"{token_info['tag']} (BLOCKED)"
                        break
                else:
                    directions[dir_name] = "Empty space"
            else:
                directions[dir_name] = "Clear"

        return directions

    def _get_direction_description(self, dx: int, dy: int) -> str:
        """Get human-readable direction description.

        Args:
            dx: X offset from agent
            dy: Y offset from agent

        Returns:
            Direction description
        """
        if dx == 0 and dy == 0:
            return "same position"

        vertical = ""
        horizontal = ""

        if dy < 0:
            vertical = "North"
        elif dy > 0:
            vertical = "South"

        if dx < 0:
            horizontal = "West"
        elif dx > 0:
            horizontal = "East"

        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        return vertical or horizontal

    def _visualize_grid(self, spatial_grid: dict[tuple[int, int], list[dict]], agent_x: int, agent_y: int) -> str:
        """Create ASCII visualization of spatial grid.

        Args:
            spatial_grid: Spatial grid of tokens (keys are (row, col) from token.row(), token.col())
            agent_x: Agent's row coordinate (from token.row())
            agent_y: Agent's col coordinate (from token.col())

        Returns:
            ASCII grid visualization
        """
        lines = []
        # Column headers
        lines.append("  " + "".join([f"{col:2}" for col in range(self.obs_width)]))

        # Grid rows - spatial_grid keys are (row, col)
        for row in range(self.obs_height):
            row_str = f"{row:2} "
            for col in range(self.obs_width):
                if row == agent_x and col == agent_y:
                    row_str += "@ "  # Agent position
                elif (row, col) in spatial_grid:
                    # Find tag to display
                    tag_found = False
                    for token_info in spatial_grid[(row, col)]:
                        if token_info["tag"]:
                            # Use first letter of tag
                            row_str += token_info["tag"][0].upper() + " "
                            tag_found = True
                            break
                    if not tag_found:
                        row_str += "· "  # Something there but no tag
                else:
                    row_str += ". "  # Empty
            lines.append(row_str)

        lines.append("\n  Legend: @ = Agent, . = Empty, · = Unknown, Letters = Object tags")
        return "\n".join(lines)

    def _extract_inventory(self, obs: AgentObservation) -> dict[str, int]:
        """Extract inventory information from observation.

        Args:
            obs: Agent observation

        Returns:
            Dictionary mapping resource names to amounts
        """
        inventory = {}

        for token in obs.tokens:
            if token.feature.name.startswith("inv:"):
                resource_name = token.feature.name[4:]
                if token.value > 0:
                    inventory[resource_name] = token.value

        return inventory
