"""Dynamic prompt builder for LLM policy with context window management.

This module provides intelligent prompt building that:
1. Sends basic game rules only once per N steps (context window)
2. Sends only observable changes at each step
3. Manages context windows of configurable size

Prompt templates are loaded from markdown files in the 'prompts/' directory:
- basic_info.md: Game rules and instructions
- full_prompt.md: Complete prompt with decision logic
- dynamic_prompt.md: Shorter prompt for within-window steps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llm_agent.utils import pos_to_dir
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig

# Load prompt templates from markdown files
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        name: Template filename (without .md extension)

    Returns:
        Template content as string
    """
    template_path = _PROMPTS_DIR / f"{name}.md"
    if template_path.exists():
        return template_path.read_text()
    else:
        raise FileNotFoundError(f"Prompt template not found: {template_path}")


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
            mg_cfg: MettaGridConfig | None = None,
            debug_mode: bool = False,
            agent_id: int = 0,
            verbose: bool = False,
    ):
        """Initialize prompt builder.

        Args:
            policy_env_info: Policy environment interface with feature/tag/action specs
            context_window_size: Number of steps before resending basic info (default: 20)
            mg_cfg: Optional MettaGridConfig to extract chest vibe transfers and other game-specific info
            debug_mode: If True, print debug information (default: False)
            agent_id: Agent ID for role assignment (default: 0)
            verbose: Alias for debug_mode (default: False)
        """
        self._policy_env_info = policy_env_info
        # Ensure context_window_size is an int (may come as string from config)
        self._context_window_size = int(context_window_size)
        self._step_counter = 0
        self._last_visible: VisibleElements | None = None
        self._debug_mode = debug_mode or verbose
        self._agent_id = agent_id

        # Store mg_cfg for id_map access
        self._mg_cfg = mg_cfg

        # Extract chest vibe transfers if config is provided
        self._chest_vibe_transfers: dict[str, dict[str, int]] = {}
        if mg_cfg is not None:
            chest_config = mg_cfg.game.objects.get("chest")
            if chest_config and hasattr(chest_config, "vibe_transfers"):
                self._chest_vibe_transfers = chest_config.vibe_transfers
                if self._debug_mode:
                    print(f"[LLMPromptBuilder] Loaded chest vibe transfers: {self._chest_vibe_transfers}")
            else:
                if self._debug_mode:
                    print(
                        f"[LLMPromptBuilder] No chest vibe transfers found. chest_config={chest_config}, has vibe_transfers={hasattr(chest_config, 'vibe_transfers') if chest_config else 'N/A'}")
        else:
            if self._debug_mode:
                print("[LLMPromptBuilder] No mg_cfg provided, chest vibe transfers will be empty")

        # Pre-build static content (game rules, coordinate system)
        self._basic_info_cache = self._build_basic_info()

        # Debug: print the recipes being used
        if self._debug_mode:
            recipes = self._build_all_recipes()
            print(f"[LLMPromptBuilder] Recipes:\n{recipes}")

    @staticmethod
    def _get_role_assignment() -> str:
        """Get role assignment based on agent ID.

        Returns:
            Role description string for this agent
        """
        # Generic role for all agents - gather ALL resources needed for hearts
        return (
            "ROLE: Resource gatherer\n"
            "- Find and use ALL extractors to collect resources\n"
            "- Check RECIPE above for exact amounts needed\n"
            "- Craft at ASSEMBLER when you have all resources"
        )

    def _build_basic_info(self) -> str:
        """Build minimal game information prompt.

        This is sent:
        - At step 1
        - Every N steps (when context window resets)

        Returns:
            Minimal game rules - only what agent cannot discover
        """
        # Build dynamic sections
        all_recipes_section = self._build_all_recipes()
        id_map_section = self._build_id_map_section()

        # Load template and substitute variables
        template = _load_prompt_template("basic_info")
        return template.replace("{{RECIPES}}", all_recipes_section).replace("{{ID_MAP}}", id_map_section)

    def _build_recipe_summary(self) -> str:
        """Build a short recipe summary for dynamic prompts.

        Returns:
            Condensed recipe string like "HEART needs: 1 carbon, 1 oxygen, 1 germanium, 1 silicon, 1 energy"
        """
        if not self._policy_env_info.assembler_protocols:
            return "HEART needs: 10 carbon, 10 oxygen, 2 germanium, 30 silicon"

        # Find the first heart protocol (1 heart output)
        for protocol in self._policy_env_info.assembler_protocols:
            if protocol.output_resources.get("heart", 0) == 1:
                parts = []
                for resource in ["carbon", "oxygen", "germanium", "silicon", "energy"]:
                    amount = protocol.input_resources.get(resource, 0)
                    if amount > 0:
                        parts.append(f"{amount} {resource}")
                if parts:
                    return "HEART needs: " + ", ".join(parts)

        # Fallback
        return "HEART needs: 10 carbon, 10 oxygen, 2 germanium, 30 silicon"

    def _build_all_recipes(self) -> str:
        """Build all assembler recipes/protocols.

        Returns:
            Human-readable list of all crafting recipes with vibe requirements
        """
        if not self._policy_env_info.assembler_protocols:
            return ""

        lines = ["=== CRAFTING RECIPES ==="]
        lines.append("Use these vibes at an ASSEMBLER to craft items:")
        lines.append("")

        # Group by output for clarity
        seen_recipes: set[str] = set()

        for protocol in self._policy_env_info.assembler_protocols:
            # Format inputs
            input_parts = []
            for resource in ["carbon", "oxygen", "germanium", "silicon", "energy", "heart", "decoder", "modulator",
                             "resonator", "scrambler"]:
                amount = protocol.input_resources.get(resource, 0)
                if amount > 0:
                    input_parts.append(f"{amount} {resource}")

            # Format outputs
            output_parts = []
            for resource, amount in protocol.output_resources.items():
                if amount > 0:
                    output_parts.append(f"{amount} {resource}")

            # Format vibes
            if protocol.vibes:
                if len(protocol.vibes) == 1:
                    vibe_str = f"vibe: {protocol.vibes[0]}"
                else:
                    # Multiple vibes means multiple agents needed
                    vibe_str = f"vibes: {' + '.join(protocol.vibes)} ({len(protocol.vibes)} agents)"
            else:
                vibe_str = "vibe: any"

            # Create recipe string
            recipe = f"  {', '.join(output_parts)} <- {', '.join(input_parts)} [{vibe_str}]"

            # Avoid duplicates
            if recipe not in seen_recipes:
                seen_recipes.add(recipe)
                lines.append(recipe)

        # Add instructions for multi-agent crafting
        lines.append("")
        lines.append("MULTI-AGENT CRAFTING:")
        lines.append("  When a recipe shows multiple vibes (e.g., 'heart_a + heart_a'), multiple agents")
        lines.append("  must stand ADJACENT to the same assembler, each with their required vibe set.")
        lines.append("  Resources are pooled from all adjacent agents. Any agent can trigger the craft.")
        lines.append("")
        lines.append("GEAR ITEMS (decoder, modulator, scrambler, resonator):")
        lines.append("  These are advanced items that require 2-agent coordination.")
        lines.append("  One agent sets vibe 'gear', the other sets the resource vibe (e.g., 'carbon_a').")
        lines.append("  Both must be adjacent to the assembler, then either triggers the craft.")

        return "\n".join(lines)

    def _build_id_map_section(self) -> str:
        """Build id_map reference section with object tags.

        Returns:
            Formatted section with object types available in this mission
        """
        if self._mg_cfg is None:
            return ""

        try:
            id_map = self._mg_cfg.game.id_map()
            lines = []

            # Tags section - what objects exist in this mission
            lines.append("=== OBJECT TYPES ===")
            lines.append("Objects you may encounter:")
            for tag in id_map.tag_names():
                lines.append(f"  - {tag}")

            return "\n".join(lines)
        except Exception as e:
            if self._debug_mode:
                print(f"[LLMPromptBuilder] Could not build id_map section: {e}")
            return ""

    def basic_info_prompt(self) -> str:
        """Get the basic game information prompt.

        Returns:
            Static game rules and mechanics
        """
        return self._basic_info_cache

    def observable_prompt(self, obs: AgentObservation, include_actions: bool = False) -> str:
        """Build prompt with ONLY currently observable elements.

        This is the "dynamic" part sent at each step, describing:
        - Directional awareness (what's adjacent)
        - Agent's inventory
        - Nearby agents with their inventories

        Args:
            obs: Current agent observation
            include_actions: Whether to include available_actions list (only on first/reset steps)

        Returns:
            Prompt describing only visible elements
        """
        agent_x = self._policy_env_info.obs_width // 2
        agent_y = self._policy_env_info.obs_height // 2

        sections = []

        # 1. Directional awareness - what's immediately adjacent (CRITICAL for avoiding walls)
        directions = self._analyze_adjacent_tiles(obs, agent_x, agent_y)
        sections.append(self._build_directional_awareness_section(directions))

        # 3. Agent's inventory summary (extracted from tokens at agent position)
        inventory = self._extract_inventory(obs, agent_x, agent_y)
        if inventory:
            sections.append(self._build_inventory_section(inventory))

        # 4. Visible objects with coordinates (spatial awareness)
        visible_objects = self._extract_visible_objects_with_coords(obs, agent_x, agent_y)
        if visible_objects:
            sections.append(self._build_visible_objects_section(visible_objects))

        # 5. Include available actions only on first/reset steps
        if include_actions:
            # Only include essential actions, not all 156 vibes
            essential_actions = self._get_essential_actions()
            sections.append(f"=== AVAILABLE ACTIONS ===\n{', '.join(essential_actions)}")

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
        # Load template and substitute variables
        template = _load_prompt_template("full_prompt")
        return (
            template
            .replace("{{BASIC_INFO}}", self.basic_info_prompt())
            .replace("{{OBSERVABLE}}", self.observable_prompt(obs, include_actions=True))
            .replace("{{AGENT_ID}}", str(self._agent_id))
            .replace("{{ROLE_ASSIGNMENT}}", self._get_role_assignment())
        )

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
            # Build common actions list
            common_actions = ["noop", "move_north", "move_south", "move_west", "move_east"]
            # Add vibe actions for resources/hearts
            vibe_actions = [name for name in self._policy_env_info.action_names if
                            name.startswith("change_vibe_") and any(
                                x in name for x in ["heart", "carbon", "oxygen", "silicon", "germanium", "default"])]
            action_list = common_actions + vibe_actions[:10]  # Limit to top 10 vibe actions

            # Load template and substitute variables
            template = _load_prompt_template("dynamic_prompt")
            prompt = (
                template
                .replace("{{OBSERVABLE}}", self.observable_prompt(obs))
                .replace("{{ACTIONS}}", ", ".join(action_list[:8]) + ", ...")
                .replace("{{AGENT_ID}}", str(self._agent_id))
                .replace("{{ROLE_ASSIGNMENT}}", self._get_role_assignment())
                .replace("{{RECIPE_SUMMARY}}", self._build_recipe_summary())
            )
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

    @staticmethod
    def _extract_visible_elements(obs: AgentObservation) -> VisibleElements:
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

    def _analyze_adjacent_tiles(self, obs: AgentObservation, agent_x: int, agent_y: int) -> dict[str, str]:
        """Analyze what's in each cardinal direction (immediately adjacent).

        Args:
            obs: Agent observation
            agent_x: Agent's X coordinate
            agent_y: Agent's Y coordinate

        Returns:
            Dictionary mapping directions to descriptions (e.g., {"North": "Clear", "South": "wall (BLOCKED)"})
        """
        # Build spatial grid from tokens
        spatial_grid: dict[tuple[int, int], list[dict]] = {}
        for token in obs.tokens:
            # SWAPPED: In MettaGrid, row() = X (East/West), col() = Y (North/South)
            x, y = token.row(), token.col()
            if (x, y) not in spatial_grid:
                spatial_grid[(x, y)] = []

            tag_name = None
            if token.feature.name == "tag" and token.value < len(self._policy_env_info.tags):
                tag_name = self._policy_env_info.tags[token.value]

            spatial_grid[(x, y)].append({
                "feature": token.feature.name,
                "value": token.value,
                "tag": tag_name,
            })

        directions = {}
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
                    directions[dir_name] = "BLOCKED"
            else:
                directions[dir_name] = "Clear"

        return directions

    def _build_directional_awareness_section(self, directions: dict[str, str]) -> str:
        """Build the directional awareness section for the prompt.

        Args:
            directions: Dictionary mapping directions to descriptions

        Returns:
            Formatted directional awareness section
        """
        lines = ["=== ADJACENT TILES (can you move there?) ==="]
        for direction in ["North", "South", "East", "West"]:
            status = directions.get(direction, "Clear")
            if "BLOCKED" in status:
                lines.append(f"  {direction}: {status} - DO NOT MOVE HERE")
            else:
                lines.append(f"  {direction}: {status} - can move")
        return "\n".join(lines)

    def _get_direction_name(self, dx: int, dy: int) -> str:
        """Get human-readable direction name from offset.

        Args:
            dx: X offset from agent
            dy: Y offset from agent

        Returns:
            Direction name (e.g., "North", "South-East")
        """
        if dx == 0 and dy == 0:
            return "here"

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

    def _extract_inventory(self, obs: AgentObservation, agent_x: int, agent_y: int) -> dict[str, int]:
        """Extract inventory from tokens at agent's position.

        Args:
            obs: Agent observation
            agent_x: Agent's X coordinate
            agent_y: Agent's Y coordinate

        Returns:
            Dictionary of inventory items and their values
        """
        inventory = {}
        for token in obs.tokens:
            # SWAPPED: row() = X, col() = Y
            if token.row() == agent_x and token.col() == agent_y:
                if token.feature.name.startswith("inv:"):
                    resource = token.feature.name[4:]  # Remove "inv:" prefix
                    inventory[resource] = token.value
        return inventory

    def _build_inventory_section(self, inventory: dict[str, int]) -> str:
        """Build the inventory section for the prompt.

        Args:
            inventory: Dictionary of inventory items

        Returns:
            Formatted inventory section
        """
        lines = ["=== YOUR INVENTORY ==="]
        for resource, amount in sorted(inventory.items()):
            lines.append(f"  {resource}: {amount}")
        return "\n".join(lines)

    def _get_essential_actions(self) -> list[str]:
        """Get list of essential actions (movement + key vibes).

        Returns:
            List of essential action names
        """
        essential = ["noop", "move_north", "move_south", "move_east", "move_west"]
        # Add resource-related vibes
        for vibe in ["heart_a", "heart_b", "carbon_a", "carbon_b", "oxygen_a", "oxygen_b",
                     "germanium_a", "germanium_b", "silicon_a", "silicon_b", "gear", "default"]:
            action_name = f"change_vibe_{vibe}"
            if action_name in self._policy_env_info.action_names:
                essential.append(action_name)
        return essential

    def _extract_visible_objects_with_coords(
            self, obs: AgentObservation, agent_x: int, agent_y: int
    ) -> list[dict]:
        """Extract all visible objects with their absolute coordinates and properties.

        Args:
            obs: Agent observation
            agent_x: Agent's X coordinate (center of view)
            agent_y: Agent's Y coordinate (center of view)

        Returns:
            List of object dicts with name, x, y, and properties
        """
        # First pass: collect all tokens by position
        positions: dict[tuple[int, int], dict] = {}

        for token in obs.tokens:
            # SWAPPED: In MettaGrid, row() = X (East/West), col() = Y (North/South)
            x, y = token.row(), token.col()

            # Skip agent's own position (we show inventory separately)
            if x == agent_x and y == agent_y:
                continue

            if (x, y) not in positions:
                positions[(x, y)] = {
                    "tag": None,
                    "properties": {},
                    "inventory": {},
                }

            pos_data = positions[(x, y)]

            # Capture tag (object type)
            if token.feature.name == "tag" and token.value < len(self._policy_env_info.tags):
                pos_data["tag"] = self._policy_env_info.tags[token.value]

            # Capture inventory for agents
            elif token.feature.name.startswith("inv:") and token.value > 0:
                resource = token.feature.name[4:]  # Remove "inv:" prefix
                pos_data["inventory"][resource] = token.value

            # Capture other useful properties
            elif token.feature.name == "cooldown_remaining" and token.value > 0:
                pos_data["properties"]["cooldown"] = token.value
            elif token.feature.name == "remaining_uses":
                pos_data["properties"]["uses"] = token.value
            elif token.feature.name == "agent:group":
                pos_data["properties"]["group"] = token.value

        # Second pass: build object list with coordinates
        objects = []
        for (x, y), data in positions.items():
            if data["tag"] is None:
                continue  # Skip positions without a recognized object

            # Skip walls - they clutter the output
            if data["tag"] == "wall":
                continue

            # Calculate relative coordinates (offset from agent)
            rel_x = x - agent_x
            rel_y = y - agent_y

            obj = {
                "name": data["tag"],
                "x": rel_x,
                "y": rel_y,
                "distance": abs(rel_x) + abs(rel_y),  # Manhattan distance
                "properties": data["properties"],
                "inventory": data["inventory"],
            }
            objects.append(obj)

        # Sort by distance (closest first)
        return sorted(objects, key=lambda o: o["distance"])

    def _build_visible_objects_section(self, objects: list[dict]) -> str:
        """Build section showing visible objects with absolute coordinates.

        Format: "object_name at (X, Y) - DIRECTION (properties)"

        Args:
            objects: List of object dicts from _extract_visible_objects_with_coords

        Returns:
            Formatted section with spatial coordinates
        """
        agent_x = self._policy_env_info.obs_width // 2
        agent_y = self._policy_env_info.obs_height // 2

        lines = [f"=== VISIBLE OBJECTS (you are at {agent_x},{agent_y}) ===",
                 "Grid: 11x11, (0,0)=top-left, x=column(E/W), y=row(N/S)", ""]

        for obj in objects:
            # Calculate absolute position from relative
            abs_x = obj['x'] + agent_x
            abs_y = obj['y'] + agent_y

            # Calculate direction from agent
            direction = self._get_direction_name(obj['x'], obj['y'])

            # Format: "assembler at (7, 5) - EAST"
            line = f"  {obj['name']} at ({abs_x},{abs_y}) - {direction}"

            # Add properties in parentheses
            props = []
            if obj["properties"]:
                for key, val in obj["properties"].items():
                    props.append(f"{key}={val}")

            # Add inventory for agents
            if obj["inventory"]:
                inv_parts = [f"{k}={v}" for k, v in sorted(obj["inventory"].items())]
                props.append(f"inventory: {', '.join(inv_parts)}")

            if props:
                line += f" ({'; '.join(props)})"

            lines.append(line)

        return "\n".join(lines)

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_counter

    @property
    def context_window_size(self) -> int:
        """Get context window size."""
        return self._context_window_size

    # =========================================================================
    # PATHFINDING (BFS within visible 11x11 grid)
    # =========================================================================

    def _build_wall_map(self, obs: AgentObservation) -> set[tuple[int, int]]:
        """Build a set of blocked tile positions from observation.

        Args:
            obs: Agent observation

        Returns:
            Set of (x, y) positions that are blocked (walls or objects)
        """
        blocked: set[tuple[int, int]] = set()

        for token in obs.tokens:
            if token.feature.name == "tag" and token.value < len(self._policy_env_info.tags):
                tag_name = self._policy_env_info.tags[token.value]
                # Walls are always blocked
                if tag_name == "wall":
                    blocked.add((token.row(), token.col()))
                # Other objects block movement but are valid destinations
                # We'll handle this in BFS by allowing the target tile

        return blocked

    @staticmethod
    def _bfs_first_move(
            start: tuple[int, int],
        target: tuple[int, int],
        blocked: set[tuple[int, int]],
        grid_width: int,
        grid_height: int,
    ) -> str | None:
        """Run BFS to find shortest path and return the first move direction.

        Args:
            start: Starting position (agent position)
            target: Target position
            blocked: Set of blocked positions (walls)
            grid_width: Width of visible grid
            grid_height: Height of visible grid

        Returns:
            First move direction ("move_north", "move_south", etc.) or None if no path
        """
        from collections import deque

        # If already at target, no move needed
        if start == target:
            return None

        # If target is blocked by wall, no path possible
        if target in blocked:
            return None

        # Direction offsets: (dx, dy, action_name)
        directions = [
            (0, -1, "move_north"),
            (0, 1, "move_south"),
            (1, 0, "move_east"),
            (-1, 0, "move_west"),
        ]

        # BFS
        queue: deque[tuple[tuple[int, int], str | None]] = deque()
        queue.append((start, None))  # (position, first_move)
        visited: set[tuple[int, int]] = {start}

        while queue:
            pos, first_move = queue.popleft()
            x, y = pos

            for dx, dy, action in directions:
                nx, ny = x + dx, y + dy

                # Check bounds
                if not (0 <= nx < grid_width and 0 <= ny < grid_height):
                    continue

                # Check if blocked (but allow target even if it has an object)
                if (nx, ny) in blocked and (nx, ny) != target:
                    continue

                # Check if visited
                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))

                # Track the first move that led here
                new_first_move = first_move if first_move else action

                # Found target!
                if (nx, ny) == target:
                    return new_first_move

                queue.append(((nx, ny), new_first_move))

        # No path found
        return None

    def get_pathfinding_hints(self, obs: AgentObservation) -> str:
        """Generate pathfinding hints for important visible objects.

        Args:
            obs: Agent observation

        Returns:
            Formatted pathfinding hints string, or empty string if no hints
        """
        agent_x = self._policy_env_info.obs_width // 2
        agent_y = self._policy_env_info.obs_height // 2

        # Important objects to pathfind to
        important_objects = {
            "charger", "assembler", "chest",
            "carbon_extractor", "oxygen_extractor",
            "germanium_extractor", "silicon_extractor",
        }

        # Build wall map
        blocked = self._build_wall_map(obs)

        # Find important objects and their positions
        targets: dict[str, tuple[int, int]] = {}
        for token in obs.tokens:
            if token.feature.name == "tag" and token.value < len(self._policy_env_info.tags):
                tag_name = self._policy_env_info.tags[token.value]
                if tag_name in important_objects:
                    pos = (token.row(), token.col())
                    # Skip if at agent position
                    if pos != (agent_x, agent_y):
                        # Keep closest of each type
                        if tag_name not in targets:
                            targets[tag_name] = pos
                        else:
                            # Compare distances
                            old_dist = abs(targets[tag_name][0] - agent_x) + abs(targets[tag_name][1] - agent_y)
                            new_dist = abs(pos[0] - agent_x) + abs(pos[1] - agent_y)
                            if new_dist < old_dist:
                                targets[tag_name] = pos

        if not targets:
            return ""

        # Calculate first move for each target
        entries = []
        grid_width = self._policy_env_info.obs_width
        grid_height = self._policy_env_info.obs_height
        start = (agent_x, agent_y)

        for obj_name, target_pos in sorted(targets.items()):
            first_move = self._bfs_first_move(start, target_pos, blocked, grid_width, grid_height)

            # Calculate relative position for context
            rel_x = target_pos[0] - agent_x
            rel_y = target_pos[1] - agent_y
            distance = abs(rel_x) + abs(rel_y)

            # Direction description
            direction = pos_to_dir(rel_x, rel_y) if (rel_x != 0 or rel_y != 0) else "here"

            if first_move:
                # Format: "assembler (3N2E, 5 tiles) -> move_east"
                entries.append(f"  {obj_name} ({direction}, {distance} tiles) -> {first_move}")
            else:
                # No path found
                entries.append(f"  {obj_name} ({direction}, {distance} tiles) -> NO PATH (blocked)")

        if not entries:
            return ""

        # Load template and substitute
        template = _load_prompt_template("pathfinding_hints")
        return template.replace("{{PATHFINDING_ENTRIES}}", "\n".join(entries))
