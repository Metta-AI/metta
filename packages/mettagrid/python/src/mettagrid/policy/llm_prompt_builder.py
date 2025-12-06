"""Dynamic prompt builder for LLM policy with context window management.

This module provides intelligent prompt building that:
1. Sends basic game rules only once per N steps (context window)
2. Sends only observable changes at each step
3. Manages context windows of configurable size
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


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
    ):
        """Initialize prompt builder.

        Args:
            policy_env_info: Policy environment interface with feature/tag/action specs
            context_window_size: Number of steps before resending basic info (default: 20)
            mg_cfg: Optional MettaGridConfig to extract chest vibe transfers and other game-specific info
            debug_mode: If True, print debug information (default: False)
        """
        self._policy_env_info = policy_env_info
        # Ensure context_window_size is an int (may come as string from config)
        self._context_window_size = int(context_window_size)
        self._step_counter = 0
        self._last_visible: VisibleElements | None = None
        self._debug_mode = debug_mode

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
                    print(f"[LLMPromptBuilder] No chest vibe transfers found. chest_config={chest_config}, has vibe_transfers={hasattr(chest_config, 'vibe_transfers') if chest_config else 'N/A'}")
        else:
            if self._debug_mode:
                print("[LLMPromptBuilder] No mg_cfg provided, chest vibe transfers will be empty")

        # Pre-build static content (game rules, coordinate system)
        self._basic_info_cache = self._build_basic_info()

    def _build_basic_info(self) -> str:
        """Build minimal game information prompt.

        This is sent:
        - At step 1
        - Every N steps (when context window resets)

        Returns:
            Minimal game rules - only what agent cannot discover
        """
        # Build all recipes section
        all_recipes_section = self._build_all_recipes()

        # Build id_map reference from mg_cfg
        id_map_section = self._build_id_map_section()

        return f"""=== GOAL ===
Deposit HEARTs into CHEST to earn rewards. Team score = total hearts deposited.

=== HOW TO PLAY ===
1. EXPLORE to find extractors (carbon, oxygen, germanium, silicon)
2. COLLECT by standing ADJACENT to extractors (not on top)
3. CRAFT at ASSEMBLER: change vibe to heart_a, then move toward it
4. DEPOSIT at CHEST: change vibe to heart_b, then move toward it
5. RECHARGE at CHARGER when energy is low

{all_recipes_section}

=== VIBES ===
- heart_a: craft hearts at assembler
- heart_b: deposit hearts at chest
- default: recharge at charger

=== TIPS ===
- Stand ADJACENT to stations, then move TOWARD them to interact
- Silicon extractor costs 20 energy - watch your energy!
- If energy < 30, find a CHARGER immediately

{id_map_section}"""

    def _build_heart_recipe(self) -> str:
        """Build heart recipe string from assembler protocols.

        Returns:
            Human-readable heart crafting recipe
        """
        if not self._policy_env_info.assembler_protocols:
            return "1 HEART = 10 carbon + 10 oxygen + 2 germanium + 30 silicon"

        # Find the first heart protocol (1 heart output)
        for protocol in self._policy_env_info.assembler_protocols:
            if protocol.output_resources.get("heart", 0) == 1:
                # Format: "1 HEART = X carbon + Y oxygen + Z germanium + W silicon"
                parts = []
                for resource in ["carbon", "oxygen", "germanium", "silicon", "energy"]:
                    amount = protocol.input_resources.get(resource, 0)
                    if amount > 0:
                        parts.append(f"{amount} {resource}")
                if parts:
                    return "1 HEART = " + " + ".join(parts)

        # Fallback to default recipe
        return "1 HEART = 10 carbon + 10 oxygen + 2 germanium + 30 silicon"

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
            for resource in ["carbon", "oxygen", "germanium", "silicon", "energy", "heart", "decoder", "modulator", "resonator", "scrambler"]:
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

    def _build_spatial_grid(self, obs: AgentObservation) -> dict[tuple[int, int], str]:
        """Build spatial grid mapping positions to object tags.

        Args:
            obs: Agent observation

        Returns:
            Dictionary mapping (x, y) to tag name at that location
        """
        grid: dict[tuple[int, int], str] = {}

        for token in obs.tokens:
            # row() = Y (North/South), col() = X (East/West)
            x, y = token.col(), token.row()

            # Only care about tag features for the grid
            if token.feature.name == "tag" and token.value < len(self._policy_env_info.tags):
                tag_name = self._policy_env_info.tags[token.value]
                # Don't overwrite if already have a tag (first one wins)
                if (x, y) not in grid:
                    grid[(x, y)] = tag_name

        return grid

    def _build_spatial_grid_section(
        self, spatial_grid: dict[tuple[int, int], str], agent_x: int, agent_y: int
    ) -> str:
        """Build ASCII visualization of spatial grid for LLM.

        Args:
            spatial_grid: Dictionary mapping (row, col) to tag name
            agent_x: Agent's row coordinate
            agent_y: Agent's col coordinate

        Returns:
            ASCII grid section
        """
        obs_width = self._policy_env_info.obs_width
        obs_height = self._policy_env_info.obs_height

        lines = ["=== MAP (11x11 view, you are @) ==="]

        # Column headers (x-axis = columns)
        header = "  "
        for col in range(obs_width):
            header += f"{col % 10}"
        lines.append(header)

        # Grid rows (y-axis = rows)
        # spatial_grid keys are (x, y) where x=col(), y=row()
        for row in range(obs_height):
            row_str = f"{row % 10} "
            for col in range(obs_width):
                if col == agent_x and row == agent_y:
                    row_str += "@"  # Agent position
                elif (col, row) in spatial_grid:
                    tag = spatial_grid[(col, row)]
                    # Use distinctive letter for each type
                    if tag == "wall":
                        row_str += "W"
                    elif tag == "agent":
                        row_str += "P"  # Player (other agent)
                    elif tag == "charger":
                        row_str += "+"  # Charger (energy)
                    elif tag == "chest":
                        row_str += "H"  # Chest (heart storage)
                    elif tag == "assembler":
                        row_str += "A"  # Assembler
                    elif "extractor" in tag:
                        # Use first letter of resource for extractor
                        row_str += tag[0].upper()  # C, O, G, S for carbon, oxygen, etc.
                    else:
                        row_str += tag[0].upper()
                else:
                    row_str += "."  # Empty
            lines.append(row_str)

        # Legend
        lines.append("")
        lines.append("Legend: @ = You, . = Empty (can walk), W = Wall")
        lines.append("  + = Charger, A = Assembler, C/O/G/S = Extractors, H = Chest, P = Other agent")

        return "\n".join(lines)

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

=== DECISION PRIORITY ===

1. Check ADJACENT TILES to see what's around you
2. If adjacent to a useful object:
   - Set the right vibe if needed (heart_a for assembler, heart_b for chest)
   - Move INTO the object to use it
3. If not adjacent to anything useful:
   - Move toward the nearest useful object based on NEARBY AGENTS/OBJECTS info
   - Need resources? Find extractors
   - Have resources? Find assembler to craft hearts
   - Have heart? Find chest to deposit
   - Low energy? Find charger

⚠️ OUTPUT FORMAT ⚠️
You MUST respond with a JSON object. No other text before or after.

{{"reasoning": "<your step-by-step thinking>", "action": "<action_name>"}}

Example:
{{"reasoning": "I need carbon. Carbon extractor is to the east. Moving east.", "action": "move_east"}}
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
            # Build common actions list
            common_actions = ["noop", "move_north", "move_south", "move_west", "move_east"]
            # Add vibe actions for resources/hearts
            vibe_actions = [name for name in self._policy_env_info.action_names if name.startswith("change_vibe_") and any(x in name for x in ["heart", "carbon", "oxygen", "silicon", "germanium", "default"])]
            action_list = common_actions + vibe_actions[:10]  # Limit to top 10 vibe actions

            prompt = f"""{self.observable_prompt(obs)}

Check ADJACENT TILES. If next to a useful object, use it. Otherwise move toward one.

VALID: {", ".join(action_list[:8])}, ...

Respond with JSON: {{"reasoning": "<thinking>", "action": "<action_name>"}}
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
            Human-readable description with actionable mechanics
        """
        descriptions = {
            "agent": "Another COG. Check agent:group (same = teammate). Can share energy by vibing 'energy' when adjacent.",
            "assembler": "Crafting station. Set vibe, position adjacent, move into it. Consumes resources, produces gear/HEARTs. Check cooldown_remaining. IMMUNE to clipping.",
            "carbon_extractor": "+2 carbon/use (25 max uses). Position adjacent, move into it. Check remaining_uses and cooldown_remaining.",
            "oxygen_extractor": "+10 oxygen/use (refills over 100 turns). Can use partially during cooldown. Check cooldown_remaining for charge level.",
            "germanium_extractor": "+(N+1) germanium for N adjacent cogs (1 use only!). COORDINATE WITH TEAM for max yield. Check remaining_uses.",
            "silicon_extractor": "+15 silicon/use (COSTS 20 ENERGY!, 10 max uses). Only use if inv:energy >20. Check remaining_uses.",
            "charger": "Solar array. +50 energy (recharges over 10 turns). Can use partially. Check cooldown_remaining for charge level.",
            "chest": "Resource storage. Set vibe, move into it. DEPOSIT HEARTs using 'heart_b' vibe. See CHEST VIBE TRANSFERS section.",
            "wall": "Impassable obstacle. DO NOT move into walls. Navigate around.",
            "altar": "Ritual site. Costs energy, provides rewards. Has cooldown. Check cooldown_remaining and inv:energy.",
            "converter": "Resource-to-energy converter. No energy cost, has cooldown. Check cooldown_remaining.",
            "generator": "Resource harvester. Has cooldown. Check cooldown_remaining.",
        }
        return descriptions.get(tag_name, f"Unknown station: {tag_name}. Move adjacent and interact cautiously.")

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

    def _build_protocols_documentation(self) -> str:
        """Build documentation for assembler protocols explaining vibe-based crafting.

        Returns:
            Formatted documentation string explaining how assembler protocols work
        """
        if not self._policy_env_info.assembler_protocols:
            return ""

        lines = ["=== ASSEMBLER PROTOCOLS (CRAFTING) ===", ""]
        lines.append("Assemblers craft items using protocols activated by VIBES:")
        lines.append("- Stand next to an assembler")
        lines.append("- Use change_vibe_* actions to set your vibe")
        lines.append("- Assembler checks if your vibe matches any protocol requirements")
        lines.append("- If match found AND you have required resources → item is crafted")
        lines.append("")
        lines.append("IMPORTANT MECHANICS:")
        lines.append("- Some protocols require MULTIPLE agents with matching vibes (check protocol_input:* features)")
        lines.append("- Assembler has cooldown after crafting (check cooldown_remaining feature)")
        lines.append("- Resources are CONSUMED from your inventory when crafting")
        lines.append("")

        # Group protocols by output resource for better organization
        protocols_by_output: dict[str, list] = {}
        for protocol in self._policy_env_info.assembler_protocols:
            # Get the primary output resource (first in output_resources dict)
            output_items = list(protocol.output_resources.keys())
            if output_items:
                primary_output = output_items[0]
                if primary_output not in protocols_by_output:
                    protocols_by_output[primary_output] = []
                protocols_by_output[primary_output].append(protocol)

        lines.append("AVAILABLE PROTOCOLS:")
        for output_resource, protocols in sorted(protocols_by_output.items()):
            lines.append(f"\n  Crafting {output_resource.upper()}:")
            for protocol in protocols:
                vibes_str = " + ".join(f"'{v}'" for v in protocol.vibes)
                if not vibes_str:
                    vibes_str = "no specific vibe"

                # Format inputs
                inputs_str = ", ".join(f"{amt} {res}" for res, amt in protocol.input_resources.items())

                # Format outputs
                outputs_str = ", ".join(f"{amt} {res}" for res, amt in protocol.output_resources.items())

                lines.append(f"    Vibe: {vibes_str}")
                lines.append(f"      Needs: {inputs_str}")
                lines.append(f"      Makes: {outputs_str}")
                if protocol.min_agents > 0:
                    lines.append(f"      Min agents: {protocol.min_agents}")

        return "\n".join(lines)

    def _build_chest_vibes_documentation(self) -> str:
        """Build documentation for chest vibe transfers explaining deposit/withdraw mechanics.

        Returns:
            Formatted documentation string explaining how chest vibe transfers work
        """
        if not self._chest_vibe_transfers:
            return ""

        lines = ["=== CHEST VIBE TRANSFERS (STORAGE) ===", ""]
        lines.append("Chests store resources. Use change_vibe_* to deposit or withdraw:")
        lines.append("- Stand next to a chest")
        lines.append("- Use change_vibe_* action to set your vibe")
        lines.append("- POSITIVE values = DEPOSIT resources into chest")
        lines.append("- NEGATIVE values = WITHDRAW resources from chest")
        lines.append("")
        lines.append("VIBE TRANSFER RULES:")

        for vibe_name, transfers in sorted(self._chest_vibe_transfers.items()):
            if vibe_name == "default":
                continue  # Skip default vibe, it's usually very permissive

            deposits = []
            withdraws = []
            for resource, delta in transfers.items():
                if delta > 0:
                    deposits.append(f"+{delta} {resource}")
                elif delta < 0:
                    withdraws.append(f"{delta} {resource}")

            if deposits or withdraws:
                lines.append(f"\n  Vibe '{vibe_name}':")
                if deposits:
                    lines.append(f"    Deposits: {', '.join(deposits)}")
                if withdraws:
                    lines.append(f"    Withdraws: {', '.join(withdraws)}")

        return "\n".join(lines)

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
            # row() = Y (North/South), col() = X (East/West)
            x, y = token.col(), token.row()
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

    def _find_nearby_objects(self, obs: AgentObservation, agent_x: int, agent_y: int) -> list[dict]:
        """Find all nearby objects with their directions and distances.

        Args:
            obs: Agent observation
            agent_x: Agent's X coordinate
            agent_y: Agent's Y coordinate

        Returns:
            List of object info dicts sorted by distance
        """
        objects = []
        seen_locations: set[tuple[int, int]] = set()

        for token in obs.tokens:
            # row() = Y (North/South), col() = X (East/West)
            x, y = token.col(), token.row()

            # Skip agent's own location
            if x == agent_x and y == agent_y:
                continue

            # Skip already processed locations
            if (x, y) in seen_locations:
                continue

            # Only process tag tokens (they identify objects)
            if token.feature.name == "tag" and token.value < len(self._policy_env_info.tags):
                seen_locations.add((x, y))
                tag_name = self._policy_env_info.tags[token.value]
                dx = x - agent_x
                dy = y - agent_y
                distance = abs(dx) + abs(dy)  # Manhattan distance
                direction = self._get_direction_name(dx, dy)

                # Collect additional properties
                properties = []
                inventory = {}
                for t in obs.tokens:
                    # row() = Y, col() = X, so compare col() with x and row() with y
                    if t.col() == x and t.row() == y:
                        if t.feature.name == "cooldown_remaining" and t.value > 0:
                            properties.append(f"cooldown: {t.value}")
                        elif t.feature.name == "remaining_uses":
                            properties.append(f"uses: {t.value}")
                        elif t.feature.name == "agent:group":
                            properties.append(f"group: {t.value}")
                        elif t.feature.name.startswith("inv:") and t.value > 0:
                            # Collect inventory for other agents
                            resource = t.feature.name[4:]  # Remove "inv:" prefix
                            inventory[resource] = t.value

                objects.append({
                    "name": tag_name,
                    "direction": direction,
                    "distance": distance,
                    "properties": properties,
                    "inventory": inventory,
                })

        # Sort by distance
        return sorted(objects, key=lambda x: x["distance"])

    def _build_nearby_objects_section(self, objects: list[dict]) -> str:
        """Build the nearby objects section for the prompt.

        Args:
            objects: List of nearby object info dicts

        Returns:
            Formatted nearby objects section
        """
        lines = ["=== NEARBY OBJECTS (what you can see) ==="]
        for obj in objects:
            desc = f"  {obj['name']} - {obj['direction']} (distance: {obj['distance']})"
            if obj["properties"]:
                desc += f" [{', '.join(obj['properties'])}]"
            # Show inventory for other agents
            if obj.get("inventory"):
                inv_str = ", ".join(f"{k}:{v}" for k, v in sorted(obj["inventory"].items()))
                desc += f" inv={{" + inv_str + "}}"
            lines.append(desc)
        return "\n".join(lines)

    def _build_nearby_agents_section(self, agents: list[dict]) -> str:
        """Build section showing nearby agents and their inventories.

        Args:
            agents: List of agent info dicts (filtered from nearby objects)

        Returns:
            Formatted section showing teammate positions and inventories
        """
        lines = ["=== NEARBY AGENTS ==="]
        for agent in agents:
            # Determine if ally or enemy based on group
            group_info = ""
            for prop in agent["properties"]:
                if prop.startswith("group:"):
                    group_info = f" ({prop})"
                    break

            desc = f"  Agent {agent['direction']} (distance: {agent['distance']}){group_info}"

            # Show their inventory
            if agent.get("inventory"):
                inv_parts = []
                for resource, amount in sorted(agent["inventory"].items()):
                    inv_parts.append(f"{resource}:{amount}")
                if inv_parts:
                    desc += f" - inventory: {', '.join(inv_parts)}"
            lines.append(desc)

        return "\n".join(lines)

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
            # row() = Y, col() = X, so compare col() with agent_x and row() with agent_y
            if token.col() == agent_x and token.row() == agent_y:
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
            # row() = Y (North/South), col() = X (East/West)
            x, y = token.col(), token.row()

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
        """Build section showing visible objects with coordinates.

        Format: "object_name at: x=X, y=Y (properties)"

        Args:
            objects: List of object dicts from _extract_visible_objects_with_coords

        Returns:
            Formatted section with spatial coordinates
        """
        lines = ["=== VISIBLE OBJECTS (relative to you at 0,0) ==="]
        lines.append("Coordinates: x+ is East, x- is West, y+ is South, y- is North")
        lines.append("")

        for obj in objects:
            # Format: "assembler at: x=2, y=-3"
            line = f"  {obj['name']} at: x={obj['x']}, y={obj['y']}"

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
