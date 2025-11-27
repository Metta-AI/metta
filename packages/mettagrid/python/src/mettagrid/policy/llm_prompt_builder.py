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
    ):
        """Initialize prompt builder.

        Args:
            policy_env_info: Policy environment interface with feature/tag/action specs
            context_window_size: Number of steps before resending basic info (default: 20)
            mg_cfg: Optional MettaGridConfig to extract chest vibe transfers and other game-specific info
        """
        self._policy_env_info = policy_env_info
        # Ensure context_window_size is an int (may come as string from config)
        self._context_window_size = int(context_window_size)
        self._step_counter = 0
        self._last_visible: VisibleElements | None = None

        # Extract chest vibe transfers if config is provided
        self._chest_vibe_transfers: dict[str, dict[str, int]] = {}
        if mg_cfg is not None:
            chest_config = mg_cfg.game.objects.get("chest")
            if chest_config and hasattr(chest_config, "vibe_transfers"):
                self._chest_vibe_transfers = chest_config.vibe_transfers
                print(f"[LLMPromptBuilder] Loaded chest vibe transfers: {self._chest_vibe_transfers}")
            else:
                print(f"[LLMPromptBuilder] No chest vibe transfers found. chest_config={chest_config}, has vibe_transfers={hasattr(chest_config, 'vibe_transfers') if chest_config else 'N/A'}")
        else:
            print("[LLMPromptBuilder] No mg_cfg provided, chest vibe transfers will be empty")

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

        # Build assembler protocols documentation
        protocols_doc = self._build_protocols_documentation()

        # Build chest vibe transfers documentation
        chest_vibes_doc = self._build_chest_vibes_documentation()

        return f"""=== MISSION: COGS VS CLIPS ===

You are a COG unit deployed to asteroid Machina VII. Your ONLY mission objective: Produce and deposit HEARTs into chests.

CRITICAL SUCCESS METRICS:
- Team performance is measured ONLY by HEARTs deposited in chests
- Individual achievement is irrelevant - coordinate with teammates
- HEARTs are crafted at assemblers using resources from extractors
- Depositing HEARTs requires using the 'heart_b' vibe at chests

=== YOUR LOADOUT ===

ENERGY MANAGEMENT (CRITICAL):
- Starting energy: 100 (max capacity: 100)
- Movement costs 2 energy per step (net -1 after +1 solar regen)
- Passive solar: +1 energy per turn
- Chargers: +50 energy (10 turn cooldown, can use partially)
- LOW ENERGY = MISSION FAILURE - monitor inv:energy constantly

CARGO LIMITS:
- Resources (carbon/oxygen/germanium/silicon): 100 total combined
- Gear (decoder/modulator/scrambler/resonator): 5 total combined
- HEARTs: 1 maximum (deposit immediately to free space)

=== COORDINATE SYSTEM ===

- Observation window is {obs_width}x{obs_height} grid (YOU at center: x={agent_x}, y={agent_y})
- Coordinates are EGOCENTRIC (relative to your position)
- x=0 is West, x={obs_width - 1} is East
- y=0 is North, y={obs_height - 1} is South

CARDINAL DIRECTIONS FROM YOUR POSITION:
- North: x={agent_x}, y={agent_y - 1}
- South: x={agent_x}, y={agent_y + 1}
- East: x={agent_x + 1}, y={agent_y}
- West: x={agent_x - 1}, y={agent_y}

=== OBSERVATION FORMAT ===

Observations are token lists. Each token has:
- "feature": Property name (tag, inv:energy, cooldown_remaining, etc.)
- "location": {{"x": col, "y": row}}
- "value": Numeric value

INTERPRETING TOKENS:
1. Tokens at YOUR location (x={agent_x}, y={agent_y}) = YOUR state (inventory, last_action, last_reward)
2. Tokens at OTHER locations = visible objects/agents
3. Multiple tokens at same location = same object with multiple properties
4. "tag" feature = object type ID

=== MOVEMENT & INTERACTION ===

MOVEMENT RULES (CRITICAL):
- Tile is WALKABLE if it has NO tokens
- Tile is BLOCKED if ANY tokens exist (wall/object/agent)
- ALWAYS verify target tile is empty before moving
- Moving costs 2 energy (typically -1 net after +1 solar regen)

STATION INTERACTION PROTOCOL:
1. Position adjacent to station (NOT on top of it)
2. Set appropriate vibe using change_vibe_* action (if needed)
3. Move TOWARD the station to activate it
4. Check cooldown_remaining and remaining_uses features

=== STATION TYPES & MECHANICS ===

EXTRACTORS (Resource Harvesting):
- Carbon Extractor: +2 carbon per use (25 max uses)
- Oxygen Extractor: +10 oxygen (refills over 100 turns, can use partially)
- Germanium Extractor: +(N+1) germanium for N cogs (1 use, coordinate with team!)
- Silicon Extractor: +15 silicon (costs 20 energy!, 10 max uses)
- Solar Array (Charger): +50 energy (recharges over 10 turns, can use partially)

ASSEMBLERS (Crafting):
- Activate by vibing specific patterns and moving into it
- Consumes input resources from adjacent cogs' inventories
- Produces output resources distributed to adjacent cogs
- See ASSEMBLER PROTOCOLS section for vibe-based crafting recipes

CHESTS (Storage):
- Vibe controls deposit (+) vs withdraw (-)
- See CHEST VIBE TRANSFERS section for vibe-resource mappings
- CRITICAL: Deposit HEARTs using 'heart_b' vibe to score points!

{protocols_doc}

{chest_vibes_doc}

AVAILABLE ACTIONS:
{chr(10).join(action_docs)}

=== CRITICAL GAMEPLAY RULES ===

MOVEMENT:
- ONLY move to coordinates with ZERO tokens (empty tiles are walkable)
- ANY token at target location = BLOCKED (wall, station, or agent)
- Verify target is empty BEFORE issuing move action

INTERACTION PROTOCOL:
1. Position adjacent to station (one tile away)
2. Set vibe if needed (change_vibe_* action)
3. Move TOWARD station to activate

ENERGY DISCIPLINE:
- Monitor inv:energy constantly (movement costs -1 net energy)
- Recharge at chargers BEFORE running low (<20 energy)
- Silicon extractors cost 20 energy - check before using!

MISSION SUCCESS SEQUENCE:
1. Gather resources from extractors (carbon, oxygen, germanium, silicon)
2. Craft HEARTs at assembler (set vibe to 'heart_a', ensure resources)
3. Deposit HEARTs at chest (set vibe to 'heart_b')
4. Repeat - team score = HEARTs in chests
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

=== DECISION PRIORITY (follow in order) ===

1. IF BLOCKED (wall/object adjacent in your current direction):
   → Try a DIFFERENT direction immediately. Never repeat a blocked move.
   → Check all 4 directions: North (y-1), South (y+1), East (x+1), West (x-1)
   → A tile is BLOCKED if it has ANY tokens. Empty tiles have NO tokens.

2. IF you see an EXTRACTOR (carbon/oxygen/germanium/silicon) nearby:
   → Move TOWARD it to gather resources. Resources are needed to craft HEARTs.
   → To use: get adjacent, then move INTO the extractor.

3. IF you have resources AND see an ASSEMBLER:
   → Move toward assembler to craft items.
   → Set vibe to 'heart_a' before using assembler to craft HEARTs.

4. IF you have a HEART AND see a CHEST:
   → Move toward chest to deposit.
   → Set vibe to 'heart_b' before moving into chest.

5. IF nothing useful visible:
   → EXPLORE! Move in any unblocked direction to discover the map.
   → Prefer directions you haven't tried recently.

NEVER USE 'noop' UNLESS:
- You are frozen (agent:frozen = 1)
- All 4 directions are blocked (completely surrounded)

Your goal: Find extractors → Gather resources → Find assembler → Craft HEARTs → Deposit in chest

⚠️ OUTPUT FORMAT - READ THIS FIRST ⚠️
Your response must be EXACTLY ONE action name. Nothing else.
NO analysis. NO explanation. NO markdown. NO "I think" or "Looking at".
Just the action name. One word (or underscore-separated words).

WRONG responses (will cause errors):
- "I need to analyze..."
- "Looking at the situation..."
- "move_east because..."
- "**move_east**"

CORRECT responses (exactly like this):
move_east
move_north
change_vibe_heart_a
noop
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

⚠️ RESPOND WITH EXACTLY ONE ACTION NAME. NO ANALYSIS. NO EXPLANATION. ⚠️

DECISION PRIORITY:
1. IF BLOCKED → Try different direction (check North/South/East/West for empty tiles)
2. IF see EXTRACTOR → Move toward it to gather resources
3. IF have resources + see ASSEMBLER → Move toward it, use heart_a vibe to craft
4. IF have HEART + see CHEST → Move toward it, use heart_b vibe to deposit
5. IF nothing visible → EXPLORE in any unblocked direction

NEVER 'noop' unless frozen or completely surrounded!

VALID: {", ".join(action_list[:8])}, ...

Output only the action name (e.g. move_east):
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
