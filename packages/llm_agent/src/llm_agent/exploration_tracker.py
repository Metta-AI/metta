"""Exploration tracking for LLM agents."""

from typing import TYPE_CHECKING

from llm_agent.utils import pos_to_dir

if TYPE_CHECKING:
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface
    from mettagrid.simulator import AgentObservation


class ExplorationTracker:
    def __init__(self, policy_env_info: "PolicyEnvInterface"):
        """Initialize exploration tracker.

        Args:
            policy_env_info: Policy environment interface for observation specs
        """
        self.policy_env_info = policy_env_info
        self.obs_width = policy_env_info.obs_width
        self.obs_height = policy_env_info.obs_height

        # Position tracking
        self._global_x = 0
        self._global_y = 0
        self._current_window_positions: list[tuple[int, int]] = [(0, 0)]
        self._all_visited_positions: set[tuple[int, int]] = {(0, 0)}

        # Discovered objects: {object_type: (global_x, global_y)}
        self._discovered_objects: dict[str, tuple[int, int]] = {}

        # Extractor stats: {extractor_type: {"position": (x,y), "visits": int, "collected": int}}
        self._extractor_stats: dict[str, dict] = {}

        # Other agents: {agent_id: {"position": (x,y), "inventory": dict, "last_seen_step": int}}
        self._other_agents_info: dict[int, dict] = {}

        # Last inventory for tracking collection
        self._last_inventory: dict[str, int] = {}

    @property
    def global_x(self) -> int:
        return self._global_x

    @property
    def global_y(self) -> int:
        return self._global_y

    @property
    def all_visited_positions(self) -> set[tuple[int, int]]:
        return self._all_visited_positions

    @property
    def discovered_objects(self) -> dict[str, tuple[int, int]]:
        return self._discovered_objects

    def update_position(self, action_name: str) -> None:
        """Update global position based on action taken.

        Args:
            action_name: Name of the action (e.g., "move_east")
        """
        if action_name == "move_east":
            self._global_x += 1
        elif action_name == "move_west":
            self._global_x -= 1
        elif action_name == "move_north":
            self._global_y -= 1
        elif action_name == "move_south":
            self._global_y += 1

        pos = (self._global_x, self._global_y)
        self._current_window_positions.append(pos)
        self._all_visited_positions.add(pos)

    def reset_window_positions(self) -> None:
        """Reset window positions for new summary interval."""
        self._current_window_positions = [(self._global_x, self._global_y)]

    def extract_discovered_objects(self, obs: "AgentObservation") -> None:
        """Extract and track discovered objects from observation.

        Args:
            obs: Agent observation
        """
        agent_x = self.obs_width // 2
        agent_y = self.obs_height // 2

        important_types = {
            "charger",
            "assembler",
            "chest",
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        }

        extractor_types = {
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        }

        for token in obs.tokens:
            if token.feature.name == "tag" and token.value < len(self.policy_env_info.tags):
                tag_name = self.policy_env_info.tags[token.value]
                if tag_name in important_types:
                    rel_x = token.row() - agent_x
                    rel_y = token.col() - agent_y
                    global_x = self._global_x + rel_x
                    global_y = self._global_y + rel_y

                    if tag_name not in self._discovered_objects:
                        self._discovered_objects[tag_name] = (global_x, global_y)

                    if tag_name in extractor_types and rel_x == 0 and rel_y == 0:
                        if tag_name not in self._extractor_stats:
                            self._extractor_stats[tag_name] = {
                                "position": (global_x, global_y),
                                "visits": 0,
                                "collected": 0,
                            }
                        self._extractor_stats[tag_name]["visits"] += 1

    def extract_other_agents_info(self, obs: "AgentObservation", current_step: int) -> None:
        """Extract other agents' positions and inventories from observation.

        Args:
            obs: Agent observation
            current_step: Current step number for tracking when last seen
        """
        agent_x = self.obs_width // 2
        agent_y = self.obs_height // 2

        agents_in_view: dict[int, dict] = {}

        for token in obs.tokens:
            if token.row() == agent_x and token.col() == agent_y:
                continue

            if token.feature.name == "agent:id":
                other_agent_id = token.value
                rel_x = token.row() - agent_x
                rel_y = token.col() - agent_y
                global_x = self._global_x + rel_x
                global_y = self._global_y + rel_y

                if other_agent_id not in agents_in_view:
                    agents_in_view[other_agent_id] = {
                        "position": (global_x, global_y),
                        "inventory": {},
                        "last_seen_step": current_step,
                    }

            if token.feature.name.startswith("inv:"):
                for _aid, info in agents_in_view.items():
                    rel_x = token.row() - agent_x
                    rel_y = token.col() - agent_y
                    agent_global = (self._global_x + rel_x, self._global_y + rel_y)
                    if info["position"] == agent_global:
                        resource = token.feature.name[4:]
                        if token.value > 0:
                            info["inventory"][resource] = token.value
                        break

        for aid, info in agents_in_view.items():
            self._other_agents_info[aid] = info

    def update_extractor_collection(self, new_inventory: dict[str, int]) -> None:
        """Update extractor collection stats based on inventory changes.

        Args:
            new_inventory: Current inventory
        """
        resource_to_extractor = {
            "carbon": "carbon_extractor",
            "oxygen": "oxygen_extractor",
            "germanium": "germanium_extractor",
            "silicon": "silicon_extractor",
        }

        for resource, extractor in resource_to_extractor.items():
            old_amount = self._last_inventory.get(resource, 0)
            new_amount = new_inventory.get(resource, 0)
            if new_amount > old_amount:
                collected = new_amount - old_amount
                if extractor in self._extractor_stats:
                    self._extractor_stats[extractor]["collected"] += collected

        self._last_inventory = new_inventory.copy()

    def get_other_agents_text(self, current_step: int) -> str:
        """Get formatted text of other agents' last known states.

        Args:
            current_step: Current step number for calculating steps ago

        Returns:
            Formatted string showing other agents' positions and inventories.
        """
        if not self._other_agents_info:
            return ""

        lines = ["=== OTHER AGENTS (last seen) ==="]
        for agent_id, info in sorted(self._other_agents_info.items()):
            pos = info["position"]
            inv = info["inventory"]
            step = info["last_seen_step"]
            steps_ago = current_step - step

            inv_str = ", ".join(f"{k}={v}" for k, v in sorted(inv.items()) if v > 0) if inv else "empty"
            lines.append(f"  Agent {agent_id}: {pos_to_dir(*pos)} ({steps_ago} steps ago) | inv: {inv_str}")

        return "\n".join(lines)

    def get_discovered_objects_text(self) -> str:
        """Get formatted text of discovered objects for the prompt.

        Returns:
            Formatted string listing discovered objects and their locations.
        """
        if not self._discovered_objects:
            return ""

        lines = ["=== DISCOVERED OBJECTS (from exploration) ==="]

        extractor_types = {
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        }

        for obj_type, (gx, gy) in sorted(self._discovered_objects.items()):
            if obj_type in extractor_types and obj_type in self._extractor_stats:
                stats = self._extractor_stats[obj_type]
                visits = stats["visits"]
                collected = stats["collected"]
                resource = obj_type.replace("_extractor", "")
                loc = pos_to_dir(gx, gy)
                lines.append(f"  - {obj_type}: {loc} (visited {visits}x, collected {collected} {resource})")
            else:
                lines.append(f"  - {obj_type}: {pos_to_dir(gx, gy)}")

        return "\n".join(lines)

    def get_visible_extractors(self, obs: "AgentObservation") -> list[str]:
        """Get list of extractor types visible in current observation.

        Args:
            obs: Agent observation

        Returns:
            List of visible extractor type names
        """
        visible = []
        extractor_types = {
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        }

        for token in obs.tokens:
            if token.feature.name == "tag" and token.value < len(self.policy_env_info.tags):
                tag_name = self.policy_env_info.tags[token.value]
                if tag_name in extractor_types and tag_name not in visible:
                    visible.append(tag_name)

        return visible

    def get_summary_info(self) -> dict:
        """Get summary information for history/debug output.

        Returns:
            Dictionary with position, exploration, and discovery info
        """
        return {
            "global_x": self._global_x,
            "global_y": self._global_y,
            "total_explored": len(self._all_visited_positions),
            "window_positions": list(self._current_window_positions),
            "discovered_objects": list(self._discovered_objects.keys()),
        }
