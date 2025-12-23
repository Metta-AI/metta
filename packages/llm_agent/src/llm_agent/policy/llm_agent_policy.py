"""LLM-based per-agent policy base class for MettaGrid."""

import random
from abc import ABC, abstractmethod
from pathlib import Path

from llm_agent.action_parser import parse_action
from llm_agent.cost_tracker import CostTracker
from llm_agent.exploration_tracker import ExplorationTracker
from llm_agent.policy.prompt_builder import LLMPromptBuilder
from llm_agent.utils import pos_to_dir
from mettagrid.policy.policy import AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation


class LLMAgentPolicy(AgentPolicy, ABC):
    """Base per-agent LLM policy that queries an LLM for action selection.

    Subclasses must implement:
    - _init_client(): Initialize the LLM client
    - _call(): Make the actual LLM API call
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str,
        temperature: float = 0.7,
        verbose: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        agent_id: int = 0,
        enable_history_tracking: bool = True,
    ):
        super().__init__(policy_env_info)
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.agent_id = agent_id
        self.last_action: str | None = None
        self.enable_history_tracking = enable_history_tracking
        self.summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval
        self.debug_summary_interval = (
            int(debug_summary_interval) if isinstance(debug_summary_interval, str) else debug_summary_interval
        )

        self._init_tracking_state()
        self._init_prompt_builder(policy_env_info, context_window_size)
        self._check_assembler_variant(policy_env_info)
        self._init_client()

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the LLM client. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _call(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        """Make LLM API call and return (response, input_tokens, output_tokens).

        Must be implemented by subclasses.
        """
        pass

    def _init_tracking_state(self) -> None:
        """Initialize all tracking-related state variables."""
        self._debug_summary_step_count = 0
        self._debug_summary_actions: list[dict[str, str]] = []
        self._debug_summary_file: str | None = None
        self.conversation_history: list[dict] = []
        self._messages: list[dict[str, str]] = []
        self._history_summaries: list[str] = []
        self._max_history_summaries = 100
        self._current_window_actions: list[dict[str, str]] = []
        self._summary_step_count = 0
        self.exploration = ExplorationTracker(self.policy_env_info)
        self.cost_tracker = CostTracker()
        self._current_direction: str | None = None
        self._steps_in_direction: int = 0
        self._direction_change_threshold: int = 8

    def _init_prompt_builder(self, policy_env_info: PolicyEnvInterface, context_window_size: int) -> None:
        self.prompt_builder = LLMPromptBuilder(
            policy_env_info=policy_env_info,
            context_window_size=context_window_size,
            verbose=self.verbose,
            agent_id=self.agent_id,
        )

    def _check_assembler_variant(self, policy_env_info: PolicyEnvInterface) -> None:
        """Check AssemblerDrawsFromChestsVariant status using policy_env_info."""
        self._assembler_draws_from_chests = policy_env_info.assembler_chest_search_distance > 0

    def _add_to_messages(self, role: str, content: str) -> None:
        """Add a message to conversation history and prune if needed.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self._messages.append({"role": role, "content": content})

        # Prune to keep only last context_window_size turns (2 messages per turn)
        max_messages = self.prompt_builder.context_window_size * 2
        if len(self._messages) > max_messages:
            # Keep system message (if first) + last N messages
            if self._messages and self._messages[0].get("role") == "system":
                self._messages = [self._messages[0]] + self._messages[-(max_messages - 1) :]
            else:
                self._messages = self._messages[-max_messages:]

    def _get_messages_for_api(self, user_prompt: str) -> list[dict[str, str]]:
        """Get messages list for API call, including history + new user prompt.

        Args:
            user_prompt: Current user prompt to add

        Returns:
            List of messages for API call
        """
        # Add current user prompt to history
        self._add_to_messages("user", user_prompt)
        # Return a copy of messages for the API call
        return list(self._messages)

    @staticmethod
    def _get_net_direction(start_pos: tuple[int, int], end_pos: tuple[int, int]) -> str:
        """Calculate net direction of movement between two positions."""
        dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
        parts = []
        if dy < 0:
            parts.append("North")
        elif dy > 0:
            parts.append("South")
        if dx > 0:
            parts.append("East")
        elif dx < 0:
            parts.append("West")
        return "-".join(parts) if parts else "stationary"

    def _summarize_current_window(self) -> str:
        """Summarize the current context window's actions."""
        if not self._current_window_actions:
            return ""

        summary_info = self.exploration.get_summary_info()
        positions = summary_info["window_positions"]
        start_pos = positions[0] if positions else (0, 0)
        end_pos = positions[-1] if positions else (self.exploration.global_x, self.exploration.global_y)

        unique_count = len(set(positions))
        window_num = len(self._history_summaries) + 1
        net_dir = self._get_net_direction(start_pos, end_pos)

        return (
            f"[Window {window_num}] {pos_to_dir(*start_pos)} → {pos_to_dir(*end_pos)} "
            f"(heading {net_dir}) | {unique_count} new spots, {summary_info['total_explored']} total"
        )

    def _add_action_to_window(self, action: str, reasoning: str = "") -> None:
        """Track an action for the current context window summary."""
        action_info = {"action": action, "reasoning": reasoning}
        self._current_window_actions.append(action_info)

        if self.debug_summary_interval > 0:
            self._debug_summary_actions.append(action_info)

        direction_map = {"move_north": "north", "move_south": "south", "move_east": "east", "move_west": "west"}
        if action in direction_map:
            new_direction = direction_map[action]
            if new_direction == self._current_direction:
                self._steps_in_direction += 1
            else:
                self._current_direction = new_direction
                self._steps_in_direction = 1

        self.exploration.update_position(action)

    def _finalize_window_summary(self) -> None:
        """Create summary for current window and reset for next window."""
        if not self._current_window_actions:
            return

        summary = self._summarize_current_window()
        if summary:
            self._history_summaries.append(summary)
            if len(self._history_summaries) > self._max_history_summaries:
                self._history_summaries = self._history_summaries[-self._max_history_summaries :]
            print(f"\n[HISTORY Agent {self.agent_id}] {summary}\n")

        self._current_window_actions = []
        self.exploration.reset_window_positions()

    def _generate_debug_summary(self) -> None:
        """Generate an LLM debug summary for the last N steps and write to file."""
        if not self._debug_summary_actions:
            return

        action_counts: dict[str, int] = {}
        for action_info in self._debug_summary_actions:
            action = action_info.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        action_summary = ", ".join(f"{k}:{v}" for k, v in sorted(action_counts.items()))

        summary_info = self.exploration.get_summary_info()
        inventory = self._last_inventory
        inv_str = ", ".join(f"{k}={v}" for k, v in inventory.items() if v > 0 and k != "energy")
        energy = inventory.get("energy", 0)

        num_actions = len(self._debug_summary_actions)
        summary_prompt = f"""Summarize what Agent {self.agent_id} did in the last {num_actions} steps.

Actions taken: {action_summary}
Current position: ({summary_info["global_x"]}, {summary_info["global_y"]}) from origin
Explored tiles: {summary_info["total_explored"]}
Current inventory: {inv_str if inv_str else "empty"}
Energy: {energy}
Discovered objects: {", ".join(summary_info["discovered_objects"]) if summary_info["discovered_objects"] else "none"}

Write a 2-3 sentence summary of progress, challenges, and current strategy. Be concise."""

        try:
            summary_text = self._call_debug_summary_llm(summary_prompt)
            self._write_debug_summary_to_file(summary_info, inv_str, energy, action_summary, summary_text)
        except Exception as e:
            print(f"[WARNING] Failed to generate debug summary: {e}")

        self._debug_summary_actions = []

    def _call_debug_summary_llm(self, prompt: str) -> str:
        """Call LLM for debug summary generation. Uses the same _call() as main policy."""
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        response, _, _ = self._call(messages)
        return response or "No summary generated"

    def _write_debug_summary_to_file(
        self, summary_info: dict, inv_str: str, energy: int, action_summary: str, summary_text: str
    ) -> None:
        """Write debug summary to file and console."""
        if self._debug_summary_file is None:
            self._debug_summary_file = f"llm_debug_summary_agent{self.agent_id}.log"

        interval_num = self._debug_summary_step_count // self.debug_summary_interval
        step_start = (interval_num - 1) * self.debug_summary_interval + 1
        step_end = interval_num * self.debug_summary_interval

        with open(self._debug_summary_file, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[Agent {self.agent_id}] Steps {step_start}-{step_end}\n")
            f.write(f"{'=' * 60}\n")
            pos = f"({summary_info['global_x']}, {summary_info['global_y']})"
            f.write(f"Position: {pos} | Explored: {summary_info['total_explored']} tiles\n")
            f.write(f"Inventory: {inv_str if inv_str else 'empty'} | Energy: {energy}\n")
            f.write(f"Actions: {action_summary}\n")
            f.write(f"\nSummary: {summary_text}\n")

        print(f"\n[DEBUG SUMMARY Agent {self.agent_id}] Steps {step_start}-{step_end}: {summary_text}\n")

    def _get_history_summary_text(self) -> str:
        """Get formatted history summaries to prepend to prompts.

        Returns:
            Formatted string of past window summaries, or empty string if none.
        """
        if not self._history_summaries:
            return ""

        # Format window summaries
        window_summaries = "\n".join(f"  {summary}" for summary in self._history_summaries)

        # Load template and substitute variables
        template_path = Path(__file__).parent / "prompts" / "exploration_history.md"
        template = template_path.read_text()

        summary_info = self.exploration.get_summary_info()
        current_pos = pos_to_dir(summary_info["global_x"], summary_info["global_y"], verbose=True)
        return (
            template.replace("{{CURRENT_POSITION}}", current_pos)
            .replace("{{TOTAL_EXPLORED}}", str(summary_info["total_explored"]))
            .replace("{{WINDOW_SUMMARIES}}", window_summaries)
        )

    def _extract_inventory_from_obs(self, obs: AgentObservation) -> dict[str, int]:
        """Extract current inventory from observation.

        Args:
            obs: Agent observation

        Returns:
            Dictionary of resource -> amount
        """
        agent_x = self.policy_env_info.obs_width // 2
        agent_y = self.policy_env_info.obs_height // 2

        inventory = {}
        for token in obs.tokens:
            if token.row() == agent_x and token.col() == agent_y:
                if token.feature.name.startswith("inv:"):
                    resource = token.feature.name[4:]
                    inventory[resource] = token.value
                elif token.feature.name == "agent:energy" or token.feature.name == "energy":
                    inventory["energy"] = token.value

        return inventory

    def _get_heart_recipe(self) -> dict[str, int]:
        """Get the heart crafting recipe requirements from assembler protocols.

        Returns:
            Dictionary mapping resource names to required amounts.
            Falls back to default values if no protocol found.
        """
        # Try to get recipe from prompt builder's policy env info
        protocols = self.prompt_builder._policy_env_info.assembler_protocols
        if protocols:
            for protocol in protocols:
                if protocol.output_resources.get("heart", 0) == 1:
                    return {
                        "carbon": protocol.input_resources.get("carbon", 0),
                        "oxygen": protocol.input_resources.get("oxygen", 0),
                        "germanium": protocol.input_resources.get("germanium", 0),
                        "silicon": protocol.input_resources.get("silicon", 0),
                    }

        # Fallback to defaults
        return {"carbon": 10, "oxygen": 10, "germanium": 2, "silicon": 30}

    def _get_strategic_hints(self, inventory: dict[str, int], obs: AgentObservation | None = None) -> str:
        """Generate strategic hints based on current state.

        Args:
            inventory: Current inventory
            obs: Optional agent observation to check for visible extractors

        Returns:
            Strategic hints text to add to prompt
        """
        hints = []

        # Get recipe requirements (dynamic based on mission)
        recipe = self._get_heart_recipe()
        req_carbon = recipe["carbon"]
        req_oxygen = recipe["oxygen"]
        req_germanium = recipe["germanium"]
        req_silicon = recipe["silicon"]

        # Check for visible extractors that we need (TOP PRIORITY HINT)
        if obs is not None:
            visible_extractors = self.exploration.get_visible_extractors(obs)
            needed_extractors = []

            carbon = inventory.get("carbon", 0)
            oxygen = inventory.get("oxygen", 0)
            germanium = inventory.get("germanium", 0)
            silicon = inventory.get("silicon", 0)

            for ext in visible_extractors:
                if ext == "carbon_extractor" and carbon < req_carbon:
                    needed_extractors.append("carbon_extractor")
                elif ext == "oxygen_extractor" and oxygen < req_oxygen:
                    needed_extractors.append("oxygen_extractor")
                elif ext == "germanium_extractor" and germanium < req_germanium:
                    needed_extractors.append("germanium_extractor")
                elif ext == "silicon_extractor" and silicon < req_silicon:
                    needed_extractors.append("silicon_extractor")

            if needed_extractors:
                ext_list = ", ".join(needed_extractors)
                hints.append(
                    f"VISIBLE EXTRACTOR YOU NEED: {ext_list} - PURSUE IT NOW! Navigate around walls if blocked!"
                )

        # Energy warning
        energy = inventory.get("energy", 100)
        if energy < 20:
            hints.append("ENERGY CRITICAL (<20): Find charger IMMEDIATELY!")
        elif energy < 40:
            hints.append("ENERGY LOW (<40): Head to charger soon!")

        # Direction change suggestion
        if self._steps_in_direction >= self._direction_change_threshold:
            opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
            suggested = opposite.get(self._current_direction, "different")
            hints.append(
                f"You've gone {self._current_direction} for {self._steps_in_direction} steps. "
                f"Consider going {suggested}!"
            )

        # Distance from origin warning
        distance = abs(self.exploration.global_x) + abs(self.exploration.global_y)
        if distance > 25:
            hints.append(
                f"You're {distance} tiles from origin. Extractors are usually within 20 tiles - try going back!"
            )

        # Resource gathering hints
        carbon = inventory.get("carbon", 0)
        oxygen = inventory.get("oxygen", 0)
        germanium = inventory.get("germanium", 0)
        silicon = inventory.get("silicon", 0)
        heart = inventory.get("heart", 0)

        if heart > 0 and "chest" in self.exploration.discovered_objects:
            hints.append(f"You have {heart} heart(s)! Go to chest to deposit for reward!")
        elif carbon >= req_carbon and oxygen >= req_oxygen and germanium >= req_germanium and silicon >= req_silicon:
            hints.append("You have all resources for a heart! Find an assembler and use heart_a vibe!")
        else:
            missing = []
            if carbon < req_carbon:
                missing.append(f"carbon ({carbon}/{req_carbon})")
            if oxygen < req_oxygen:
                missing.append(f"oxygen ({oxygen}/{req_oxygen})")
            if germanium < req_germanium:
                missing.append(f"germanium ({germanium}/{req_germanium})")
            if silicon < req_silicon:
                missing.append(f"silicon ({silicon}/{req_silicon})")
            if missing:
                hints.append(f"Still need: {', '.join(missing)}")

        if not hints:
            return ""

        return "=== STRATEGIC HINTS ===\n" + "\n".join(hints)

    def _update_tracking(self, obs: AgentObservation) -> dict[str, int]:
        """Update all tracking state from observation and return inventory."""
        self.exploration.extract_discovered_objects(obs)
        self.exploration.extract_other_agents_info(obs, self.prompt_builder.step_count)
        inventory = self._extract_inventory_from_obs(obs)
        self.exploration.update_extractor_collection(inventory)
        self._last_inventory = inventory
        self._summary_step_count += 1
        self._debug_summary_step_count += 1
        return inventory

    def _handle_boundaries(self) -> None:
        """Handle summary, debug, and context window boundaries.

        Can be disabled by setting enable_history_tracking=False in constructor.
        """
        if not self.enable_history_tracking:
            return

        at_boundary = (self._summary_step_count - 1) % self.summary_interval == 0
        if self._summary_step_count > 1 and at_boundary:
            self._finalize_window_summary()

        if (
            self.debug_summary_interval > 0
            and self._debug_summary_step_count > 0
            and self._debug_summary_step_count % self.debug_summary_interval == 0
        ):
            self._generate_debug_summary()

        next_step = self.prompt_builder.step_count + 1
        if next_step > 1 and (next_step - 1) % self.prompt_builder.context_window_size == 0:
            self._messages = []

    def _build_prompt(self, obs: AgentObservation, inventory: dict[str, int]) -> str:
        """Build the full prompt with all context additions."""
        user_prompt, includes_basic_info = self.prompt_builder.context_prompt(obs)

        if includes_basic_info and self._history_summaries:
            user_prompt = self._get_history_summary_text() + "\n" + user_prompt

        additions = [
            self.exploration.get_discovered_objects_text(),
            self.exploration.get_other_agents_text(self.prompt_builder.step_count),
            self._get_strategic_hints(inventory, obs),
            self.prompt_builder.get_pathfinding_hints(obs),
        ]
        for addition in additions:
            if addition:
                user_prompt = user_prompt + "\n\n" + addition

        return user_prompt

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM and return the response text."""
        if self.verbose:
            print(f"\n[PROMPT Agent {self.agent_id}]\n{user_prompt}\n")

        messages = self._get_messages_for_api(user_prompt)

        self.conversation_history.append(
            {
                "step": len(self.conversation_history) + 1,
                "prompt": user_prompt,
                "num_messages": len(messages),
                "response": None,
            }
        )

        raw_response, input_tokens, output_tokens = self._call(messages)

        print(f"[LLM Agent {self.agent_id}] {raw_response}")

        action_name = raw_response.strip()
        self._add_to_messages("assistant", action_name)
        self.conversation_history[-1]["response"] = action_name
        self.cost_tracker.record_usage(input_tokens, output_tokens)

        return action_name

    def step(self, obs: AgentObservation) -> Action:
        """Get action from LLM given observation."""
        inventory = self._update_tracking(obs)
        self._handle_boundaries()
        user_prompt = self._build_prompt(obs, inventory)

        try:
            action_name = self._call_llm(user_prompt)
            parsed_action, reasoning = parse_action(action_name, self.policy_env_info.actions.actions())
            self._add_action_to_window(parsed_action.name, reasoning)
            self.last_action = parsed_action.name
            return parsed_action
        except Exception as e:
            print(f"[ERROR] LLM API error: {e}. Falling back to random action.")
            fallback_action = random.choice(self.policy_env_info.actions.actions())
            self._add_action_to_window(fallback_action.name, "API error fallback")
            self.last_action = fallback_action.name
            return fallback_action

    def print_conversation_history(self) -> None:
        """Print all LLM prompts and responses from this episode."""
        if not self.conversation_history:
            print("\n" + "=" * 70)
            print("No conversation history recorded.")
            print("=" * 70 + "\n")
            return

        print("\n" + "=" * 70)
        print(f"LLM CONVERSATION HISTORY ({len(self.conversation_history)} steps)")
        print("=" * 70 + "\n")

        for entry in self.conversation_history:
            step = entry["step"]
            print(f"{'=' * 70}")
            print(f"STEP {step}")
            print(f"{'=' * 70}")

            # Print system message if present (static prompts only)
            if "system" in entry:
                print("\n[SYSTEM MESSAGE]")
                print(entry["system"])
                print()

            # Print user prompt
            print("[USER PROMPT]")
            print(entry["prompt"])
            print()

            # Print LLM response
            print("[LLM RESPONSE]")
            print(entry.get("response", "(no response)"))
            print()

        print("=" * 70 + "\n")
