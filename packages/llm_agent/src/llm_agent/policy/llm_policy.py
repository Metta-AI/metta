"""LLM-based policy for MettaGrid using GPT or Claude."""

import atexit
import logging
import os
import random
import sys
from typing import Literal

from llm_agent.action_parser import parse_action
from llm_agent.cost_tracker import CostTracker
from llm_agent.exploration_tracker import ExplorationTracker
from llm_agent.model_config import validate_model_context
from llm_agent.observation_debugger import ObservationDebugger
from llm_agent.providers import (
    ensure_ollama_model,
    select_anthropic_model,
    select_openai_model,
)
from llm_agent.utils import pos_to_dir
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

logger = logging.getLogger(__name__)


class LLMAgentPolicy(AgentPolicy):
    """Per-agent LLM policy that queries GPT or Claude for action selection."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg = None,
        agent_id: int = 0,
    ):
        """Initialize LLM agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults: gpt-4o-mini, claude-3-5-sonnet-20240620, or llama3.2 for ollama)
            temperature: Sampling temperature for LLM
            debug_mode: If True, print human-readable observation debug info (default: True)
            context_window_size: Number of steps before resending basic info (default: 20)
            summary_interval: Number of steps between history summaries (default: 5)
            debug_summary_interval: Steps between LLM debug summaries written to file (0=disabled, e.g., 100)
            mg_cfg: Optional MettaGridConfig for extracting game-specific info (chest vibes, etc.)
            agent_id: Agent ID for this policy instance (used for debug output filtering)
        """
        super().__init__(policy_env_info)
        self.provider = provider
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.agent_id = agent_id
        self.last_action: str | None = None
        self.summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval
        self.debug_summary_interval = int(debug_summary_interval) if isinstance(debug_summary_interval, str) else debug_summary_interval

        # Debug summary tracking (for long runs)
        self._debug_summary_step_count = 0
        self._debug_summary_actions: list[dict[str, str]] = []  # Actions since last debug summary
        self._debug_summary_file: str | None = None

        # Track conversation history for debugging
        self.conversation_history: list[dict] = []

        # Stateful conversation messages (for multi-turn conversations with LLM)
        # Format: [{"role": "user"/"assistant", "content": "..."}, ...]
        self._messages: list[dict[str, str]] = []

        # History summaries - one summary per summary_interval (up to 100)
        # Each summary captures what the agent thought and did in that interval
        self._history_summaries: list[str] = []
        self._max_history_summaries = 100

        # Track actions within current summary interval for summarization
        self._current_window_actions: list[dict[str, str]] = []

        # Step counter for summary intervals (separate from context window)
        self._summary_step_count = 0

        # Exploration tracker handles position, discovery, and other agent tracking
        self.exploration = ExplorationTracker(policy_env_info)

        # Cost tracker (singleton - shared across all policy instances)
        self.cost_tracker = CostTracker()

        # Track exploration direction and steps in that direction
        self._current_direction: str | None = None
        self._steps_in_direction: int = 0
        self._direction_change_threshold: int = 8  # Change direction after this many steps

        # Initialize prompt builder
        from llm_agent.policy.prompt_builder import LLMPromptBuilder

        self.prompt_builder = LLMPromptBuilder(
            policy_env_info=policy_env_info,
            context_window_size=context_window_size,
            mg_cfg=mg_cfg,
            debug_mode=debug_mode,
            agent_id=agent_id,
        )
        if self.debug_mode:
            logger.info(f"Using dynamic prompts with context window size: {context_window_size}")

        # Debug: Check if AssemblerDrawsFromChestsVariant is active
        self._assembler_draws_from_chests = False
        if mg_cfg is not None:
            assembler_cfg = mg_cfg.game.objects.get("assembler")
            if assembler_cfg is not None and hasattr(assembler_cfg, "chest_search_distance"):
                chest_dist = assembler_cfg.chest_search_distance
                if chest_dist > 0:
                    self._assembler_draws_from_chests = True
                    print(f"[DEBUG AssemblerDrawsFromChestsVariant] ACTIVE - chest_search_distance={chest_dist}")
                else:
                    print(f"[DEBUG AssemblerDrawsFromChestsVariant] NOT ACTIVE - chest_search_distance={chest_dist}")
            else:
                print("[DEBUG AssemblerDrawsFromChestsVariant] NOT ACTIVE - no assembler config found")
        else:
            print("[DEBUG AssemblerDrawsFromChestsVariant] NOT ACTIVE - no mg_cfg provided")

        # Initialize observation debugger if debug mode is enabled
        if self.debug_mode:
            self.debugger = ObservationDebugger(policy_env_info)
        else:
            self.debugger = None


        # Initialize LLM client
        # Note: API key validation is handled by LLMMultiAgentPolicy before creating agent policies
        if self.provider == "openai":
            from openai import OpenAI

            self.client: OpenAI | None = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.anthropic_client = None
            self.ollama_client = None
            self.model = model if model else select_openai_model()
        elif self.provider == "anthropic":
            from anthropic import Anthropic

            self.client = None
            self.anthropic_client: Anthropic | None = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.ollama_client = None
            self.model = model if model else select_anthropic_model()
        elif self.provider == "ollama":
            from openai import OpenAI

            self.client = None
            self.anthropic_client = None

            # Ensure Ollama is available and model is pulled (or select if not provided)
            self.model = ensure_ollama_model(model)

            self.ollama_client: OpenAI | None = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # Ollama doesn't need a real API key
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

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
                self._messages = [self._messages[0]] + self._messages[-(max_messages - 1):]
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

    def _should_show(self, component: str) -> bool:
        """Check if a debug component should be shown.

        Only shows debug output for agent 0 to avoid cluttering the console.

        Args:
            component: Component name to check (e.g., "prompt", "llm", "grid")

        Returns:
            True if component should be shown
        """
        # Only show debug for agent 0
        if self.agent_id != 0:
            return False

        # Handle boolean debug_mode
        if isinstance(self.debug_mode, bool):
            return self.debug_mode
        # Handle set debug_mode
        if isinstance(self.debug_mode, set):
            return "all" in self.debug_mode or component in self.debug_mode
        return False

    def _summarize_current_window(self) -> str:
        """Summarize the current context window's actions into a compact summary.

        Returns:
            A compact string summarizing what the agent did in this window.
        """
        if not self._current_window_actions:
            return ""

        # Get summary info from exploration tracker
        summary_info = self.exploration.get_summary_info()
        window_positions = summary_info["window_positions"]

        # Get start and end positions for this window
        if window_positions:
            start_pos = window_positions[0]
            end_pos = window_positions[-1]
        else:
            start_pos = (0, 0)
            end_pos = (self.exploration.global_x, self.exploration.global_y)

        # Build summary with position info
        window_num = len(self._history_summaries) + 1

        # Get unique positions visited this window (excluding duplicates)
        unique_positions = []
        seen = set()
        for pos in window_positions:
            if pos not in seen:
                unique_positions.append(pos)
                seen.add(pos)

        # Calculate net direction of movement this window
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        net_direction = []
        if dy < 0:
            net_direction.append("North")
        elif dy > 0:
            net_direction.append("South")
        if dx > 0:
            net_direction.append("East")
        elif dx < 0:
            net_direction.append("West")
        net_dir_str = "-".join(net_direction) if net_direction else "stationary"

        summary = f"[Window {window_num}] {pos_to_dir(*start_pos)} → {pos_to_dir(*end_pos)} (heading {net_dir_str})"
        summary += f" | {len(unique_positions)} new spots, {summary_info['total_explored']} total"

        return summary

    def _add_action_to_window(self, action: str, reasoning: str = "") -> None:
        """Track an action for the current context window summary.

        Args:
            action: The action taken
            reasoning: The reasoning behind the action (from LLM response)
        """
        action_info = {"action": action, "reasoning": reasoning}
        self._current_window_actions.append(action_info)

        # Also track for debug summary
        if self.debug_summary_interval > 0:
            self._debug_summary_actions.append(action_info)

        # Track direction changes for exploration strategy
        direction_map = {
            "move_north": "north",
            "move_south": "south",
            "move_east": "east",
            "move_west": "west",
        }

        if action in direction_map:
            new_direction = direction_map[action]
            if new_direction == self._current_direction:
                self._steps_in_direction += 1
            else:
                self._current_direction = new_direction
                self._steps_in_direction = 1

        # Delegate position tracking to exploration tracker
        self.exploration.update_position(action)

    def _finalize_window_summary(self) -> None:
        """Create summary for current window and reset for next window."""
        if self._current_window_actions:
            summary = self._summarize_current_window()
            if summary:
                self._history_summaries.append(summary)
                # Prune to max size
                if len(self._history_summaries) > self._max_history_summaries:
                    self._history_summaries = self._history_summaries[-self._max_history_summaries:]

                # Always print history summary at window boundary
                print(f"\n[HISTORY Agent {self.agent_id}] {summary}\n")

        # Reset for next window
        self._current_window_actions = []
        # Reset exploration tracker window positions
        self.exploration.reset_window_positions()

    def _generate_debug_summary(self) -> None:
        """Generate an LLM debug summary for the last N steps and write to file.

        This creates a concise summary of what happened in the last debug_summary_interval
        steps, useful for debugging long runs without reading through all actions.
        """
        if not self._debug_summary_actions:
            return

        # Build a compact representation of recent actions
        action_counts: dict[str, int] = {}
        for action_info in self._debug_summary_actions:
            action = action_info.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

        # Format action summary
        action_summary = ", ".join(f"{k}:{v}" for k, v in sorted(action_counts.items()))

        # Get current state summary from exploration tracker
        summary_info = self.exploration.get_summary_info()
        inventory = self._last_inventory
        inv_str = ", ".join(f"{k}={v}" for k, v in inventory.items() if v > 0 and k != "energy")
        energy = inventory.get("energy", 0)

        # Create the prompt for LLM to summarize
        summary_prompt = f"""Summarize what Agent {self.agent_id} did in the last {len(self._debug_summary_actions)} steps.

Actions taken: {action_summary}
Current position: ({summary_info['global_x']}, {summary_info['global_y']}) from origin
Explored tiles: {summary_info['total_explored']}
Current inventory: {inv_str if inv_str else 'empty'}
Energy: {energy}
Discovered objects: {', '.join(summary_info['discovered_objects']) if summary_info['discovered_objects'] else 'none'}

Write a 2-3 sentence summary of progress, challenges, and current strategy. Be concise."""

        try:
            # Use the same LLM to generate summary
            summary_text = ""
            if self.provider == "openai" and self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=150,
                    temperature=0.3,
                )
                summary_text = response.choices[0].message.content or "No summary generated"
            elif self.provider == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=150,
                    temperature=0.3,
                )
                from anthropic.types import TextBlock
                for block in response.content:
                    if isinstance(block, TextBlock):
                        summary_text = block.text
                        break
            elif self.provider == "ollama" and self.ollama_client:
                response = self.ollama_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=150,
                    temperature=0.3,
                )
                summary_text = response.choices[0].message.content or "No summary generated"

            # Write to file
            if self._debug_summary_file is None:
                self._debug_summary_file = f"llm_debug_summary_agent{self.agent_id}.log"

            interval_num = self._debug_summary_step_count // self.debug_summary_interval
            with open(self._debug_summary_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"[Agent {self.agent_id}] Steps {(interval_num-1)*self.debug_summary_interval + 1}-{interval_num*self.debug_summary_interval}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Position: ({summary_info['global_x']}, {summary_info['global_y']}) | Explored: {summary_info['total_explored']} tiles\n")
                f.write(f"Inventory: {inv_str if inv_str else 'empty'} | Energy: {energy}\n")
                f.write(f"Actions: {action_summary}\n")
                f.write(f"\nSummary: {summary_text}\n")

            # Also print to console
            print(f"\n[DEBUG SUMMARY Agent {self.agent_id}] Steps {(interval_num-1)*self.debug_summary_interval + 1}-{interval_num*self.debug_summary_interval}: {summary_text}\n")

        except Exception as e:
            logger.warning(f"Failed to generate debug summary: {e}")

        # Reset for next interval
        self._debug_summary_actions = []

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
        from pathlib import Path
        template_path = Path(__file__).parent / "prompts" / "exploration_history.md"
        template = template_path.read_text()

        summary_info = self.exploration.get_summary_info()
        return (
            template
            .replace("{{CURRENT_POSITION}}", pos_to_dir(summary_info["global_x"], summary_info["global_y"], verbose=True))
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

    def _get_strategic_hints(
        self, inventory: dict[str, int], obs: AgentObservation | None = None
    ) -> str:
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
                    f"🎯 VISIBLE EXTRACTOR YOU NEED: {ext_list} - "
                    "PURSUE IT NOW! Navigate around walls if blocked!"
                )

        # Energy warning
        energy = inventory.get("energy", 100)
        if energy < 20:
            hints.append("⚠️ ENERGY CRITICAL (<20): Find charger IMMEDIATELY!")
        elif energy < 40:
            hints.append("⚠️ ENERGY LOW (<40): Head to charger soon!")

        # Direction change suggestion
        if self._steps_in_direction >= self._direction_change_threshold:
            opposite = {
                "north": "south", "south": "north",
                "east": "west", "west": "east"
            }
            suggested = opposite.get(self._current_direction, "different")
            hints.append(
                f"⚠️ You've gone {self._current_direction} for {self._steps_in_direction} steps. "
                f"Consider going {suggested}!"
            )

        # Distance from origin warning
        distance = abs(self.exploration.global_x) + abs(self.exploration.global_y)
        if distance > 25:
            hints.append(
                f"⚠️ You're {distance} tiles from origin. "
                "Extractors are usually within 20 tiles - try going back!"
            )

        # Resource gathering hints
        carbon = inventory.get("carbon", 0)
        oxygen = inventory.get("oxygen", 0)
        germanium = inventory.get("germanium", 0)
        silicon = inventory.get("silicon", 0)
        heart = inventory.get("heart", 0)

        if heart > 0 and "chest" in self.exploration.discovered_objects:
            hints.append(f"💡 You have {heart} heart(s)! Go to chest to deposit for reward!")
        elif carbon >= req_carbon and oxygen >= req_oxygen and germanium >= req_germanium and silicon >= req_silicon:
            hints.append("💡 You have all resources for a heart! Find an assembler and use heart_a vibe!")
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
                hints.append(f"📋 Still need: {', '.join(missing)}")

        if not hints:
            return ""

        return "=== STRATEGIC HINTS ===\n" + "\n".join(hints)

    def step(self, obs: AgentObservation) -> Action:
        """Get action from LLM given observation.

        Args:
            obs: Agent observation

        Returns:
            Action to take
        """
        # Extract and track discovered objects from this observation
        self.exploration.extract_discovered_objects(obs)

        # Extract other agents' positions and inventories
        self.exploration.extract_other_agents_info(obs, self.prompt_builder.step_count)

        # Extract current inventory for strategic hints
        inventory = self._extract_inventory_from_obs(obs)

        # Track resource collection before updating last inventory
        self.exploration.update_extractor_collection(inventory)
        self._last_inventory = inventory

        # Increment summary step counter
        self._summary_step_count += 1

        # Increment debug summary step counter
        self._debug_summary_step_count += 1

        # Check if we're at a summary interval boundary (every N steps)
        at_boundary = (self._summary_step_count - 1) % self.summary_interval == 0
        is_summary_boundary = self._summary_step_count > 1 and at_boundary

        # Check if we're at a debug summary interval boundary
        is_debug_summary_boundary = (
            self.debug_summary_interval > 0 and
            self._debug_summary_step_count > 0 and
            self._debug_summary_step_count % self.debug_summary_interval == 0
        )

        # Check if we're about to start a new context window (before incrementing step counter)
        next_step = self.prompt_builder.step_count + 1
        is_window_boundary = next_step > 1 and (next_step - 1) % self.prompt_builder.context_window_size == 0

        # At summary boundary, finalize the current interval's summary
        if is_summary_boundary:
            self._finalize_window_summary()

        # At debug summary boundary, generate LLM debug summary
        if is_debug_summary_boundary:
            self._generate_debug_summary()

        # At context window boundary, also clear conversation messages for fresh window
        if is_window_boundary:
            self._messages = []

        user_prompt, includes_basic_info = self.prompt_builder.context_prompt(obs)

        # Prepend history summaries to prompts that include basic info
        if includes_basic_info and self._history_summaries:
            history_text = self._get_history_summary_text()
            user_prompt = history_text + "\n" + user_prompt

        # Add discovered objects, other agents info, strategic hints, and pathfinding to every prompt
        discovered_text = self.exploration.get_discovered_objects_text()
        other_agents_text = self.exploration.get_other_agents_text(self.prompt_builder.step_count)
        strategic_hints = self._get_strategic_hints(inventory, obs)
        pathfinding_hints = self.prompt_builder.get_pathfinding_hints(obs)

        if discovered_text:
            user_prompt = user_prompt + "\n\n" + discovered_text
        if other_agents_text:
            user_prompt = user_prompt + "\n\n" + other_agents_text
        if strategic_hints:
            user_prompt = user_prompt + "\n\n" + strategic_hints
        if pathfinding_hints:
            user_prompt = user_prompt + "\n\n" + pathfinding_hints

        # Query LLM
        try:
            action_name = "noop"  # Default fallback

            if self.provider == "openai":
                assert self.client is not None
                # GPT-5 and o1 models use different parameters and don't support system messages
                is_gpt5_or_o1 = self.model.startswith("gpt-5") or self.model.startswith("o1")

                # Use stateful conversation history
                messages = self._get_messages_for_api(user_prompt)

                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": 150 if is_gpt5_or_o1 else None,
                    "max_tokens": None if is_gpt5_or_o1 else 150,
                    "temperature": None if is_gpt5_or_o1 else self.temperature,
                }
                # Remove None values
                completion_params = {k: v for k, v in completion_params.items() if v is not None}

                # Track prompt
                self.conversation_history.append({
                    "step": len(self.conversation_history) + 1,
                    "prompt": user_prompt,
                    "num_messages": len(messages),
                    "response": None,  # Will be filled in below
                })

                response = self.client.chat.completions.create(**completion_params)
                raw_response = response.choices[0].message.content
                if raw_response is None:
                    raw_response = "noop"

                # Always print LLM response with agent ID
                print(f"[LLM Agent {self.agent_id}] {raw_response}")

                action_name = raw_response.strip()

                # Add assistant response to stateful conversation history
                self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track token usage
                usage = response.usage
                if usage:
                    self.cost_tracker.record_usage(usage.prompt_tokens, usage.completion_tokens)

                    if self.debug_mode:
                        logger.debug(
                            f"OpenAI response: '{action_name}' | "
                            f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out"
                        )

            elif self.provider == "ollama":
                assert self.ollama_client is not None

                # Use stateful conversation history
                messages = self._get_messages_for_api(user_prompt)

                # Track prompt
                self.conversation_history.append({
                    "step": len(self.conversation_history) + 1,
                    "prompt": user_prompt,
                    "num_messages": len(messages),
                    "response": None,
                })

                response = self.ollama_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=150,
                )

                if self.debug_mode:
                    # Debug: log the raw response
                    logger.debug(f"Ollama response object: {response}")
                    logger.debug(f"Ollama response choices: {response.choices}")

                # Some models (like gpt-oss) put output in 'reasoning' field instead of 'content'
                message = response.choices[0].message
                raw_response = message.content or ""

                # Check reasoning field if content is empty
                if not raw_response and hasattr(message, "reasoning") and message.reasoning:
                    if self.debug_mode:
                        logger.warning(f"Model used reasoning field instead of content: {message.reasoning[:100]}...")
                    # Try to extract action from reasoning (take last line or last word)
                    raw_response = message.reasoning

                if not raw_response:
                    if self.debug_mode:
                        reasoning = getattr(message, "reasoning", None)
                        logger.error(
                            f"Ollama returned empty response! content='{message.content}', reasoning='{reasoning}'"
                        )
                    raw_response = "noop"

                # Always print LLM response with agent ID
                print(f"[LLM Agent {self.agent_id}] {raw_response}")

                action_name = raw_response.strip()

                # Add assistant response to stateful conversation history
                self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track token usage
                usage = response.usage
                if usage:
                    self.cost_tracker.record_usage(usage.prompt_tokens, usage.completion_tokens)

                    if self.debug_mode:
                        logger.debug(
                            f"Ollama response: '{action_name}' | "
                            f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out"
                        )

            elif self.provider == "anthropic":
                assert self.anthropic_client is not None

                # Use stateful conversation history
                messages = self._get_messages_for_api(user_prompt)

                # Track prompt
                self.conversation_history.append({
                    "step": len(self.conversation_history) + 1,
                    "prompt": user_prompt,
                    "num_messages": len(messages),
                    "response": None,
                })

                response = self.anthropic_client.messages.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=150,
                )

                # Extract text from response content blocks
                from anthropic.types import TextBlock

                raw_response = "noop"
                for block in response.content:
                    if isinstance(block, TextBlock):
                        raw_response = block.text
                        break

                # Always print LLM response with agent ID
                print(f"[LLM Agent {self.agent_id}] {raw_response}")

                action_name = raw_response.strip()

                # Add assistant response to stateful conversation history
                self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track token usage
                usage = response.usage
                self.cost_tracker.record_usage(usage.input_tokens, usage.output_tokens)

                if self.debug_mode:
                    logger.debug(
                        f"Anthropic response: '{action_name}' | "
                        f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out"
                    )

            # Parse and return action
            parsed_action, reasoning = parse_action(action_name, self.policy_env_info.actions.actions())

            # Track action for history summary (with reasoning if available)
            self._add_action_to_window(parsed_action.name, reasoning)

            # Track last action for debug output
            self.last_action = parsed_action.name

            return parsed_action

        except Exception as e:
            logger.error(f"LLM API error: {e}. Falling back to random action.")
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


class LLMMultiAgentPolicy(MultiAgentPolicy):
    """LLM-based multi-agent policy for MettaGrid."""

    short_names = ["llm"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg = None,
    ):
        """Initialize LLM multi-agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature for LLM
            debug_mode: If True, print human-readable observation debug info (default: True)
            context_window_size: Number of steps before resending basic info (default: 20)
            summary_interval: Number of steps between history summaries (default: 5)
            debug_summary_interval: Steps between LLM debug summaries written to file (0=disabled, e.g., 100)
            mg_cfg: Optional MettaGridConfig for extracting game-specific info (chest vibes, etc.)
        """
        super().__init__(policy_env_info)
        self.provider: Literal["openai", "anthropic", "ollama"] = provider
        self.temperature = temperature
        self.cost_tracker = CostTracker()  # Singleton - shared across all policy instances
        # Handle string "true"/"false" from CLI kwargs
        if isinstance(debug_mode, str):
            self.debug_mode = debug_mode.lower() not in ("false", "0", "no", "")
        else:
            self.debug_mode = bool(debug_mode)
        self.context_window_size = context_window_size
        self.summary_interval = int(summary_interval) if isinstance(summary_interval, str) else summary_interval
        self.debug_summary_interval = int(debug_summary_interval) if isinstance(debug_summary_interval, str) else debug_summary_interval
        self.mg_cfg = mg_cfg

        # Check API key before model selection for paid providers
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            print(
                "\n\033[1;31mError:\033[0m OPENAI_API_KEY environment variable is not set.\n\n"
                "To use OpenAI GPT models, you need to:\n"
                "  1. Get an API key from https://platform.openai.com/api-keys\n"
                "  2. Export it in your terminal:\n"
                "     export OPENAI_API_KEY='your-api-key-here'\n\n"
                "Alternatively, use local Ollama (free):\n"
                "  cogames play -m <mission> -p class=llm-ollama\n"
            )
            sys.exit(1)
        elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            print(
                "\n\033[1;31mError:\033[0m ANTHROPIC_API_KEY environment variable is not set.\n\n"
                "To use Anthropic Claude models, you need to:\n"
                "  1. Get an API key from https://console.anthropic.com/settings/keys\n"
                "  2. Export it in your terminal:\n"
                "     export ANTHROPIC_API_KEY='your-api-key-here'\n\n"
                "Alternatively, use local Ollama (free):\n"
                "  cogames play -m <mission> -p class=llm-ollama\n"
            )
            sys.exit(1)

        # Select model once for all agents if not specified
        if model is None:
            if provider == "openai":
                self.model = select_openai_model()
            elif provider == "anthropic":
                self.model = select_anthropic_model()
            elif provider == "ollama":
                self.model = ensure_ollama_model(None)
            else:
                self.model = None
        else:
            self.model = model

        # Validate model context window is sufficient for the config
        if self.model:
            validate_model_context(
                model=self.model,
                context_window_size=self.context_window_size,
                summary_interval=self.summary_interval,
            )

        # Register atexit handler to print costs when program ends (for paid APIs only)
        if provider in ("openai", "anthropic") and not hasattr(LLMMultiAgentPolicy, '_atexit_registered'):
            atexit.register(self.cost_tracker.print_summary)
            LLMMultiAgentPolicy._atexit_registered = True

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
            temperature=self.temperature,
            debug_mode=self.debug_mode,
            context_window_size=self.context_window_size,
            summary_interval=self.summary_interval,
            debug_summary_interval=self.debug_summary_interval,
            mg_cfg=self.mg_cfg,
            agent_id=agent_id,
        )



class LLMGPTMultiAgentPolicy(LLMMultiAgentPolicy):
    """OpenAI GPT-based policy for MettaGrid."""

    short_names = ["llm-openai"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="openai",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
            debug_summary_interval=debug_summary_interval,
            mg_cfg=mg_cfg,
        )


class LLMClaudeMultiAgentPolicy(LLMMultiAgentPolicy):
    """Anthropic Claude-based policy for MettaGrid."""

    short_names = ["llm-anthropic"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="anthropic",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
            debug_summary_interval=debug_summary_interval,
            mg_cfg=mg_cfg,
        )


class LLMOllamaMultiAgentPolicy(LLMMultiAgentPolicy):
    """Ollama local LLM-based policy for MettaGrid."""

    short_names = ["llm-ollama"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        context_window_size: int = 20,
        summary_interval: int = 5,
        debug_summary_interval: int = 0,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="ollama",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            context_window_size=context_window_size,
            summary_interval=summary_interval,
            debug_summary_interval=debug_summary_interval,
            mg_cfg=mg_cfg,
        )
