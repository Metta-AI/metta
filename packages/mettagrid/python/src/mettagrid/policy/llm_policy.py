"""LLM-based policy for MettaGrid using GPT or Claude."""

import atexit
import json
import logging
import os
import random
import subprocess
import sys
from typing import Literal

from mettagrid.policy.observation_debugger import ObservationDebugger
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

logger = logging.getLogger(__name__)


def calculate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM API call based on model and token usage.

    Prices are per 1M tokens as of January 2025.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    # OpenAI pricing (per 1M tokens)
    openai_prices = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-5.1": {"input": 5.00, "output": 15.00},
    }

    # Anthropic pricing (per 1M tokens)
    anthropic_prices = {
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    }

    # Combine all pricing
    all_prices = {**openai_prices, **anthropic_prices}

    # Get pricing for the model
    if model not in all_prices:
        logger.warning(f"Unknown model '{model}' for cost calculation. Using default pricing.")
        # Default to GPT-4o-mini pricing if model is unknown
        prices = openai_prices["gpt-4o-mini"]
    else:
        prices = all_prices[model]

    # Calculate cost (prices are per 1M tokens, so divide by 1,000,000)
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]

    return input_cost + output_cost


def _print_cost_summary_on_exit() -> None:
    """Print cost summary when program exits or is interrupted."""
    # Access the class through globals to avoid circular import
    try:
        llm_policy_class = globals().get("LLMAgentPolicy")
        if llm_policy_class is None:
            return

        summary = llm_policy_class.get_cost_summary()
        if summary["total_calls"] > 0:
            print("\n" + "=" * 60)
            print("LLM API USAGE SUMMARY")
            print("=" * 60)
            print(f"Total API calls: {summary['total_calls']}")
            print(f"Total tokens: {summary['total_tokens']:,}")
            print(f"  - Input tokens: {summary['total_input_tokens']:,}")
            print(f"  - Output tokens: {summary['total_output_tokens']:,}")
            print(f"Total cost: ${summary['total_cost']:.4f}")
            print("=" * 60 + "\n")
    except Exception:
        # Silently fail if there's any issue printing the summary
        pass

def check_ollama_available() -> bool:
    """Check if Ollama server is running.

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # Try to list models as a health check
        client.models.list()
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """List available Ollama models.

    Returns:
        List of model names
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        # Parse output: skip header line, extract model names
        lines = result.stdout.strip().split("\n")[1:]  # Skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return []


def get_openai_models() -> list[tuple[str, str]]:
    """Get tested working OpenAI models.

    Returns:
        List of (model_name, description) tuples
    """
    return [
        ("gpt-4o-mini", "Cheapest - Fast and cost-effective"),
        ("gpt-4o", "Capable - Best GPT-4 model"),
        ("gpt-5.1", "Best - Latest GPT-5 for complex reasoning"),
    ]


def select_openai_model() -> str:
    """Prompt user to select an OpenAI model.

    Returns:
        Selected model name
    """
    models = get_openai_models()

    print("\n" + "=" * 60)
    print("Select OpenAI Model:")
    print("=" * 60)
    for idx, (model_name, description) in enumerate(models, 1):
        print(f"  [{idx}] {model_name}")
        print(f"      {description}")
    print("=" * 60)

    while True:
        try:
            selection = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                model = models[idx][0]
                print(f"\n✓ Selected: {model}\n")
                return model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  No model selected. Exiting.\n")
            sys.exit(0)


def get_anthropic_models() -> list[tuple[str, str]]:
    """Get tested working Anthropic Claude models.

    Returns:
        List of (model_name, description) tuples
    """
    return [
        ("claude-haiku-4-5", "Cheapest - Fastest with near-frontier intelligence"),
        ("claude-sonnet-4-5", "Best - Smartest for complex agents & coding"),
        ("claude-opus-4-5", "Premium - Maximum intelligence & performance"),
    ]


def select_anthropic_model() -> str:
    """Prompt user to select an Anthropic Claude model.

    Returns:
        Selected model name
    """
    models = get_anthropic_models()

    print("\n" + "=" * 60)
    print("Select Claude Model:")
    print("=" * 60)
    for idx, (model_name, description) in enumerate(models, 1):
        print(f"  [{idx}] {model_name}")
        print(f"      {description}")
    print("=" * 60)

    while True:
        try:
            selection = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                model = models[idx][0]
                print(f"\n✓ Selected: {model}\n")
                return model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  No model selected. Exiting.\n")
            sys.exit(0)


def ensure_ollama_model(model: str | None = None) -> str:
    """Ensure an Ollama model is available, pulling if necessary.

    Args:
        model: Model name to check/pull, or None to prompt user to select

    Returns:
        The model name that is available

    Raises:
        RuntimeError: If Ollama is not available or model pull fails
    """
    if not check_ollama_available():
        raise RuntimeError(
            "Ollama server is not running. Please start it with 'ollama serve' or install from https://ollama.ai"
        )

    available_models = list_ollama_models()

    # If no model specified, prompt user to select
    if model is None:
        if not available_models:
            # No models available, prompt user
            print("\n" + "=" * 60)
            print("⚠️  No Ollama models found!")
            print("=" * 60)
            print("\nOptions:")
            print("  1. Install default model (llama3.2) - ~2GB download")
            print("  2. Install a model manually with 'ollama pull <model>'")
            print("  3. Use llm-claude or llm-gpt instead")
            print("=" * 60)

            try:
                response = input("\nInstall default model (llama3.2)? [y/N]: ").strip().lower()
                if response in ("y", "yes"):
                    model = "llama3.2"
                    print(f"\n📥 Pulling {model}...")
                    print("(This may take a few minutes...)\n")
                    subprocess.run(["ollama", "pull", model], check=True)
                    print(f"\n✓ Successfully installed {model}\n")
                    return model
                else:
                    print("\n" + "=" * 60)
                    print("To use Ollama:")
                    print("  1. Pull a model: ollama pull llama3.2")
                    print("  2. Run again: cogames play -m <mission> -p llm-ollama")
                    print("\nAlternatively, use cloud LLMs:")
                    print("  • cogames play -m <mission> -p llm-gpt")
                    print("  • cogames play -m <mission> -p llm-claude")
                    print("=" * 60 + "\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  Cancelled by user.\n")
                sys.exit(0)

        # Show available models and prompt user to select
        print("\n" + "=" * 60)
        print("Available Ollama Models:")
        print("=" * 60)
        for idx, model_name in enumerate(available_models, 1):
            print(f"  [{idx}] {model_name}")
        print("=" * 60)

        while True:
            try:
                selection = input(f"\nSelect a model (1-{len(available_models)}): ").strip()
                idx = int(selection) - 1
                if 0 <= idx < len(available_models):
                    model = available_models[idx]
                    print(f"\n✓ Selected: {model}\n")
                    return model
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  No model selected. Exiting.\n")
                sys.exit(0)

    # Model was explicitly specified, check if it's available
    if any(model in m for m in available_models):
        return model

    # Try to pull the specified model
    print(f"\nModel '{model}' not found. Pulling from Ollama...")
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
            capture_output=False,  # Show progress
        )
        print(f"\n✓ Successfully pulled model: {model}\n")
        return model
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}") from e


def build_game_rules_prompt(policy_env_info: PolicyEnvInterface) -> str:
    """Build comprehensive game rules prompt with observation format guide.

    Args:
        policy_env_info: Policy environment interface with feature specs

    Returns:
        Complete system prompt for LLM
    """
    # Build feature ID reference
    feature_docs = []
    for feature in policy_env_info.obs_features:
        feature_docs.append(f"  {feature.id}: '{feature.name}' (normalization: {feature.normalization})")

    # Build tag ID reference
    tag_docs = []
    for tag_id, tag_name in enumerate(policy_env_info.tags):
        tag_docs.append(f"  {tag_id}: '{tag_name}'")

    # Build action reference
    action_docs = []
    for action_id, action_name in enumerate(policy_env_info.action_names):
        action_docs.append(f"  {action_id}: '{action_name}'")

    obs_width = policy_env_info.obs_width
    obs_height = policy_env_info.obs_height
    agent_x = obs_width // 2
    agent_y = obs_height // 2

    prompt = f"""You are playing MettaGrid, a multi-agent gridworld game.

=== OBSERVATION FORMAT ===

You receive observations as a list of tokens. Each token has:
- "feature": Feature name (see Feature Reference below)
- "location": {{"x": col, "y": row}} coordinates
- "value": Feature value

COORDINATE SYSTEM:
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

UNDERSTANDING TOKENS:
1. Tokens at YOUR location (x={agent_x}, y={agent_y}) describe YOUR state (inventory, frozen status, etc.)
2. Tokens at OTHER locations describe objects/agents you can see
3. Multiple tokens at the SAME location = same object with multiple properties
4. "tag" feature tells you what type of object it is (see Tag Reference)

SEMANTIC MEANING OF FEATURES:

AGENT STATE FEATURES (at your location x={agent_x}, y={agent_y}):

- "agent:group": Your team ID number
  Values: 0, 1, 2, 3, etc. (team numbers)
  → Same value as yours = ally, different value = enemy

- "agent:frozen": Whether you can act
  Values:
    0 = not frozen (can act normally)
    1 = frozen (cannot take actions)

- "vibe": Your current interaction state/mode
  Values: Integer representing current vibe state
  → Different vibes may enable different interactions with objects

- "agent:compass": Direction to nearest objective
  Values:
    0 = North
    1 = East
    2 = South
    3 = West

GLOBAL STATE FEATURES (no specific location):

- "episode_completion_pct": Progress through episode
  Values: 0-255 (0 = start, 255 = nearly done)

- "last_action": Your previous action
  Values (action IDs):
{chr(10).join(f"    {action_id} = {action_name}" for action_id, action_name in enumerate(policy_env_info.action_names))}

- "last_reward": Reward from last step
  Values: Positive = good, negative = bad, 0 = neutral

INVENTORY FEATURES (at your location):

- "inv:*": Resources and items you're carrying
  Features: {", ".join([f.name for f in policy_env_info.obs_features if f.name.startswith("inv:")])}
  Values: 0-255 (higher = more of that resource)
  → Value 0 or missing = you don't have any of that resource

OBJECT IDENTIFICATION (at other locations):

- "tag": Type of object at this location
  Values (tag IDs → object types):
{chr(10).join(f"    {tag_id} = {tag_name}" for tag_id, tag_name in enumerate(policy_env_info.tags))}
  → This is THE KEY feature for knowing what objects are

OBJECT STATE FEATURES (at other locations):

- "cooldown_remaining": Steps until object can be used again
  Values: 0-255 (0 = ready now, higher = must wait longer)

- "remaining_uses": Times object can still be used
  Values: 0-255 (0 = depleted, higher = more uses left)

- "clipped": Special clipped state
  Values: 0 = not clipped, 1 = clipped

PROTOCOL FEATURES (at assembler/extractor locations):

- "protocol_input:*": Resources this object REQUIRES
  Features: {", ".join([f.name for f in policy_env_info.obs_features if f.name.startswith("protocol_input:")])}
  Values: Amount of each resource needed
  → You must have these resources to use the object

- "protocol_output:*": Resources this object PRODUCES
  Features: {", ".join([f.name for f in policy_env_info.obs_features if f.name.startswith("protocol_output:")])}
  Values: Amount of each resource produced
  → What you'll get when you successfully use the object

OTHER AGENT FEATURES (at locations with "tag"=agent):
- "agent:group": Their team ID (compare to yours: same=ally, different=enemy)
- "inv:*": Their inventory (if visible, useful for threat assessment)

FEATURE REFERENCE:
{chr(10).join(feature_docs)}

TAG REFERENCE (for "tag" feature):
{chr(10).join(tag_docs)}

ACTION REFERENCE:
{chr(10).join(action_docs)}

=== GAME MECHANICS ===

OBJECTS YOU MIGHT SEE:
- altar: Use energy here to gain rewards (costs energy, has cooldown)
- converter: Convert resources to energy (no energy cost, has cooldown)
- generator: Harvest resources from here (has cooldown)
- wall: Impassable barrier - YOU CANNOT MOVE THROUGH WALLS
- agent: Other players in the game

KEY RULES:
- Energy is required for most actions
- Harvest resources from generators
- Convert resources to energy at converters
- Use altars to gain rewards (this is your main goal)
- Attacks freeze targets and steal their resources
- Shield protects you but drains energy
- YOU CANNOT MOVE INTO A TILE THAT HAS A WALL OR OBJECT

=== MOVEMENT LOGIC (CRITICAL) ===

From the working Nim agent implementation, here's how to determine if you can move:

WALKABILITY RULE:
- A tile is WALKABLE if it has NO tokens at that location
- A tile is BLOCKED if it has ANY of these:
  * "tag" feature (indicates an object: wall, extractor, chest, etc.)
  * "agent:group" feature (another agent is there)

BEFORE MOVING, CHECK THE TARGET LOCATION:
1. Look at the target coordinates (North/South/East/West from your position)
2. Check if ANY tokens exist at those coordinates
3. If tokens exist → BLOCKED, choose different direction
4. If no tokens exist → WALKABLE, safe to move

EXAMPLE WALKABILITY CHECK:
Your position: x={agent_x}, y={agent_y}
North tile: x={agent_x}, y={agent_y - 1}

To move North, check all tokens:
- If you see ANY token with location x={agent_x}, y={agent_y - 1} → DON'T move North
- If you see NO tokens at x={agent_x}, y={agent_y - 1} → SAFE to move North

GROUPING TOKENS BY LOCATION:
Multiple tokens at the same location = same object with multiple properties:
- Location (5, 5) with tag="wall" → Wall object
- Location (6, 4) with tag="agent" + "agent:group"=1 + "inv:energy"=50 → Enemy agent with 50 energy

=== DECISION-MAKING EXAMPLES ===

EXAMPLE 1 - Should I use this generator?
Tokens at location (6, 5):
- {{"feature": "tag", "location": {{"x": 6, "y": 5}}, "value": 2}}  # value 2 = "carbon_extractor" tag
- {{"feature": "cooldown_remaining", "location": {{"x": 6, "y": 5}}, "value": 0}}

Analysis:
→ It's a carbon_extractor (tag=2)
→ Cooldown is 0, so it's READY to use
→ DECISION: Move adjacent to (6,5) and use "use" action to harvest carbon

EXAMPLE 2 - Do I have enough resources?
Tokens at my location ({agent_x}, {agent_y}):
- {{"feature": "inv:carbon", "location": {{"x": {agent_x}, "y": {agent_y}}}, "value": 5}}
- {{"feature": "inv:energy", "location": {{"x": {agent_x}, "y": {agent_y}}}, "value": 20}}

Assembler at location (7, 6) needs:
- {{"feature": "protocol_input:carbon", "location": {{"x": 7, "y": 6}}, "value": 10}}

Analysis:
→ I have 5 carbon, but assembler needs 10 carbon
→ I don't have enough!
→ DECISION: Find a carbon generator first, harvest more carbon, THEN come back to assembler

EXAMPLE 3 - Is this agent friendly or hostile?
My tokens:
- {{"feature": "agent:group", "location": {{"x": {agent_x}, "y": {agent_y}}}, "value": 0}}  # I'm team 0

Other agent at (4, 3):
- {{"feature": "tag", "location": {{"x": 4, "y": 3}}, "value": 0}}  # value 0 = "agent" tag
- {{"feature": "agent:group", "location": {{"x": 4, "y": 3}}, "value": 1}}  # They're team 1

Analysis:
→ I'm team 0, they're team 1
→ Different teams = ENEMY
→ DECISION: Avoid them or prepare to attack if beneficial

EXAMPLE 4 - Can I move East?
Tokens in observation:
- {{"feature": "tag", "location": {{"x": {agent_x + 1}, "y": {agent_y}}}, "value": 8}}  # value 8 = "wall"

Analysis:
→ East is location ({agent_x + 1}, {agent_y})
→ There IS a token at ({agent_x + 1}, {agent_y}) - it's a wall
→ ANY token at a location = BLOCKED
→ DECISION: DON'T move East, try a different direction

=== STRATEGY TIPS ===

MOVEMENT:
1. ALWAYS check target location for tokens before moving
2. Empty locations (no tokens) = walkable
3. Any tokens at location = blocked/occupied
4. When stuck, try different cardinal directions

RESOURCE MANAGEMENT:
- Prioritize energy management
- Harvest resources from generators
- Convert resources to energy at converters
- Use altars when you have enough energy

Your goal is to maximize rewards by using the altar efficiently while managing your resources and energy.
"""
    return prompt


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
            "location": {"x": token.col(), "y": token.row()},
            "value": token.value,
        }
        tokens_list.append(token_dict)

    return {
        "agent_id": obs.agent_id,
        "visible_objects": tokens_list,
        "available_actions": policy_env_info.action_names,
        "num_visible_objects": len(tokens_list),
    }


class LLMAgentPolicy(AgentPolicy):
    """Per-agent LLM policy that queries GPT or Claude for action selection."""

    # Class-level tracking for all instances
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        use_dynamic_prompts: bool = True,
        context_window_size: int = 20,
        mg_cfg = None,
    ):
        """Initialize LLM agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults: gpt-4o-mini, claude-3-5-sonnet-20240620, or llama3.2 for ollama)
            temperature: Sampling temperature for LLM
            debug_mode: If True, print human-readable observation debug info (default: True)
            use_dynamic_prompts: If True, use dynamic prompt builder (default: True)
            context_window_size: Number of steps before resending basic info (default: 20)
            mg_cfg: Optional MettaGridConfig for extracting game-specific info (chest vibes, etc.)
        """
        super().__init__(policy_env_info)
        self.provider = provider
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.last_action: str | None = None
        # Handle string "false"/"true" from config system
        if isinstance(use_dynamic_prompts, str):
            self.use_dynamic_prompts = use_dynamic_prompts.lower() not in ("false", "0", "no")
        else:
            self.use_dynamic_prompts = bool(use_dynamic_prompts)

        # Track conversation history for debugging
        self.conversation_history: list[dict] = []

        # Stateful conversation messages (for multi-turn conversations with LLM)
        # Format: [{"role": "user"/"assistant", "content": "..."}, ...]
        self._messages: list[dict[str, str]] = []

        # Initialize prompt builder (new dynamic approach)
        if self.use_dynamic_prompts:
            from mettagrid.policy.llm_prompt_builder import LLMPromptBuilder

            self.prompt_builder = LLMPromptBuilder(
                policy_env_info=policy_env_info,
                context_window_size=context_window_size,
                mg_cfg=mg_cfg,
            )
            self.game_rules_prompt = None  # Not used with dynamic prompts
            if self.debug_mode:
                logger.info(f"Using dynamic prompts with context window size: {context_window_size}")
        else:
            # Fallback to old static prompt
            self.prompt_builder = None
            self.game_rules_prompt = build_game_rules_prompt(policy_env_info)
            if self.debug_mode:
                logger.info("Using static prompts (legacy mode)")

        # Initialize observation debugger if debug mode is enabled
        if self.debug_mode:
            self.debugger = ObservationDebugger(policy_env_info)
        else:
            self.debugger = None

        # Initialize LLM client
        # Note: Model selection is handled by LLMMultiAgentPolicy, so model should always be provided
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
        max_messages = self.prompt_builder.context_window_size * 2 if self.prompt_builder else 40
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

    def step(self, obs: AgentObservation) -> Action:
        """Get action from LLM given observation.

        Args:
            obs: Agent observation

        Returns:
            Action to take
        """
        # Print human-readable debug info if debug mode is enabled
        if self.debug_mode and self.debugger:
            debug_output = self.debugger.debug_observation(obs, self.last_action)
            print("\n" + debug_output + "\n")

        # Build prompt using dynamic or static approach
        if self.use_dynamic_prompts:
            # Use dynamic prompt builder with context window management
            user_prompt, includes_basic_info = self.prompt_builder.context_prompt(obs)
            if self.debug_mode:
                num_msgs = len(self._messages) + 1  # +1 for the new user message we're about to add
                step = self.prompt_builder.step_count
                if includes_basic_info:
                    logger.info(f"[DYNAMIC] Sent basic_info + observable (step {step}, {num_msgs} msgs)")
                else:
                    logger.info(f"[DYNAMIC] Sent observable only (step {step}, {num_msgs} msgs)")
        else:
            # Use old static prompt approach
            obs_json = observation_to_json(obs, self.policy_env_info)
            user_prompt = f"""Current game state:
{json.dumps(obs_json, indent=2)}

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

        # Query LLM
        try:
            action_name = "noop"  # Default fallback

            if self.provider == "openai":
                assert self.client is not None
                # GPT-5 and o1 models use different parameters and don't support system messages
                is_gpt5_or_o1 = self.model.startswith("gpt-5") or self.model.startswith("o1")

                if self.use_dynamic_prompts or is_gpt5_or_o1:
                    # Dynamic prompts or GPT-5/o1: use stateful conversation history
                    messages = self._get_messages_for_api(user_prompt)

                    completion_params = {
                        "model": self.model,
                        "messages": messages,
                        "max_completion_tokens": 50 if is_gpt5_or_o1 else None,
                        "max_tokens": None if is_gpt5_or_o1 else 50,
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
                else:
                    # Standard GPT-4 models with static prompts: use system message
                    completion_params = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.game_rules_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 50,
                        "temperature": self.temperature,
                    }

                    # Track prompt (include system message for static prompts)
                    self.conversation_history.append({
                        "step": len(self.conversation_history) + 1,
                        "system": self.game_rules_prompt,
                        "prompt": user_prompt,
                        "response": None,
                    })

                response = self.client.chat.completions.create(**completion_params)
                action_name = response.choices[0].message.content
                if action_name is None:
                    action_name = "noop"
                action_name = action_name.strip()

                # Add assistant response to stateful conversation history
                if self.use_dynamic_prompts:
                    self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track usage and cost
                usage = response.usage
                if usage:
                    LLMAgentPolicy.total_calls += 1
                    LLMAgentPolicy.total_input_tokens += usage.prompt_tokens
                    LLMAgentPolicy.total_output_tokens += usage.completion_tokens

                    # Calculate cost based on model
                    call_cost = calculate_llm_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
                    LLMAgentPolicy.total_cost += call_cost

                    if self.debug_mode:
                        logger.debug(
                            f"OpenAI response: '{action_name}' | "
                            f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out | "
                            f"Cost: ${call_cost:.6f} | "
                            f"Total so far: ${LLMAgentPolicy.total_cost:.4f}"
                        )

            elif self.provider == "ollama":
                assert self.ollama_client is not None

                if self.use_dynamic_prompts:
                    # Dynamic prompts: use stateful conversation history
                    messages = self._get_messages_for_api(user_prompt)

                    # Track prompt
                    self.conversation_history.append({
                        "step": len(self.conversation_history) + 1,
                        "prompt": user_prompt,
                        "num_messages": len(messages),
                        "response": None,
                    })
                else:
                    # Static prompts: system + user
                    messages = [
                        {"role": "system", "content": self.game_rules_prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                    # Track prompt (include system message for static prompts)
                    self.conversation_history.append({
                        "step": len(self.conversation_history) + 1,
                        "system": self.game_rules_prompt,
                        "prompt": user_prompt,
                        "response": None,
                    })

                response = self.ollama_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=50,
                )

                if self.debug_mode:
                    # Debug: log the raw response
                    logger.debug(f"Ollama response object: {response}")
                    logger.debug(f"Ollama response choices: {response.choices}")

                # Some models (like gpt-oss) put output in 'reasoning' field instead of 'content'
                message = response.choices[0].message
                action_name = message.content or ""

                # Check reasoning field if content is empty
                if not action_name and hasattr(message, "reasoning") and message.reasoning:
                    if self.debug_mode:
                        logger.warning(f"Model used reasoning field instead of content: {message.reasoning[:100]}...")
                    # Try to extract action from reasoning (take last line or last word)
                    reasoning_lines = message.reasoning.strip().split("\n")
                    action_name = reasoning_lines[-1].strip().split()[-1] if reasoning_lines else ""

                if not action_name:
                    if self.debug_mode:
                        reasoning = getattr(message, "reasoning", None)
                        logger.error(
                            f"Ollama returned empty response! content='{message.content}', reasoning='{reasoning}'"
                        )
                    action_name = "noop"

                action_name = action_name.strip()

                # Add assistant response to stateful conversation history
                if self.use_dynamic_prompts:
                    self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track usage (Ollama is free/local)
                usage = response.usage
                if usage:
                    LLMAgentPolicy.total_calls += 1
                    LLMAgentPolicy.total_input_tokens += usage.prompt_tokens
                    LLMAgentPolicy.total_output_tokens += usage.completion_tokens
                    # No cost for local Ollama

                    if self.debug_mode:
                        logger.debug(
                            f"Ollama response: '{action_name}' | "
                            f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out | "
                            f"Cost: $0.00 (local)"
                        )

            elif self.provider == "anthropic":
                assert self.anthropic_client is not None

                if self.use_dynamic_prompts:
                    # Dynamic prompts: use stateful conversation history
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
                        max_tokens=50,
                    )
                else:
                    # Static prompts: system + user
                    # Track prompt (include system message for static prompts)
                    self.conversation_history.append({
                        "step": len(self.conversation_history) + 1,
                        "system": self.game_rules_prompt,
                        "prompt": user_prompt,
                        "response": None,
                    })

                    response = self.anthropic_client.messages.create(
                        model=self.model,
                        system=self.game_rules_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        temperature=self.temperature,
                        max_tokens=50,
                    )
                # Extract text from response content blocks
                from anthropic.types import TextBlock

                action_name = "noop"
                for block in response.content:
                    if isinstance(block, TextBlock):
                        action_name = block.text.strip()
                        break

                # Add assistant response to stateful conversation history
                if self.use_dynamic_prompts:
                    self._add_to_messages("assistant", action_name)

                # Track response for debugging
                self.conversation_history[-1]["response"] = action_name

                # Track usage and cost
                usage = response.usage
                LLMAgentPolicy.total_calls += 1
                LLMAgentPolicy.total_input_tokens += usage.input_tokens
                LLMAgentPolicy.total_output_tokens += usage.output_tokens

                # Calculate cost based on model
                call_cost = calculate_llm_cost(self.model, usage.input_tokens, usage.output_tokens)
                LLMAgentPolicy.total_cost += call_cost

                if self.debug_mode:
                    logger.debug(
                        f"Anthropic response: '{action_name}' | "
                        f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out | "
                        f"Cost: ${call_cost:.6f} | "
                        f"Total so far: ${LLMAgentPolicy.total_cost:.4f}"
                    )

            # Parse and return action
            parsed_action = self._parse_action(action_name)

            # Track last action for debug output
            self.last_action = parsed_action.name

            return parsed_action

        except Exception as e:
            logger.error(f"LLM API error: {e}. Falling back to random action.")
            fallback_action = random.choice(self.policy_env_info.actions.actions())
            self.last_action = fallback_action.name
            return fallback_action

    def _parse_action(self, action_name: str) -> Action:
        """Parse LLM response and return valid Action.

        Args:
            action_name: Action name from LLM response

        Returns:
            Valid Action object
        """
        # Clean up response
        action_name = action_name.strip().strip("\"'").lower()

        # Try exact match first (best case - LLM followed instructions)
        for action in self.policy_env_info.actions.actions():
            if action.name.lower() == action_name:
                return action

        # If response contains multiple words, try to extract action from end
        # (LLM might have said "I will move_east" instead of just "move_east")
        words = action_name.split()
        if len(words) > 1:
            # Check last word first (most likely to be the actual action)
            last_word = words[-1].strip(".,!?;:")
            for action in self.policy_env_info.actions.actions():
                if action.name.lower() == last_word:
                    return action

            # Check each word from end to start
            for word in reversed(words):
                word = word.strip(".,!?;:")
                for action in self.policy_env_info.actions.actions():
                    if action.name.lower() == word:
                        return action

        # Last resort: partial match
        # This is dangerous because it might pick up "don't move_north" as move_north
        for action in self.policy_env_info.actions.actions():
            if action.name.lower() in action_name:
                return action

        # Fallback to random action if parsing completely fails
        return random.choice(self.policy_env_info.actions.actions())

    @classmethod
    def get_cost_summary(cls) -> dict:
        """Get summary of LLM API usage and costs.

        Returns:
            Dictionary with total_calls, total_tokens, total_input_tokens,
            total_output_tokens, and total_cost
        """
        return {
            "total_calls": cls.total_calls,
            "total_tokens": cls.total_input_tokens + cls.total_output_tokens,
            "total_input_tokens": cls.total_input_tokens,
            "total_output_tokens": cls.total_output_tokens,
            "total_cost": cls.total_cost,
        }

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
                print(f"\n[SYSTEM MESSAGE]")
                print(entry["system"])
                print()

            # Print user prompt
            print(f"[USER PROMPT]")
            print(entry["prompt"])
            print()

            # Print LLM response
            print(f"[LLM RESPONSE]")
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
        use_dynamic_prompts: bool = True,
        context_window_size: int = 20,
        mg_cfg = None,
    ):
        """Initialize LLM multi-agent policy.

        Args:
            policy_env_info: Policy environment interface
            provider: LLM provider ("openai", "anthropic", or "ollama")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature for LLM
            debug_mode: If True, print human-readable observation debug info (default: True)
            use_dynamic_prompts: If True, use dynamic prompt builder (default: True)
            context_window_size: Number of steps before resending basic info (default: 20)
            mg_cfg: Optional MettaGridConfig for extracting game-specific info (chest vibes, etc.)
        """
        super().__init__(policy_env_info)
        self.provider: Literal["openai", "anthropic", "ollama"] = provider
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.use_dynamic_prompts = use_dynamic_prompts
        self.context_window_size = context_window_size
        self.mg_cfg = mg_cfg

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

        # Register atexit handler to print costs when program ends (for paid APIs only)
        if provider in ("openai", "anthropic") and not hasattr(LLMMultiAgentPolicy, '_atexit_registered'):
            atexit.register(_print_cost_summary_on_exit)
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
            use_dynamic_prompts=self.use_dynamic_prompts,
            context_window_size=self.context_window_size,
            mg_cfg=self.mg_cfg,
        )



class LLMGPTMultiAgentPolicy(LLMMultiAgentPolicy):
    """OpenAI GPT-based policy for MettaGrid."""

    short_names = ["llm-gpt", "llm-openai"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        use_dynamic_prompts: bool = True,
        context_window_size: int = 20,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="openai",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            use_dynamic_prompts=use_dynamic_prompts,
            context_window_size=context_window_size,
            mg_cfg=mg_cfg,
        )


class LLMClaudeMultiAgentPolicy(LLMMultiAgentPolicy):
    """Anthropic Claude-based policy for MettaGrid."""

    short_names = ["llm-claude", "llm-anthropic"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        use_dynamic_prompts: bool = True,
        context_window_size: int = 20,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="anthropic",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            use_dynamic_prompts=use_dynamic_prompts,
            context_window_size=context_window_size,
            mg_cfg=mg_cfg,
        )


class LLMOllamaMultiAgentPolicy(LLMMultiAgentPolicy):
    """Ollama local LLM-based policy for MettaGrid."""

    short_names = ["llm-ollama", "llm-local"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        model: str | None = None,
        temperature: float = 0.7,
        debug_mode: bool = False,
        use_dynamic_prompts: bool = True,
        context_window_size: int = 20,
        mg_cfg = None,
    ):
        super().__init__(
            policy_env_info,
            provider="ollama",
            model=model,
            temperature=temperature,
            debug_mode=debug_mode,
            use_dynamic_prompts=use_dynamic_prompts,
            context_window_size=context_window_size,
            mg_cfg=mg_cfg,
        )
