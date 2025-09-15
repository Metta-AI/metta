Recipe-Based Tool Invocation Design
Goals

Simplify tool invocation - Replace dotted module paths with intuitive verb-recipe syntax
Standardize tool inputs - Every tool has a typed config field
External conversions - Conversion logic lives outside config classes in a registry
Minimal tool complexity - Tools declare config type and preferred builder

Sample Usage
bash# Play an environment
./tools/run.py play navigation

# Run evaluation suite
./tools/run.py sim navigation suite=eval policy_uris=wandb://run/alice.1

# Train a model
./tools/run.py train arena run=local.jack.1 trainer.total_timesteps=10_000_000

# Replay with a policy
./tools/run.py replay navigation policy_uri=wandb://run/alice.1
Core Design
Standardized Tool Structure
Every tool follows the same pattern:
python# metta/common/tool.py
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

TConfig = TypeVar("TConfig", bound=BaseModel)

class Tool(BaseModel, Generic[TConfig]):
    """Base tool class with standardized config field."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    config: TConfig  # The primary configuration

    # Tools declare their preferred builder
    recipe_builder: str = ""

    @abstractmethod
    def invoke(self, args: dict[str, str]) -> int | None: ...
Tool Implementations
python# metta/tools/train.py
class TrainTool(Tool[TrainerConfig]):
    """Training tool."""
    recipe_builder = "trainer"
    run: str  # Additional required field

# metta/tools/play.py
class PlayTool(Tool[SimulationConfig]):
    """Interactive play tool."""
    recipe_builder = "mettagrid"  # Returns MettaGridConfig, will convert
    policy_uri: str | None = None

# metta/tools/sim.py
class SimTool(Tool[list[SimulationConfig]]):
    """Simulation runner."""

    @classmethod
    def resolve_builder(cls, cli_args: dict[str, str]) -> str:
        suite = cli_args.get("suite")
        if suite == "eval":
            return "evals"
        elif suite == "curriculum":
            return "curricula"
        else:
            raise ValueError("sim requires suite=eval or suite=curriculum")

    policy_uris: list[str] = Field(default_factory=list)
Global Conversion Registry
Since configs can't depend on each other, we use a central registry:
python# metta/common/recipes/conversions.py
from typing import Any, Callable, TypeVar, get_type_hints
from dataclasses import dataclass

S = TypeVar("S")
T = TypeVar("T")

@dataclass
class Conversion:
    """A registered conversion between types."""
    source_type: type
    target_type: type
    converter: Callable[[Any, str], Any]  # (value, context) -> converted
    name: str

class ConversionRegistry:
    """Global registry of type conversions."""

    def __init__(self):
        self._conversions: dict[tuple[type, type], Conversion] = {}

    def register(self, source: type[S], target: type[T],
                 converter: Callable[[S, str], T], name: str = "") -> None:
        """Register a conversion function."""
        key = (source, target)
        if key in self._conversions:
            raise ValueError(f"Conversion {source.__name__} → {target.__name__} already registered")

        self._conversions[key] = Conversion(
            source_type=source,
            target_type=target,
            converter=converter,
            name=name or f"{source.__name__}→{target.__name__}"
        )

    def convert(self, value: Any, target_type: type[T], context: str = "") -> T:
        """Convert value to target type if possible."""
        source_type = type(value)

        # Already correct type?
        if isinstance(value, target_type):
            return value

        # Direct conversion?
        key = (source_type, target_type)
        if key in self._conversions:
            conv = self._conversions[key]
            result = conv.converter(value, context)
            print(f"Applied conversion: {conv.name}")
            return result

        # No conversion available
        available = [k[1].__name__ for k in self._conversions if k[0] == source_type]
        msg = f"Cannot convert {source_type.__name__} to {target_type.__name__}"
        if available:
            msg += f"\nAvailable conversions from {source_type.__name__}: {', '.join(available)}"
        raise TypeError(msg)

    def find_conversions_to(self, target_type: type) -> list[type]:
        """Find all types that can convert to target_type."""
        return [k[0] for k in self._conversions if k[1] == target_type]

# Global instance
REGISTRY = ConversionRegistry()
Registering Conversions
python# metta/common/recipes/builtin_conversions.py
from metta.common.recipes.conversions import REGISTRY
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

def register_builtin_conversions():
    """Register core conversions."""

    # MettaGridConfig → SimulationConfig (for play/replay)
    def mg_to_sim(mg: MettaGridConfig, context: str) -> SimulationConfig:
        return SimulationConfig(
            name=context or mg.label,
            env=mg,
            num_episodes=1,
            max_time_s=120,
        )

    REGISTRY.register(MettaGridConfig, SimulationConfig, mg_to_sim, "MG→Sim")

    # Future v2 conversions (commented out for v1):
    # REGISTRY.register(MettaGridConfig, CurriculumConfig, mg_to_curriculum)
    # REGISTRY.register(CurriculumConfig, TrainerConfig, curriculum_to_trainer)
Recipe Builder Tags
For disambiguating multiple builders that return the same type:
python# experiments/recipes/_tags.py
def builder(tag: str = "default"):
    """Tag a builder function for disambiguation."""
    def decorator(func):
        func._builder_tag = tag
        return func
    return decorator
Recipe Example
python# experiments/recipes/navigation.py
from typing import List
from experiments.recipes._tags import builder

@builder()  # Default builder for MettaGridConfig
def mettagrid() -> MettaGridConfig:
    """Base navigation environment."""
    return eb.make_navigation(num_agents=4)

@builder(tag="eval")  # Tagged for suite=eval
def evals() -> List[SimulationConfig]:
    """Evaluation suite."""
    return make_navigation_eval_suite()

@builder(tag="curriculum")  # Tagged for suite=curriculum
def curricula() -> List[SimulationConfig]:
    """Training-style simulations."""
    base = mettagrid()
    return [SimulationConfig(name="navigation/train", env=base)]

@builder()
def trainer() -> TrainerConfig:
    """Training configuration."""
    env = mettagrid()
    curriculum = cc.env_curriculum(env)
    return TrainerConfig(
        curriculum=curriculum,
        evaluation=EvaluationConfig(simulations=evals()),
    )
Runner Implementation
pythondef run_recipe_mode(verb: str, recipe_name: str, cli_args: list[str]):
    """Execute tool with recipe."""

    # 1. Load tool and get expected type
    tool_module = importlib.import_module(f"metta.tools.{verb}")
    tool_class = find_tool_class(tool_module)
    expected_type = get_type_hints(tool_class)["config"]

    # 2. Parse args and determine builder
    parsed_args = parse_cli_args(cli_args)

    if hasattr(tool_class, "resolve_builder"):
        builder_name = tool_class.resolve_builder(parsed_args)
    else:
        builder_name = tool_class.recipe_builder

    # 3. Load recipe module
    recipe_module = importlib.import_module(f"experiments.recipes.{recipe_name}")

    # 4. Find builder function
    builder_func = None

    # First try exact name match
    if hasattr(recipe_module, builder_name):
        builder_func = getattr(recipe_module, builder_name)
    else:
        # For sim tool, also check tagged builders
        if verb == "sim" and "suite" in parsed_args:
            tag = parsed_args["suite"]  # "eval" or "curriculum"
            builder_func = find_tagged_builder(recipe_module, expected_type, tag)

    if not builder_func:
        # Show available builders
        available = find_builders_returning(recipe_module, expected_type)
        if available:
            print(f"Error: Recipe '{recipe_name}' has no '{builder_name}()' function")
            print(f"Available builders returning {expected_type.__name__}: {', '.join(available)}")
        else:
            # Check if any builder can convert to expected type
            convertible = find_convertible_builders(recipe_module, expected_type)
            if convertible:
                print(f"Error: No direct builder for {expected_type.__name__}")
                print(f"Builders that can convert: {', '.join(convertible)}")
            else:
                print(f"Error: No compatible builders in recipe '{recipe_name}'")
        return 1

    # 5. Call builder and convert if needed
    raw_config = builder_func()

    # Import conversions
    from metta.common.recipes.conversions import REGISTRY
    from metta.common.recipes.builtin_conversions import register_builtin_conversions
    register_builtin_conversions()

    config = REGISTRY.convert(raw_config, expected_type, context=recipe_name)

    # 6. Create tool
    func_args, remaining = extract_function_args(parsed_args, tool_class)

    if "config" in parsed_args:
        print("Error: Cannot pass 'config' - it comes from the recipe")
        return 1

    tool = tool_class(config=config, **func_args)

    # 7. Apply overrides and run
    overrides, unknown = classify_remaining_args(remaining, get_tool_fields(tool_class))
    for key, value in overrides.items():
        tool = tool.override(key, value)

    return tool.invoke(func_args) or 0

def find_tagged_builder(module, return_type: type, tag: str):
    """Find builder with specific tag that returns the right type."""
    for name in dir(module):
        obj = getattr(module, name)
        if not callable(obj):
            continue

        # Check return type
        hints = get_type_hints(obj)
        if hints.get("return") != return_type:
            continue

        # Check tag
        if getattr(obj, "_builder_tag", "default") == tag:
            return obj

    return None

def find_builders_returning(module, return_type: type) -> list[str]:
    """Find all builders that return a specific type."""
    builders = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("_"):
            hints = get_type_hints(obj)
            if hints.get("return") == return_type:
                builders.append(name)
    return builders

def find_convertible_builders(module, target_type: type) -> list[str]:
    """Find builders whose return types can convert to target."""
    from metta.common.recipes.conversions import REGISTRY

    convertible_types = REGISTRY.find_conversions_to(target_type)
    builders = []

    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("_"):
            hints = get_type_hints(obj)
            return_type = hints.get("return")
            if return_type in convertible_types:
                builders.append(f"{name} ({return_type.__name__})")

    return builders
Key Design Points
1. Explicit Builder Names
Tools still declare their preferred builder name (recipe_builder = "trainer"), making the system more understandable. The runner tries this first before falling back to type-based matching.
2. Simple Conversion Registry
The registry is straightforward - no complex type shape matching or automatic vectorization. Each conversion is explicitly registered with clear source and target types.
3. Builder Tags for Disambiguation
When multiple builders return the same type (like evals() and curricula() both returning list[SimulationConfig]), we use simple tags rather than complex type inspection.
4. Clear Error Messages
If a builder isn't found, the runner shows:

What builders are available with the right return type
What builders could convert to the right type
What type conversions are registered

5. Minimal V1, Extensible V2
V1 only registers MettaGridConfig → SimulationConfig for play/replay ergonomics. V2 can add more conversions without changing tools or the runner architecture.
Benefits

Clean separation - Configs don't depend on each other
Explicit and discoverable - Clear builder names and conversion registry
Simple tools - Just Tool[ConfigType] with a builder name
Type safe - Type annotations drive the system
Good errors - Shows available options when things don't match
Extensible - Easy to add conversions without touching existing code
RetryJHwhat is recipe_builder doing here and why is it jsut a string?EditPondered removing redundant recipe builder string field.11sThe user is asking about recipe_builder in the design - why it's there and why it's just a string. Looking at the design, I see that each tool has a recipe_builder: str = "" field that declares which builder function name to look for in the recipe module.
This seems redundant with the type-based approach. If we're already using type annotations to match tools to builders, why do we need an explicit string? Let me think about the tradeoffs:

Without recipe_builder:

Runner would inspect tool's config: ConfigType annotation
Search recipe for functions returning ConfigType
If multiple matches, need disambiguation (tags)


With recipe_builder:

Tool explicitly says "I want the 'trainer' function"
More explicit, but coupling tool to recipe structure
What if recipe doesn't have that function but has one returning the right type?



The junior engineer's proposal was more type-driven - just match on types. My design kept the explicit builder name, which does seem like unnecessary coupling.
I should revise the design to remove recipe_builder and rely purely on type matching, with tags for disambiguation when needed.You're right to question this. The recipe_builder string is redundant and creates unnecessary coupling between tools and recipe function names. Let me revise the design to remove it and rely purely on type-driven matching:
Recipe-Based Tool Invocation Design (Type-Driven)
Goals

Simplify tool invocation - Replace dotted module paths with intuitive verb-recipe syntax
Type-driven matching - Tools declare their config type, runner finds matching builders
External conversions - Conversion logic lives in a registry
Zero tool-side complexity - Tools only declare their config type

Core Design
Simplified Tool Structure
Tools only need to declare their config type:
python# metta/common/tool.py
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

TConfig = TypeVar("TConfig", bound=BaseModel)

class Tool(BaseModel, Generic[TConfig]):
    """Base tool class with standardized config field."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    config: TConfig  # The primary configuration - type IS the contract

    @abstractmethod
    def invoke(self, args: dict[str, str]) -> int | None: ...
Tool Implementations
python# metta/tools/train.py
class TrainTool(Tool[TrainerConfig]):
    """Training tool - expects TrainerConfig."""
    run: str  # Additional required field

# metta/tools/play.py
class PlayTool(Tool[SimulationConfig]):
    """Play tool - expects SimulationConfig."""
    policy_uri: str | None = None

# metta/tools/sim.py
class SimTool(Tool[list[SimulationConfig]]):
    """Sim tool - expects list[SimulationConfig]."""
    policy_uris: list[str] = Field(default_factory=list)
Recipe Builders with Tags
When multiple builders return the same type, use tags:
python# experiments/recipes/_tags.py
def builder(tag: str = "default"):
    """Tag a builder function for disambiguation."""
    def decorator(func):
        func._builder_tag = tag
        return func
    return decorator

# experiments/recipes/navigation.py
from typing import List

# No tag needed - only builder returning MettaGridConfig
def mettagrid() -> MettaGridConfig:
    return eb.make_navigation(num_agents=4)

# Tagged because both return list[SimulationConfig]
@builder(tag="eval")
def evals() -> List[SimulationConfig]:
    return make_navigation_eval_suite()

@builder(tag="curriculum")
def curricula() -> List[SimulationConfig]:
    base = mettagrid()
    return [SimulationConfig(name="navigation/train", env=base)]

# No tag needed - only builder returning TrainerConfig
def trainer() -> TrainerConfig:
    env = mettagrid()
    return TrainerConfig(
        curriculum=cc.env_curriculum(env),
        evaluation=EvaluationConfig(simulations=evals())
    )
Runner Logic
The runner finds builders by matching return types:
pythondef run_recipe_mode(verb: str, recipe_name: str, cli_args: list[str]):
    # 1. Get expected type from tool
    tool_class = load_tool_class(verb)
    expected_type = get_type_hints(tool_class)["config"]

    # 2. Load recipe and find matching builders
    recipe_module = importlib.import_module(f"experiments.recipes.{recipe_name}")
    matching_builders = find_builders_returning(recipe_module, expected_type)

    # 3. Select builder
    if len(matching_builders) == 0:
        # No direct match - try conversions
        builder_func = find_convertible_builder(recipe_module, expected_type)
        if not builder_func:
            print(f"Error: No builder in '{recipe_name}' returns {expected_type.__name__}")
            return 1

    elif len(matching_builders) == 1:
        # Single match - use it
        builder_func = matching_builders[0]

    else:
        # Multiple matches - need disambiguation
        tag = parsed_args.get("builder")
        if verb == "sim":
            # Special case: suite is an alias for builder tag
            tag = parsed_args.get("suite", tag)

        if not tag:
            names = [b.__name__ for b in matching_builders]
            print(f"Error: Multiple builders return {expected_type.__name__}: {names}")
            print(f"Specify one with --builder=<tag> or suite=<tag> for sim")
            return 1

        builder_func = find_tagged_builder(matching_builders, tag)
        if not builder_func:
            print(f"Error: No builder with tag '{tag}'")
            return 1

    # 4. Call builder and convert if needed
    config = builder_func()
    if not isinstance(config, expected_type):
        config = REGISTRY.convert(config, expected_type, context=recipe_name)

    # 5. Create and run tool
    tool = tool_class(config=config, **func_args)
    # ... apply overrides and invoke
Why This Is Better

No coupling - Tools don't know recipe function names
Type-driven - The config type annotation IS the contract
Automatic matching - Runner finds compatible builders
Clear disambiguation - Tags only when needed
Simpler tools - Just declare the config type

The type system does all the work - tools declare what they need, recipes declare what they provide, and the runner connects them.
