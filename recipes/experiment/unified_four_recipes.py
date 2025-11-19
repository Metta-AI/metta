"""Unified Multi-Domain Recipe: Navigation + Assembly Lines + Arena + CvC

This recipe combines four task domains into a single curriculum using TaskGeneratorSet
to train a generalist policy capable of handling:
1. Navigation tasks (path planning, obstacle avoidance)
2. Assembly lines (in-context learning, ordered chains)
3. Arena tasks (resource competition, combat)
4. CvC evaluation tasks (integrative missions requiring multiple skills)

The goal is to evaluate the trained policy on CvC's integrative evaluation suite
to measure transfer learning and multi-skill coordination.

MAJOR CHALLENGES AND CONSIDERATIONS:

1. ACTION SPACE COMPATIBILITY
   - Navigation: minimal actions (move, interact with altars)
   - Assembly: glyph system for assembly, movement
   - Arena: combat actions (attack), resource management
   - CvC: full action suite (extract, assemble, glyph, combat-like)

   SOLUTION: All environments use MettaGridConfig which provides a consistent
   action space. We rely on action masking to disable unavailable actions in
   each domain. We should verify that all task generators produce environments
   with compatible action spaces.

2. REWARD SCALE NORMALIZATION
   - Different domains have different episode lengths and reward distributions
   - Navigation: normalized heart reward = 0.333
   - Assembly: hearts from chain completion
   - Arena: various item rewards (0.1-1.0 range)
   - CvC: heart-based rewards with auxiliary resource rewards

   SOLUTION: Normalize all heart rewards to similar scales (0.1-1.0 range).
   Use consistent reward structure across domains to prevent the learning
   progress algorithm from heavily favoring one domain.

3. TASK DISTRIBUTION IMBALANCE
   - Learning progress algorithm may favor easier tasks
   - Could lead to domain imbalance (e.g., 60% navigation, 10% arena)

   SOLUTION: Monitor per-domain statistics via W&B using task labels.
   Add explicit labels to track domain origin. Consider adjusting weights
   if severe imbalance occurs during training.

4. OBSERVATION SPACE CONSISTENCY
   - All domains use MettaGridConfig with spatial observations
   - Different domains may emphasize different observation features
   - CvC has more complex object types (extractors, assemblers, resources)

   SOLUTION: MettaGridConfig provides unified observations. The policy
   architecture should be capable of handling diverse object types.
   Vision Transformer (ViT) based policies work well here.

5. GENERALIZATION GAP
   - Policy may overfit to training distribution
   - CvC integrative evals test skills not seen in isolated domain training

   SOLUTION: Include diverse task configurations in each domain.
   Use terrain variations, map sizes, and difficulty buckets.
   Regular evaluation on held-out integrative tasks.

6. TRAINING EFFICIENCY
   - Four domains may slow convergence vs single-domain training
   - Need sufficient samples from each domain per batch

   SOLUTION: Use larger num_active_tasks (1500+) to maintain diversity.
   Increase batch size if needed. Monitor learning curves per domain.
   Consider curriculum staging if needed.

7. EPISODE LENGTH VARIABILITY
   - Navigation: typically shorter episodes
   - Assembly: medium length (depends on chain length)
   - Arena: longer episodes with resource gathering
   - CvC: variable length (750-1500 steps)

   SOLUTION: Balance max_steps across domains. Use consistent step budgets
   where possible, or scale proportionally to task complexity.

8. AGENT COUNT COMPATIBILITY
   - Navigation: uses instancing (terrain maps have embedded agent counts)
   - Assembly: 1 agent
   - Arena_CvC: 24 agents (large multi-agent) -> standardized
   - CvC: 1-8 agents (typically 4)

   SOLUTION: Standardize on 6 agents across all domains.
   - Navigation: 6 agents works with terrain maps (divisible by 1, 2, 3, 6)
   - Arena_CvC: uses 6 agents
   - CvC: uses 6 agents (called num_cogs)
   - Assembly: uses 1 agent (handled internally)

   Note: 6 is chosen because terrain maps may have 1 or 6 agents embedded,
   and 6 is divisible by common agent counts (1, 2, 3, 6).
"""

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.map.terrain_from_numpy import NavigationFromNumpy
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.builder import empty_assemblers
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AttackActionConfig,
    ChangeVibeActionConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ResourceModActionConfig,
    WallConfig,
)
from recipes.experiment import cogs_v_clips
from recipes.experiment.assembly_lines import (
    AssemblyLinesTaskGenerator,
)
from recipes.experiment.assembly_lines import (
    make_task_generator_cfg as make_assembly_cfg,
)


def make_unified_action_config() -> ActionsConfig:
    """Create a standardized action configuration for all domains.

    This ensures all environments have the same action space size.
    Domains that don't use certain actions will have them enabled but
    they'll be effectively masked out through action masking or high costs.

    Returns:
        ActionsConfig with all action types enabled
    """
    return ActionsConfig(
        noop=NoopActionConfig(enabled=True),
        move=MoveActionConfig(
            enabled=True,
            allowed_directions=[
                "north",
                "south",
                "east",
                "west",
                "northeast",
                "northwest",
                "southeast",
                "southwest",
            ],
        ),
        attack=AttackActionConfig(
            enabled=True,
            consumed_resources={"laser": 1},  # Will be overridden per domain
        ),
        change_vibe=ChangeVibeActionConfig(
            enabled=True,
            number_of_vibes=4,  # Standard for most domains
        ),
        resource_mod=ResourceModActionConfig(
            enabled=True,  # Enable for compatibility
        ),
    )


def apply_unified_actions(env: MettaGridConfig) -> MettaGridConfig:
    """Apply unified action configuration to an environment.

    This replaces the environment's action config with a standardized one,
    preserving domain-specific settings where needed (like attack costs).
    Also ensures all necessary resources exist in the environment.

    Args:
        env: The environment to modify

    Returns:
        The modified environment with unified actions
    """
    # Store domain-specific attack settings before overwriting
    original_attack_cost = {}
    if hasattr(env.game.actions, "attack") and hasattr(env.game.actions.attack, "consumed_resources"):
        original_attack_cost = env.game.actions.attack.consumed_resources.copy()

    # CRITICAL: Replace vibe list with a standardized, complete list
    # We cannot append to existing lists because different domains have different
    # starting vibe counts, which would result in different action space sizes.
    # ALL environments MUST have EXACTLY the same vibe list for action space compatibility.
    unified_vibes = [
        "default",
        "charger",
        "carbon_a",
        "carbon_b",
        "oxygen_a",
        "oxygen_b",
        "germanium_a",
        "germanium_b",
        "silicon_a",
        "silicon_b",
        "heart_a",
        "heart_b",
        "gear",
        "assembler",
        "chest",
        "wall",
    ]

    # Replace (not append!) the vibe list to ensure all environments have identical vibes
    env.game.vibe_names = unified_vibes.copy()

    # Now create unified actions with the correct vibe count
    unified_actions = make_unified_action_config()
    unified_actions.change_vibe.number_of_vibes = len(env.game.vibe_names)

    # Preserve domain-specific attack settings
    if original_attack_cost:
        unified_actions.attack.consumed_resources = original_attack_cost

    env.game.actions = unified_actions

    # CRITICAL: Replace resource list with a standardized, complete list
    # We cannot append to existing lists because different domains have different
    # starting resources, which would result in different resource_names lists.
    # ALL environments MUST have EXACTLY the same resource list for config compatibility.
    unified_resources = [
        # Assembly line / Arena resources
        "ore_red",
        "ore_blue",
        "ore_green",
        "battery_red",
        "battery_blue",
        "battery_green",
        "heart",
        "armor",
        "laser",
        "blueprint",
        # CvC resources
        "energy",
        "carbon",
        "oxygen",
        "germanium",
        "silicon",
        "decoder",
        "modulator",
        "resonator",
        "scrambler",
    ]

    # Replace (not append!) the resource list to ensure all environments have identical resources
    env.game.resource_names = unified_resources.copy()

    # CRITICAL: Replace objects dict with a standardized, complete dict
    # We cannot merge with existing objects because different domains have different
    # object sets, which would result in different object_type_names.
    # ALL environments MUST have EXACTLY the same objects for config compatibility.

    # Define ALL object types used across ALL domains
    # These must ALL be present even if not used on a specific map
    unified_objects = {
        "wall": WallConfig(name="wall"),
        "altar": empty_assemblers.altar,
        # Assembly line objects
        "generator_red": empty_assemblers.generator_red,
        "generator_blue": empty_assemblers.generator_blue,
        "generator_green": empty_assemblers.generator_green,
        "mine_red": empty_assemblers.mine_red,
        "mine_blue": empty_assemblers.mine_blue,
        "mine_green": empty_assemblers.mine_green,
        "factory": empty_assemblers.factory,
        "temple": empty_assemblers.temple,
        "armory": empty_assemblers.armory,
        "lab": empty_assemblers.lab,
        "lasery": empty_assemblers.lasery,
        # CvC objects (instantiate with defaults)
        "assembler": CvCAssemblerConfig().station_cfg(),
        "chest": CvCChestConfig().station_cfg(),
        "charger": ChargerConfig().station_cfg(),
        "carbon_extractor": CarbonExtractorConfig().station_cfg(),
        "oxygen_extractor": OxygenExtractorConfig().station_cfg(),
        "germanium_extractor": GermaniumExtractorConfig().station_cfg(),
        "silicon_extractor": SiliconExtractorConfig().station_cfg(),
    }

    # Replace (not merge!) the objects dict to ensure all environments have identical object types
    env.game.objects = unified_objects.copy()

    return env


def make_navigation_tasks(num_agents: int = 6, num_instances: int = 4) -> cc.BucketedTaskGenerator.Config:
    """Create navigation task generator with bucketed curriculum.

    Navigation tasks test:
    - Path planning and shortest path finding
    - Obstacle avoidance in varied terrain
    - Multi-goal navigation (multiple altars)

    Args:
        num_agents: Total number of agents desired (used to calculate instances)
        num_instances: Number of map instances (default 4)

    Returns:
        BucketedTaskGenerator.Config for navigation tasks

    Note:
        NavigationFromNumpy reads agent counts from the terrain map files themselves,
        which typically have 1 agent per map. We use instancing to achieve the desired
        total agent count: total_agents = agents_per_instance * instances.

        For compatibility with terrain maps that have 1 agent, we always use
        agents_per_instance=1 and adjust instances to match num_agents.
    """
    from mettagrid.mapgen.mapgen import MapGen

    # IMPORTANT: Terrain map files have a fixed number of agents embedded in them.
    # NavigationFromNumpy reads this from the file, not from the config.
    # Terrain maps can have varying agent counts (1, 6, etc.). We need to ensure
    # num_agents is divisible by the agents in each map file.
    # Using agents_per_instance=1 and instances=num_agents works if maps have 1 agent.
    # For maps with 6 agents, num_agents must be divisible by 6.
    agents_per_instance = 1
    instances = num_agents  # To get 6 total agents, use 6 instances of 1 agent each

    # Base navigation environment with altars as goals
    # Total agents = agents_per_instance * instances
    nav_env = eb.make_navigation(num_agents=agents_per_instance * instances)
    nav_env.label = "navigation"  # Track domain in W&B stats

    # Normalize heart reward for consistency across domains
    nav_env.game.agent.rewards.inventory["heart"] = 0.333
    nav_env.game.max_steps = 1000

    # Use MapGen with NavigationFromNumpy as instance
    # This matches the structure in recipes/experiment/navigation.py
    nav_env.game.map_builder = MapGen.Config(
        instances=instances,
        border_width=6,
        instance_border_width=3,
        instance=NavigationFromNumpy.Config(
            agents=agents_per_instance,  # Must match what's in the terrain map files
            objects={"altar": 10},
            dir="varied_terrain/dense_large",
        ),
    )

    # Apply unified action configuration for compatibility
    nav_env = apply_unified_actions(nav_env)

    nav_tasks = cc.bucketed(nav_env)

    # Bucket 1: Terrain types (diverse navigation challenges)
    # IMPORTANT: Path must be game.map_builder.instance.dir (not game.map_builder.dir)
    terrain_dirs = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            terrain_dirs.append(f"varied_terrain/{terrain}_{size}")

    nav_tasks.add_bucket("game.map_builder.instance.dir", terrain_dirs)

    # Bucket 2: Number of goal objects (task complexity)
    nav_tasks.add_bucket("game.map_builder.instance.objects.altar", [Span(3, 50)])

    # Bucket 3: Episode length (time pressure)
    nav_tasks.add_bucket("game.max_steps", [750, 1000, 1250])

    return nav_tasks


def make_assembly_tasks() -> AssemblyLinesTaskGenerator.Config:
    """Create assembly lines task generator.

    Assembly tasks test:
    - In-context learning (understanding novel assembly chains)
    - Resource routing and ordering
    - Sink exploration (finding useless assemblers)

    Note: Assembly tasks use 1 agent by default. The task generator handles
    agent count internally.

    Returns:
        AssemblyLinesTaskGenerator.Config for assembly tasks
    """
    # Full difficulty curriculum with varied chain lengths and terrains
    # CHALLENGE: Assembly uses 1 agent while other domains use 4
    # MITIGATION: This is acceptable as assembly tasks test different skills
    # and the policy should handle variable agent counts
    return make_assembly_cfg(
        chain_lengths=[1, 2, 3, 4, 5, 6],
        num_sinks=[0, 1, 2],
        room_sizes=["tiny", "small", "medium", "large"],
        terrains=["no-terrain", "sparse", "balanced", "dense"],
    )


def make_arena_cvc_tasks(num_agents: int = 6) -> cc.BucketedTaskGenerator.Config:
    """Create arena_cvc task generator with bucketed curriculum.

    Arena_cvc tasks test:
    - Resource competition and gathering
    - Combat and strategic decision making
    - Multi-agent coordination

    Args:
        num_agents: Number of agents (reduced from 24 to 6 for compatibility)

    Returns:
        BucketedTaskGenerator.Config for arena_cvc tasks
    """
    # CHALLENGE: Original arena recipe uses 24 agents, but CvC evals use 6
    # SOLUTION: Use 6 agents for consistency with evaluation suite
    arena_env = eb.make_arena(num_agents=num_agents)
    arena_env.label = "arena_cvc"  # Track domain in W&B stats
    arena_env.game.max_steps = 1000

    # Apply unified action configuration for compatibility
    arena_env = apply_unified_actions(arena_env)

    arena_tasks = cc.bucketed(arena_env)

    # Bucket 1-4: Reward structure for different resources
    # These buckets teach the agent to value different resources
    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # Bucket 5: Combat settings (enable/disable attacks via cost)
    # High cost (100) effectively disables combat
    # Low cost (1) enables combat strategy
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Bucket 6-7: Initial inventory variations
    arena_tasks.add_bucket("game.agent.initial_inventory.ore_red", [0, 1, 3])
    arena_tasks.add_bucket("game.agent.initial_inventory.battery_red", [0, 3])

    # Bucket 8: Episode length
    arena_tasks.add_bucket("game.max_steps", [750, 1000, 1250])

    return arena_tasks


def make_cvc_tasks(num_cogs: int = 6) -> list[cc.BucketedTaskGenerator.Config]:
    """Create task generators for CvC training missions.

    CvC tasks test integrative skills:
    - Resource extraction and routing
    - Assembly and crafting
    - Multi-agent coordination
    - Energy management
    - Strategic planning

    Args:
        num_cogs: Number of CoGs (agents) per mission

    Returns:
        List of BucketedTaskGenerator.Config, one per base mission
    """
    # Use the default curriculum missions from CvC
    # These are the missions that scripted agents perform well on
    base_missions = list(cogs_v_clips.DEFAULT_CURRICULUM_MISSIONS)

    cvc_task_generators = []

    for mission_name in base_missions:
        # Create base environment for this mission
        try:
            mission_env = cogs_v_clips.make_training_env(
                num_cogs=num_cogs,
                mission=mission_name,
                variants=None,
            )
            mission_env.label = f"cvc_{mission_name}"  # Track per-mission stats

            # IMPORTANT: Check if the mission uses instancing that might conflict
            # Some CvC missions may have MapGen with instances configured
            map_builder = mission_env.game.map_builder
            if hasattr(map_builder, "__class__") and map_builder.__class__.__name__ == "MapGen":
                # Skip missions with instancing for now to avoid agent count conflicts
                print(f"  Skipping CvC mission '{mission_name}' - uses MapGen instancing")
                continue

            # Apply unified action configuration for compatibility
            mission_env = apply_unified_actions(mission_env)

        except (ValueError, AttributeError) as e:
            # Skip missions that don't exist or fail to load
            print(f"  Skipping CvC mission '{mission_name}': {e}")
            continue

        # Create bucketed task generator for this mission
        mission_tasks = cc.bucketed(mission_env)

        # Bucket 1: Episode length (time pressure variations)
        mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])

        # Bucket 2: Heart reward scaling
        mission_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])

        # Bucket 3-6: Small auxiliary rewards to encourage resource collection
        # These help the agent learn resource gathering as a subskill
        for resource in ["carbon", "oxygen", "germanium", "silicon"]:
            mission_tasks.add_bucket(f"game.agent.rewards.inventory.{resource}", [0.0, 0.01, 0.02, 0.05])

        cvc_task_generators.append(mission_tasks)
        print(f"  ✓ Added CvC mission: {mission_name}")

    return cvc_task_generators


def make_curriculum(
    num_agents: int = 6,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    domain_weights: Optional[dict[str, float]] = None,
) -> CurriculumConfig:
    """Create unified curriculum combining all four task domains.

    This curriculum uses TaskGeneratorSet (via cc.merge) to sample tasks from:
    1. Navigation (path planning, terrain navigation)
    2. Assembly Lines (in-context learning, ordered chains)
    3. Arena_CvC (resource competition, combat)
    4. CvC Missions (integrative tasks)

    Args:
        num_agents: Number of agents for navigation/arena (default 4)
        enable_detailed_slice_logging: Enable detailed curriculum slice logging
        algorithm_config: Optional curriculum algorithm (defaults to LearningProgress)
        domain_weights: Optional dict with keys ['navigation', 'assembly', 'arena_cvc', 'cvc']
                       to control sampling weights (defaults to equal weights)

    Returns:
        CurriculumConfig with unified task distribution
    """
    # STEP 1: Create task generators for each domain
    print("Creating task generators for four domains...")
    print(f"  Target agent count: {num_agents}")

    print("  1. Creating navigation tasks...")
    nav_tasks = make_navigation_tasks(num_agents=num_agents)
    print("  ✓ Navigation tasks created")

    print("  2. Creating assembly tasks...")
    assembly_tasks = make_assembly_tasks()
    print("  ✓ Assembly tasks created")

    print("  3. Creating arena_cvc tasks...")
    arena_cvc_tasks = make_arena_cvc_tasks(num_agents=num_agents)
    print("  ✓ Arena_CvC tasks created")

    print("  4. Creating CvC tasks...")
    cvc_tasks = make_cvc_tasks(num_cogs=num_agents)
    print(f"  ✓ CvC tasks created ({len(cvc_tasks)} missions)")

    # STEP 2: Determine sampling weights
    # Default: equal probability across all domains
    # User can override to emphasize specific domains
    if domain_weights is None:
        domain_weights = {
            "navigation": 1.0,
            "assembly": 1.0,
            "arena_cvc": 1.0,
            "cvc": 1.0,
        }

    print(f"Domain sampling weights: {domain_weights}")

    # STEP 3: Merge all task generators
    # CHALLENGE: CvC has multiple task generators (one per mission)
    # SOLUTION: Treat each CvC mission as a separate sub-generator, then weight
    # the entire CvC domain collectively

    # We'll merge CvC missions first, then merge with other domains
    if cvc_tasks:
        # Equal weight across all CvC missions
        # Note: cc.merge doesn't support custom weights in the simple API
        # so we use TaskGeneratorSet.Config directly if we need fine control
        cvc_merged = cc.merge(cvc_tasks)
    else:
        cvc_merged = None

    # Build list of all domain generators
    all_domains = [nav_tasks, assembly_tasks, arena_cvc_tasks]
    domain_weight_list = [
        domain_weights["navigation"],
        domain_weights["assembly"],
        domain_weights["arena_cvc"],
    ]

    if cvc_merged is not None:
        all_domains.append(cvc_merged)
        domain_weight_list.append(domain_weights["cvc"])

    # CHALLENGE: Ensuring balanced sampling across domains
    # SOLUTION: Use explicit weights and monitor via W&B using task labels
    print(f"Merging {len(all_domains)} domain task generators...")

    # Merge all domains with specified weights
    # cc.merge() by default uses equal weights, but we can create TaskGeneratorSet
    # manually for custom weights
    from metta.cogworks.curriculum.task_generator import TaskGeneratorSet

    unified_tasks = TaskGeneratorSet.Config(
        task_generators=all_domains,
        weights=domain_weight_list,
    )

    # STEP 4: Configure learning progress algorithm
    if algorithm_config is None:
        # CHALLENGE: Balancing exploration across four diverse domains
        # SOLUTION: Use bidirectional learning progress with larger task pool
        # and moderate exploration bonus
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Sample both easy and hard tasks
            ema_timescale=0.001,  # Smooth learning progress estimates
            exploration_bonus=0.1,  # Encourage trying new tasks
            max_memory_tasks=2000,  # Larger pool for four domains
            max_slice_axes=5,  # Allow fine-grained task slicing
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    print("Creating curriculum with learning progress algorithm...")

    # STEP 5: Create curriculum with larger active task pool
    # CHALLENGE: Four domains need more active tasks to maintain diversity
    # SOLUTION: Use 1500-2000 active tasks (vs 1000 for single domain)
    return unified_tasks.to_curriculum(
        num_active_tasks=1500,  # Large pool to sample from all domains
        algorithm_config=algorithm_config,
    )


def make_integrative_eval_suite(num_cogs: int = 6) -> list[SimulationConfig]:
    """Create evaluation suite focusing on integrative CvC missions.

    These missions test whether the multi-domain training enables the policy
    to handle complex tasks requiring navigation, assembly, coordination, and
    resource management simultaneously.

    Selected missions:
    - Resource Collection: routing, prioritization, exploration
    - Coordination: specialization, synchronized movement, trading
    - Resource Management: scarcity, energy management, strategic planning
    - Complex Integration: many objects, routing optimization

    Args:
        num_cogs: Number of CoGs per evaluation mission

    Returns:
        List of SimulationConfig for integrative evaluation
    """
    # EVALUATION STRATEGY:
    # Focus on missions that require skills from multiple domains:
    # - Navigation skills: path planning, obstacle avoidance
    # - Assembly skills: understanding object interactions, in-context learning
    # - Arena_CvC skills: resource competition, strategic decision-making
    # - Coordination: multi-agent synchronization

    integrative_missions = [
        # Basic resource collection (navigation + extraction)
        "collect_resources_classic",
        "collect_resources_spread",
        # Long-distance coordination (navigation + planning)
        "collect_far",
        # Multi-agent coordination (arena-like + assembly)
        "divide_and_conquer",
        "go_together",
        # Resource scarcity (strategic planning)
        "oxygen_bottleneck",
        "energy_starved",
        "single_use_swarm",
        # Complex environments (all skills)
        "extractor_hub_30",
        "extractor_hub_50",
        "extractor_hub_70",
        "extractor_hub_80",
    ]

    return cogs_v_clips.make_eval_suite(
        num_cogs=num_cogs,
        difficulty="standard",
        subset=integrative_missions,
    )


def simulations(num_agents: int = 6) -> list[SimulationConfig]:
    """Create comprehensive evaluation suite across all four domains.

    This includes:
    - Domain-specific evals (to check per-domain performance)
    - Integrative CvC evals (to check transfer and multi-skill coordination)

    Args:
        num_agents: Number of agents for evaluation

    Returns:
        List of all evaluation simulations
    """
    from recipes.experiment.arena import simulations as arena_cvc_sims
    from recipes.experiment.assembly_lines import make_assembly_line_eval_suite
    from recipes.experiment.navigation import make_navigation_eval_suite

    all_sims = []

    # Navigation evals
    all_sims.extend(make_navigation_eval_suite())

    # Assembly line evals
    all_sims.extend(make_assembly_line_eval_suite())

    # Arena_CvC evals
    all_sims.extend(arena_cvc_sims(env=None))

    # CvC integrative evals (primary evaluation target)
    all_sims.extend(make_integrative_eval_suite(num_cogs=num_agents))

    return all_sims


def train(
    num_agents: int = 6,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    domain_weights: Optional[dict[str, float]] = None,
) -> TrainTool:
    """Create training tool for unified multi-domain curriculum.

    Args:
        num_agents: Number of agents (default 6 for terrain map compatibility)
        curriculum: Optional pre-configured curriculum
        enable_detailed_slice_logging: Enable detailed logging
        domain_weights: Optional sampling weights for each domain

    Returns:
        TrainTool configured for multi-domain training
    """
    resolved_curriculum = curriculum or make_curriculum(
        num_agents=num_agents,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        domain_weights=domain_weights,
    )

    # Use integrative CvC evals as primary evaluation target
    # RATIONALE: These missions test whether multi-domain training provides
    # positive transfer to complex integrative tasks
    evaluator_cfg = EvaluatorConfig(
        simulations=make_integrative_eval_suite(num_cogs=num_agents),
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_agents: int = 6,
    eval_suite: str = "integrative",
) -> EvaluateTool:
    """Evaluate policies on unified curriculum.

    Args:
        policy_uris: Policy URIs to evaluate
        num_agents: Number of agents
        eval_suite: Which eval suite to use:
            - "integrative": CvC integrative missions only (default)
            - "all": All domain evals + integrative
            - "cvc": All CvC missions

    Returns:
        EvaluateTool configured for evaluation
    """
    if eval_suite == "integrative":
        sims = make_integrative_eval_suite(num_cogs=num_agents)
    elif eval_suite == "all":
        sims = simulations(num_agents=num_agents)
    elif eval_suite == "cvc":
        sims = cogs_v_clips.make_eval_suite(num_cogs=num_agents, difficulty="standard")
    else:
        raise ValueError(f"Unknown eval_suite: {eval_suite}")

    return EvaluateTool(
        simulations=sims,
        policy_uris=policy_uris,
    )


def play(
    policy_uri: Optional[str] = None,
    domain: str = "cvc",
    mission: str = "extractor_hub_30",
    num_agents: int = 6,
) -> PlayTool:
    """Play interactively with a policy.

    Args:
        policy_uri: Optional policy to load
        domain: Which domain to play ("navigation", "assembly", "arena_cvc", "cvc")
        mission: Mission name (for CvC domain)
        num_agents: Number of agents

    Returns:
        PlayTool for interactive play
    """
    if domain == "navigation":
        from recipes.experiment.navigation import play as nav_play

        return nav_play(policy_uri=policy_uri)
    elif domain == "assembly":
        from recipes.experiment.assembly_lines import play as assembly_play

        return assembly_play()
    elif domain == "arena_cvc":
        from recipes.experiment.arena import play as arena_cvc_play

        return arena_cvc_play(policy_uri=policy_uri)
    elif domain == "cvc":
        return cogs_v_clips.play(
            policy_uri=policy_uri,
            mission=mission,
            num_cogs=num_agents,
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")


# MONITORING AND DEBUGGING UTILITIES


def print_curriculum_stats(curriculum: CurriculumConfig, num_samples: int = 1000):
    """Sample from curriculum and print domain distribution statistics.

    Useful for debugging domain imbalance issues.

    Args:
        curriculum: The curriculum to analyze
        num_samples: Number of tasks to sample
    """

    print(f"Sampling {num_samples} tasks from curriculum...")

    # This would require instantiating the curriculum and sampling
    # Left as a placeholder for monitoring during training
    print("Domain distribution will be tracked via W&B task labels during training.")
    print("Check for labels: 'navigation', 'assembly_lines', 'arena', 'cvc_*'")


def verify_action_space_compatibility():
    """Verify that all domains produce compatible action spaces.

    This is a critical check to ensure the unified curriculum will work.
    Run this before starting expensive training runs.
    """
    print("Verifying action space compatibility across domains...")

    # Sample one task from each domain to ensure they can be created
    _ = make_navigation_tasks(num_agents=4)
    _ = make_assembly_tasks()
    _ = make_arena_cvc_tasks(num_agents=4)
    _ = make_cvc_tasks(num_cogs=4)

    print("✓ Task generators created successfully")
    print("\nNOTE: Full action space verification requires instantiating environments.")
    print("Run a smoke test (train for 100k steps) to verify compatibility.")
    print("\nExpected action spaces should be compatible via MettaGridConfig.")
    print("Key actions: move (8 dirs), interact, glyph, attack (masked when unavailable)")


if __name__ == "__main__":
    # Quick verification
    print("=" * 60)
    print("Unified Four-Domain Recipe")
    print("=" * 60)
    verify_action_space_compatibility()
    print("\n" + "=" * 60)
    print("Ready to train!")
    print("=" * 60)
    print("\nUsage:")
    print("  uv run ./tools/run.py recipes.experiment.unified_four_recipes.train")
    print("  uv run ./tools/run.py recipes.experiment.unified_four_recipes.evaluate")
