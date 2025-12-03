"""Lightweight mission difficulty estimation system.

Analyzes mission + variant configurations to estimate difficulty without rendering the map.
Detects conflicts that would make missions impossible and provides a difficulty score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, ProtocolConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHub

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import Mission


@dataclass
class RecipeAnalysis:
    """Analysis of assembler recipe requirements."""

    min_agents_required: int
    cheapest_heart_cost: dict[str, int]  # resource -> amount
    all_recipes_accessible: bool  # can num_cogs access all heart recipes?
    coordination_difficulty: float  # 0-1 based on vibe sync needs


@dataclass
class EnergyAnalysis:
    """Analysis of energy economy."""

    capacity: int
    regen_per_step: int
    move_cost: int
    charger_output: int
    sustainable_steps: int  # steps before energy death without charger
    charger_available: bool
    energy_positive: bool  # regen >= move cost


@dataclass
class ResourceAnalysis:
    """Analysis of resource extraction requirements."""

    heart_cost: dict[str, int]  # total resources needed for 1 heart
    extractor_visits: dict[str, int]  # min visits per extractor type
    extractor_output: dict[str, int]  # output per visit
    total_extractor_visits: int
    extraction_energy_cost: int  # energy needed for silicon extraction


@dataclass
class InitialResourceAnalysis:
    """Analysis of pre-stocked inventories."""

    agent_inventory: dict[str, int]
    chest_inventory: dict[str, int]
    effective_extraction_need: dict[str, int]  # what still needs to be extracted
    first_heart_covered: bool  # initial inventory covers first heart


@dataclass
class SpatialAnalysis:
    """Analysis of map spatial complexity."""

    map_width: int
    map_height: int
    map_area: int
    building_coverage: float
    distribution_type: str
    estimated_avg_distance: float  # estimated avg distance to extractors
    is_hub_based: bool  # hub exists (assembler/chest/charger in center)
    extractors_in_hub: bool  # are extractors near hub or scattered across arena?
    extractor_counts: dict[str, int] = field(default_factory=dict)  # count per type


@dataclass
class ExplorationAnalysis:
    """Analysis of exploration requirements.

    Models the time needed to *discover* extractors before they can be used.
    Even an oracle needs to physically visit locations to "see" them.
    """

    vision_radius: int = 5  # How far agents can see
    num_agents: int = 4  # Standard agent count for comparison

    # Exploration estimates (in steps)
    single_agent_discovery: int = 0  # Steps for 1 agent to discover all needed extractors
    multi_agent_discovery: int = 0  # Steps for num_agents to discover (with parallelism)

    # Coverage metrics
    map_exploration_fraction: float = 0.0  # What fraction of map needs exploring
    discovery_efficiency: float = 1.0  # How efficiently agents can cover the map


@dataclass
class DifficultyReport:
    """Complete difficulty analysis report.

    Models mission difficulty using enzyme kinetics analogy:
    - Startup cost (Km): Time to first heart
    - Steady state rate (Vmax): Hearts per 1000 steps once warmed up
    - Substrate depletion: Max hearts if extractors have max_uses limits

    The difficulty_score represents expected steps per heart in steady state / 1000:
    - 0.1 = expect 1 heart per ~100 steps (very easy)
    - 1.0 = expect 1 heart per ~1000 steps
    - 10.0 = expect 1 heart per ~10000 steps (max_steps limit)
    - inf = absorbing state likely (energy death, etc.)
    """

    feasible: bool
    conflicts: list[str] = field(default_factory=list)

    # Component analyses
    energy: EnergyAnalysis | None = None
    resources: ResourceAnalysis | None = None
    recipe: RecipeAnalysis | None = None
    initial_resources: InitialResourceAnalysis | None = None
    spatial: SpatialAnalysis | None = None
    exploration: ExplorationAnalysis | None = None

    # Enzyme kinetics model
    first_heart_steps: int = 0  # Startup cost: steps to first heart
    steady_state_steps: int = 0  # Steady rate: steps per heart after warmup
    max_hearts: int | None = None  # Substrate limit: None = infinite, else max hearts possible
    total_expected_hearts: float = 0  # Expected hearts in max_steps episode

    # Risk factors
    coordination_overhead: float = 1.0  # Multiplier for multi-agent coordination
    success_probability: float = 1.0  # P(completing without absorbing state)

    # Derived metrics
    expected_steps_per_heart: float = 0  # E[steps] = steady_state * coordination / P(success)
    min_extractor_visits: dict[str, int] = field(default_factory=dict)

    # Final score: E[steps per heart] / 1000
    difficulty_score: float = 1.0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        if self.difficulty_score == float("inf"):
            score_str = "∞"
            rate_str = "0 hearts (absorbing state)"
        elif self.difficulty_score > 100:
            score_str = f"{self.difficulty_score:.0f}"
            rate_str = f"~1 heart per {int(self.difficulty_score * 1000)} steps"
        else:
            score_str = f"{self.difficulty_score:.2f}"
            rate_str = f"~1 heart per {int(self.difficulty_score * 1000)} steps"

        coord_str = f"{self.coordination_overhead:.1f}x" if self.coordination_overhead > 1.0 else "none"

        # Max hearts info
        if self.max_hearts is None:
            substrate_str = "∞ (no depletion)"
        else:
            substrate_str = f"{self.max_hearts} hearts max"

        # Exploration info
        if self.exploration:
            explore_str = f"{self.exploration.multi_agent_discovery} steps ({self.exploration.num_agents} agents)"
        else:
            explore_str = "N/A"

        lines = [
            f"Difficulty: {score_str} (E[steps/heart] / 1000)",
            "",
            "Kinetics:",
            f"  Exploration: {explore_str}",
            f"  Startup: {self.first_heart_steps} steps to first heart",
            f"  Steady:  {self.steady_state_steps} steps/heart",
            f"  Rate:    {rate_str}",
            f"  Substrate: {substrate_str}",
            f"  Expected: ~{self.total_expected_hearts:.1f} hearts in 10k steps",
            "",
            f"Risk: Coord={coord_str} | P(success)={self.success_probability:.1%}",
        ]
        if not self.feasible:
            lines.insert(0, "⚠️  NOT FEASIBLE")
        if self.conflicts:
            lines.append(f"Conflicts: {', '.join(self.conflicts)}")
        lines.append(f"Visits/Heart: {self.min_extractor_visits}")
        return "\n".join(lines)


def estimate_difficulty(mission: Mission) -> DifficultyReport:
    """Estimate difficulty of a mission configuration.

    Difficulty = E[steps per heart in steady state] / 1000

    Considers:
    - First heart (may be easier with initial inventory)
    - Steady-state extraction cycle (full resource gathering)
    - Coordination overhead for multi-agent recipes
    - Probability of absorbing states (energy death)

    Args:
        mission: Mission instance (with variants already applied)

    Returns:
        DifficultyReport with expected steps per heart and difficulty score
    """
    conflicts: list[str] = []

    # Run all analyses
    energy = _analyze_energy(mission)
    resources = _analyze_resources(mission)
    recipe = _analyze_recipe(mission)
    initial = _analyze_initial_resources(mission, resources)
    spatial = _analyze_spatial(mission)
    exploration = _analyze_exploration(mission, spatial, resources)

    # Detect hard conflicts (definitely impossible)
    conflicts.extend(_detect_conflicts(mission, energy, resources, recipe, initial))

    # === FIRST HEART: Includes exploration + extraction ===
    # First heart requires discovering extractors before using them
    first_heart_visits = {}
    for resource, need in (initial.effective_extraction_need if initial else resources.heart_cost).items():
        output = resources.extractor_output.get(resource, 1)
        first_heart_visits[resource] = max(0, math.ceil(need / output)) if output > 0 else 0

    first_heart_movement = _estimate_steps_to_heart(
        mission, energy, resources, recipe, initial, spatial, first_heart_visits
    )
    first_heart_movement_steps = _calculate_total_steps(energy, first_heart_movement, resources)

    # Add exploration time for first heart (agents must discover extractors)
    exploration_steps = exploration.multi_agent_discovery if exploration else 0
    first_heart_steps = first_heart_movement_steps + exploration_steps

    # === STEADY STATE: Full extraction cycle (no initial inventory benefit) ===
    steady_state_visits = {}
    for resource, need in resources.heart_cost.items():
        output = resources.extractor_output.get(resource, 1)
        steady_state_visits[resource] = max(0, math.ceil(need / output)) if output > 0 else 0

    # Create a "no initial inventory" version for steady state calculation
    steady_state_movement = _estimate_steps_to_heart(
        mission, energy, resources, recipe, None, spatial, steady_state_visits
    )
    steady_state_steps = _calculate_total_steps(energy, steady_state_movement, resources)

    # === COORDINATION OVERHEAD for multi-agent recipes ===
    coordination_overhead = _calculate_coordination_overhead(recipe, mission)

    # === SUCCESS PROBABILITY (risk of absorbing states) ===
    # Use steady-state steps for probability since that's the repeating cycle
    success_probability = _calculate_success_probability(energy, steady_state_steps, resources, spatial)

    # === SUBSTRATE DEPLETION (max_hearts from extractor max_uses) ===
    max_hearts = _calculate_max_hearts(mission, steady_state_visits, spatial)

    # === EXPECTED STEPS PER HEART ===
    # E[steps] = steady_state_steps * coordination_overhead / P(success)
    if success_probability <= 0:
        expected_steps_per_heart = float("inf")
        conflicts.append("Absorbing state certain: cannot complete mission")
    else:
        expected_steps_per_heart = (steady_state_steps * coordination_overhead) / success_probability

    # === TOTAL EXPECTED HEARTS (in max_steps episode) ===
    max_steps = 10000  # Standard episode length
    if expected_steps_per_heart == float("inf"):
        total_expected_hearts = 0.0
    elif max_hearts is not None:
        # Substrate-limited: can't exceed max_hearts
        hearts_from_rate = max_steps / expected_steps_per_heart
        total_expected_hearts = min(max_hearts, hearts_from_rate)
    else:
        # Unlimited substrate
        total_expected_hearts = max_steps / expected_steps_per_heart

    # === DIFFICULTY SCORE ===
    # Difficulty = E[steps per heart] / 1000
    if expected_steps_per_heart == float("inf"):
        difficulty_score = float("inf")
    else:
        difficulty_score = expected_steps_per_heart / 1000

    feasible = len(conflicts) == 0 and difficulty_score < float("inf")

    return DifficultyReport(
        feasible=feasible,
        conflicts=conflicts,
        energy=energy,
        resources=resources,
        recipe=recipe,
        initial_resources=initial,
        spatial=spatial,
        exploration=exploration,
        first_heart_steps=first_heart_steps,
        steady_state_steps=steady_state_steps,
        max_hearts=max_hearts,
        total_expected_hearts=total_expected_hearts,
        coordination_overhead=coordination_overhead,
        success_probability=success_probability,
        expected_steps_per_heart=expected_steps_per_heart,
        min_extractor_visits=steady_state_visits,
        difficulty_score=difficulty_score,
    )


# =============================================================================
# Energy Analysis
# =============================================================================


def _analyze_energy(mission: Mission) -> EnergyAnalysis:
    """Analyze energy economy from mission config."""
    capacity = mission.energy_capacity
    regen = mission.energy_regen_amount
    move_cost = mission.move_energy_cost

    # Charger output: 50 * efficiency / 100
    charger_efficiency = mission.charger.efficiency
    charger_output = 50 * charger_efficiency // 100

    # Check if chargers are available (not clipped, or can be unclipped via clip_period)
    # clip_period > 0 means periodic unclipping happens
    charger_available = not mission.charger.start_clipped or mission.clip_period > 0

    # Sustainable steps without charger: how many moves before energy runs out
    # Each step: lose move_cost, gain regen
    net_per_step = regen - move_cost
    if net_per_step >= 0:
        sustainable_steps = 10000  # effectively unlimited
        energy_positive = True
    else:
        # Steps until energy depleted
        sustainable_steps = capacity // abs(net_per_step) if net_per_step != 0 else 10000
        energy_positive = False

    return EnergyAnalysis(
        capacity=capacity,
        regen_per_step=regen,
        move_cost=move_cost,
        charger_output=charger_output,
        sustainable_steps=sustainable_steps,
        charger_available=charger_available,
        energy_positive=energy_positive,
    )


def _calculate_total_steps(
    energy: EnergyAnalysis,
    movement_steps: int,
    resources: ResourceAnalysis,
) -> int:
    """Calculate total steps including waiting/charger time for energy management.

    If regen > 0, agents may need to wait between moves to recharge.
    If regen = 0 but charger available, agents must make charger trips.
    Total time = movement_steps + overhead_steps

    Returns:
        Total timesteps needed (movement + waiting/charger overhead)
    """
    if energy.energy_positive:
        # No waiting needed - can move continuously
        return movement_steps

    # Total energy needed for the mission
    total_energy_needed = movement_steps * energy.move_cost + resources.extraction_energy_cost

    if energy.regen_per_step == 0:
        # ZERO REGEN: Must rely entirely on chargers
        if not energy.charger_available:
            # Impossible - return movement steps (will be flagged as infeasible elsewhere)
            return movement_steps

        # Calculate mandatory charger trips
        # Each trip: travel to charger (~10 steps avg) + travel back (~10 steps)
        # But these trips ALSO cost energy!
        steps_per_charge_cycle = energy.capacity // energy.move_cost
        charger_trip_cost = 20  # steps to/from charger

        # How many times do we need to recharge?
        # We can take steps_per_charge_cycle moves, then must recharge
        # But the charger trip itself costs charger_trip_cost * move_cost energy
        effective_steps_per_cycle = steps_per_charge_cycle - charger_trip_cost
        if effective_steps_per_cycle <= 0:
            # Charger is too far - each recharge costs more than it gives
            # This is effectively impossible without perfect pathing
            return movement_steps * 10  # Flag as very expensive

        charger_visits_needed = max(0, (movement_steps - steps_per_charge_cycle) // effective_steps_per_cycle)
        charger_overhead = charger_visits_needed * charger_trip_cost

        return movement_steps + charger_overhead

    # POSITIVE REGEN: Can wait to recharge between moves
    # Energy we get from regen during movement
    energy_from_movement_regen = movement_steps * energy.regen_per_step

    # Energy deficit we need to wait for
    energy_deficit = total_energy_needed - energy.capacity - energy_from_movement_regen

    if energy_deficit <= 0:
        # No waiting needed - initial capacity + movement regen covers it
        return movement_steps

    # Waiting steps needed to regenerate the deficit
    waiting_steps = math.ceil(energy_deficit / energy.regen_per_step)

    # Factor in charger availability - but only if charger is net positive!
    # Charger trip cost: ~20 steps * move_cost energy
    # Charger benefit: charger_output energy
    # Net energy per charger visit = charger_output - (trip_cost * move_cost)
    if energy.charger_available and energy.charger_output > 0:
        charger_trip_steps = 20
        charger_trip_energy = charger_trip_steps * energy.move_cost
        net_charger_benefit = energy.charger_output - charger_trip_energy

        if net_charger_benefit > 0:
            # Charger is worth using - faster than waiting
            charger_visits = max(1, math.ceil(energy_deficit / net_charger_benefit))
            charger_overhead = charger_visits * charger_trip_steps
            # Use whichever is faster: waiting or charger visits
            waiting_steps = min(waiting_steps, charger_overhead)
        # else: charger costs more energy than it provides, just wait

    return movement_steps + waiting_steps


def _calculate_success_probability(
    energy: EnergyAnalysis,
    optimal_steps: int,
    resources: ResourceAnalysis,
    spatial: SpatialAnalysis | None,
) -> float:
    """Calculate probability of successfully reaching first heart.

    Accounts for risk of hitting absorbing states (energy death).

    Key insight: Even with positive regen, if regen < move_cost, the agent
    has a finite energy budget. Must verify:
      capacity + (steps * regen) >= (steps * move_cost) + extraction_cost

    Hub-based maps have PREDICTABLE charger locations, reducing gap risk.

    Returns:
        P(success) in [0, 1]
        - 1.0 = guaranteed success (energy positive or budget sufficient)
        - 0.5 = 50% chance of absorbing state
        - 0.0 = certain failure
    """
    if energy.energy_positive:
        # Regen >= move_cost: infinite sustainable movement
        return 1.0

    # Calculate total energy budget
    # Available = initial capacity + (regen accumulated over all steps)
    # But regen only happens when NOT moving, so it's more nuanced:
    # If we take S steps and wait W steps:
    #   energy_used = S * move_cost + extraction_cost
    #   energy_gained = (S + W) * regen + capacity
    # We can wait arbitrarily, so if regen > 0, we can always recover...
    # BUT if we're moving continuously without waiting, we drain energy.
    #
    # Key insight: The MINIMUM energy needed assumes optimal play where
    # agent waits to recharge between bursts of movement.
    # Total energy needed = optimal_steps * move_cost + extraction_cost
    # Energy budget = capacity + infinite_waiting_time * regen
    #
    # If regen > 0, we can always wait long enough to accumulate energy.
    # The question is: how many TOTAL steps (move + wait) does that take?

    total_energy_needed = optimal_steps * energy.move_cost + resources.extraction_energy_cost

    if energy.regen_per_step > 0:
        # With positive regen, we can ALWAYS complete by waiting.
        # Calculate how much waiting is needed:
        # Wait steps W such that: capacity + (optimal_steps + W) * regen >= total_energy_needed
        # W >= (total_energy_needed - capacity - optimal_steps * regen) / regen

        energy_from_moving = optimal_steps * energy.regen_per_step
        energy_deficit = total_energy_needed - energy.capacity - energy_from_moving

        if energy_deficit <= 0:
            # No waiting needed - budget is sufficient
            return 1.0

        # Waiting is needed, but still guaranteed success (just slower)
        # However, if waiting requirement is extreme, it's impractical
        wait_steps_needed = math.ceil(energy_deficit / energy.regen_per_step)
        total_steps = optimal_steps + wait_steps_needed

        if total_steps > 10000:
            # Would take more than max_steps - effectively impossible
            if energy.charger_available:
                # Charger can help, but still very hard
                return 0.5
            return 0.1  # Theoretically possible but impractical

        # Guaranteed success, just need patience
        return 1.0

    # ZERO REGEN: Cannot recover from energy depletion - risky!
    if not energy.charger_available:
        # No regen, no charger = finite energy pool
        total_energy_needed = optimal_steps * energy.move_cost + resources.extraction_energy_cost
        if total_energy_needed > energy.capacity:
            return 0.0  # Certain failure
        margin = 1.0 - (total_energy_needed / energy.capacity)
        if margin < 0.1:
            return 0.1
        elif margin < 0.3:
            return 0.5
        else:
            return 0.8

    # Zero regen but charger available
    # Risk depends heavily on map structure

    steps_per_charge = energy.capacity // energy.move_cost

    # Get charger distance from spatial analysis
    if spatial is not None:
        avg_charger_dist = spatial.estimated_avg_distance  # Hub: ~6, Large: ~30+
        is_hub = spatial.is_hub_based
    else:
        avg_charger_dist = 15.0
        is_hub = False

    charger_round_trip = avg_charger_dist * 2
    effective_range = steps_per_charge - charger_round_trip

    if effective_range <= 0:
        return 0.1  # Charger at edge of range

    # Hub-based maps: predictable charger location = much safer
    # Agent always knows where charger is, no "gaps" in the sense of unknown territory
    if is_hub:
        # In hub maps, the main risk is suboptimal pathing, not getting lost
        # With charger in center and avg_distance ~6, effective_range ~38
        # This is very safe for a hub map (diagonal ~15)

        if effective_range > avg_charger_dist * 3:
            # Range covers hub comfortably
            return 0.98
        elif effective_range > avg_charger_dist * 2:
            return 0.95
        elif effective_range > avg_charger_dist:
            return 0.90
        else:
            # Tight but manageable in predictable hub
            return 0.80

    # Large procedural maps: scattered chargers, potential gaps
    # Risk of wandering into area where no charger is reachable

    coverage_ratio = steps_per_charge / max(1, optimal_steps)

    if coverage_ratio > 2.0:
        # Range covers mission 2x over - safe even with some exploration
        return 0.95
    elif coverage_ratio > 1.0:
        # Range covers mission - reasonably safe with good planning
        return 0.85
    elif coverage_ratio > 0.5:
        # Need multiple charge cycles
        charge_cycles = math.ceil(1.0 / coverage_ratio)
        # Each cycle has ~3% risk in medium-sized maps
        return 0.97 ** charge_cycles
    else:
        # Many charge cycles needed - higher risk
        charge_cycles = math.ceil(1.0 / coverage_ratio)
        risk_per_cycle = 0.05 + (1.0 - coverage_ratio) * 0.05
        return max(0.2, (1.0 - risk_per_cycle) ** min(charge_cycles, 15))


def _calculate_coordination_overhead(recipe: RecipeAnalysis, mission: Mission) -> float:
    """Calculate coordination factor for multi-agent gathering.

    This is actually a SPEEDUP for most multi-agent scenarios because:
    1. N agents gather N times the resources per unit time
    2. Parallel exploration/extraction
    3. Some coordination overhead for multi-agent recipes

    Returns:
        Factor to apply to per-heart steps.
        < 1.0 = multi-agent speedup (more common)
        > 1.0 = coordination overhead dominates
    """
    num_cogs = mission.num_cogs if mission.num_cogs is not None else STANDARD_NUM_AGENTS
    agents_required = recipe.min_agents_required

    # === MULTI-AGENT GATHERING SPEEDUP ===
    # With N agents independently gathering, throughput scales almost linearly.
    # Each agent brings resources back independently.
    # Efficiency factor accounts for contention and coordination.

    if num_cogs <= 1:
        # Single agent: no parallelism
        gathering_speedup = 1.0
    else:
        # N agents can gather N resources in parallel
        # ~85% efficiency due to contention (same extractors, paths)
        gathering_efficiency = 0.85
        gathering_speedup = num_cogs * gathering_efficiency

    # === COORDINATION OVERHEAD (for multi-agent recipes) ===
    if agents_required <= 1:
        # Single agent recipe: no coordination needed
        coordination_penalty = 1.0
    else:
        # Multi-agent recipe: agents must meet at assembler
        # This is actually mild because agents cycle back naturally

        # Formula: 1.0 + 0.05 * (k-1)^1.3
        # k=2: 1.05x, k=3: 1.12x, k=4: 1.20x
        coordination_penalty = 1.0 + 0.05 * ((agents_required - 1) ** 1.3)

        # Vibe complexity adds some overhead
        vibe_penalty = 1.0 + recipe.coordination_difficulty * 0.1
        coordination_penalty *= vibe_penalty

        # Agent slack: more agents than needed = easier
        if num_cogs > agents_required:
            slack_bonus = 0.97 ** (num_cogs - agents_required)
            coordination_penalty *= slack_bonus

    # === COMBINED FACTOR ===
    # Net factor = coordination_penalty / gathering_speedup
    # For 4 agents, single-agent recipe: 1.0 / 3.4 ≈ 0.29 (3.5x faster)
    # For 4 agents, 4-agent recipe: 1.2 / 3.4 ≈ 0.35 (2.8x faster)

    return coordination_penalty / gathering_speedup


# =============================================================================
# Resource Analysis
# =============================================================================


def _analyze_resources(mission: Mission) -> ResourceAnalysis:
    """Analyze resource extraction requirements."""
    # Heart cost from assembler config
    first_heart_cost = mission.assembler.first_heart_cost

    heart_cost = {
        "carbon": first_heart_cost,
        "oxygen": first_heart_cost,
        "germanium": max(1, first_heart_cost // 5),
        "silicon": 3 * first_heart_cost,
    }

    # Extractor outputs per use (from stations.py formulas)
    carbon_output = 2 * mission.carbon_extractor.efficiency // 100
    oxygen_output = 10  # fixed, cooldown-gated
    germanium_output = 2  # fixed
    silicon_output = 15 * mission.silicon_extractor.efficiency // 100

    extractor_output = {
        "carbon": max(1, carbon_output),
        "oxygen": oxygen_output,
        "germanium": germanium_output,
        "silicon": max(1, silicon_output),
    }

    # Calculate visits needed
    extractor_visits = {}
    for resource, need in heart_cost.items():
        output = extractor_output[resource]
        extractor_visits[resource] = math.ceil(need / output) if output > 0 else need

    total_visits = sum(extractor_visits.values())

    # Silicon extraction costs energy (20 per use)
    silicon_visits = extractor_visits.get("silicon", 0)
    extraction_energy_cost = silicon_visits * 20

    return ResourceAnalysis(
        heart_cost=heart_cost,
        extractor_visits=extractor_visits,
        extractor_output=extractor_output,
        total_extractor_visits=total_visits,
        extraction_energy_cost=extraction_energy_cost,
    )


# =============================================================================
# Substrate Depletion (Max Hearts)
# =============================================================================


def _calculate_max_hearts(
    mission: Mission,
    visits_per_heart: dict[str, int],
    spatial: SpatialAnalysis | None = None,
) -> int | None:
    """Calculate maximum hearts possible before extractor depletion.

    IMPORTANT: max_uses is PER EXTRACTOR, not global!
    Total uses = num_extractors × max_uses_per_extractor

    For hub maps: 1 extractor of each type (in corners)
    For arena maps: Many extractors, estimated from building coverage

    Returns:
        None if unlimited (all extractors have max_uses=0)
        int if limited by extractor depletion
    """
    extractors = {
        "carbon": mission.carbon_extractor,
        "oxygen": mission.oxygen_extractor,
        "germanium": mission.germanium_extractor,
        "silicon": mission.silicon_extractor,
    }

    # Estimate extractor counts based on map type
    if spatial and spatial.extractors_in_hub:
        # Hub maps: exactly 1 of each extractor in corners
        extractor_counts = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}
    elif spatial:
        # Arena maps: estimate from building coverage
        # Typical arena: ~1% coverage, ~25% of buildings are extractors
        # Split roughly evenly among 4 types
        area = spatial.map_area
        total_buildings = max(4, int(area * spatial.building_coverage))
        extractors_per_type = max(1, total_buildings // 8)  # ~12.5% per extractor type
        extractor_counts = {
            "carbon": extractors_per_type,
            "oxygen": extractors_per_type,
            "germanium": extractors_per_type,
            "silicon": extractors_per_type,
        }
    else:
        # Unknown: assume hub-like (conservative)
        extractor_counts = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}

    max_hearts_per_resource: list[int] = []

    for resource, extractor in extractors.items():
        max_uses_per = extractor.max_uses
        if max_uses_per == 0:
            # Unlimited uses for this extractor type
            continue

        count = extractor_counts.get(resource, 1)
        total_uses = count * max_uses_per
        visits_needed = visits_per_heart.get(resource, 1)

        if visits_needed > 0:
            hearts_from_resource = total_uses // visits_needed
            max_hearts_per_resource.append(hearts_from_resource)

    if not max_hearts_per_resource:
        # All extractors are unlimited
        return None

    # Limited by the scarcest resource
    return min(max_hearts_per_resource)


# =============================================================================
# Recipe Analysis
# =============================================================================


def _analyze_recipe(mission: Mission) -> RecipeAnalysis:
    """Analyze assembler recipe complexity."""
    num_cogs = mission.num_cogs if mission.num_cogs is not None else mission.site.min_cogs

    # We need to look at the actual protocols after variants are applied
    # Create env to get the modified assembler config
    env = mission.make_env()
    assembler = env.game.objects.get("assembler")

    if not isinstance(assembler, AssemblerConfig):
        # Fallback to mission config
        return _analyze_recipe_from_mission(mission, num_cogs)

    heart_protocols: list[ProtocolConfig] = []
    for proto in assembler.protocols:
        if proto.output_resources.get("heart", 0) > 0:
            heart_protocols.append(proto)

    if not heart_protocols:
        return RecipeAnalysis(
            min_agents_required=1,
            cheapest_heart_cost={},
            all_recipes_accessible=False,
            coordination_difficulty=1.0,
        )

    # Find cheapest accessible recipe
    cheapest_cost: dict[str, int] = {}
    cheapest_total = float("inf")
    min_agents_required = 1
    accessible_count = 0

    for proto in heart_protocols:
        # Agents needed = max(len(vibes), min_agents)
        agents_needed = max(len(proto.vibes), proto.min_agents)

        if agents_needed <= num_cogs:
            accessible_count += 1
            total_cost = sum(proto.input_resources.values())
            if total_cost < cheapest_total:
                cheapest_total = total_cost
                cheapest_cost = dict(proto.input_resources)
                min_agents_required = agents_needed

    all_accessible = accessible_count == len(heart_protocols)

    # Coordination difficulty based on vibe requirements
    # More vibes = harder to coordinate
    max_vibes = max((len(p.vibes) for p in heart_protocols if len(p.vibes) <= num_cogs), default=1)
    coordination_difficulty = min(1.0, (max_vibes - 1) / 3)  # 1 vibe = 0, 4 vibes = 1

    return RecipeAnalysis(
        min_agents_required=min_agents_required,
        cheapest_heart_cost=cheapest_cost,
        all_recipes_accessible=all_accessible,
        coordination_difficulty=coordination_difficulty,
    )


def _analyze_recipe_from_mission(mission: Mission, num_cogs: int) -> RecipeAnalysis:
    """Fallback recipe analysis from mission config."""
    first_cost = mission.assembler.first_heart_cost
    cheapest_cost = {
        "carbon": first_cost,
        "oxygen": first_cost,
        "germanium": max(1, first_cost // 5),
        "silicon": 3 * first_cost,
    }
    return RecipeAnalysis(
        min_agents_required=1,
        cheapest_heart_cost=cheapest_cost,
        all_recipes_accessible=num_cogs >= 4,
        coordination_difficulty=0.0 if num_cogs == 1 else 0.3,
    )


# =============================================================================
# Initial Resource Analysis
# =============================================================================


def _analyze_initial_resources(mission: Mission, resources: ResourceAnalysis) -> InitialResourceAnalysis:
    """Analyze pre-stocked inventories."""
    # Get actual initial inventories from env (after variants applied)
    env = mission.make_env()

    agent_inventory: dict[str, int] = dict(env.game.agent.initial_inventory)
    chest_inventory: dict[str, int] = {}

    # Check chest initial inventory
    chest = env.game.objects.get("chest")
    if isinstance(chest, ChestConfig):
        chest_inventory = dict(chest.initial_inventory)

    # Calculate effective extraction need
    effective_need: dict[str, int] = {}
    for resource, need in resources.heart_cost.items():
        available = agent_inventory.get(resource, 0) + chest_inventory.get(resource, 0)
        effective_need[resource] = max(0, need - available)

    # Check if first heart is covered
    first_heart_covered = all(effective_need.get(r, 0) == 0 for r in resources.heart_cost)

    return InitialResourceAnalysis(
        agent_inventory=agent_inventory,
        chest_inventory=chest_inventory,
        effective_extraction_need=effective_need,
        first_heart_covered=first_heart_covered,
    )


# =============================================================================
# Spatial Analysis
# =============================================================================


def _analyze_spatial(mission: Mission) -> SpatialAnalysis:
    """Analyze map spatial complexity from config.

    Key distinction:
    - is_hub_based: Does the map have a central hub (assembler/chest/charger)?
    - extractors_in_hub: Are extractors INSIDE the hub (small maps) or
      scattered across the arena (large maps like MachinaArena)?

    For distance estimation:
    - If extractors_in_hub: Use hub distances (~6-8 tiles)
    - If extractors scattered: Use map size / 3 for uniform distribution
    """
    map_builder = mission.site.map_builder

    # Defaults
    width = 50
    height = 50
    building_coverage = 0.01
    distribution_type = "uniform"
    is_hub_based = False
    extractors_in_hub = False

    if isinstance(map_builder, MapGen.Config):
        width = map_builder.width or 50
        height = map_builder.height or 50

        instance = map_builder.instance
        if instance is not None:
            from cogames.cogs_vs_clips.procedural import MachinaArena, RandomTransform

            if isinstance(instance, MachinaArena.Config):
                # MachinaArena: hub at center, extractors scattered across arena
                building_coverage = instance.building_coverage
                dist_config = instance.distribution
                distribution_type = dist_config.type.value if dist_config else "uniform"
                is_hub_based = True  # Has BaseHub inside
                extractors_in_hub = False  # Extractors are in the ARENA, not hub

            elif isinstance(instance, RandomTransform.Config):
                inner = instance.scene
                if isinstance(inner, BaseHub.Config):
                    # Pure hub map (no arena): extractors are in the hub
                    is_hub_based = True
                    extractors_in_hub = True
                    width = min(width, 21)
                    height = min(height, 21)

            elif isinstance(instance, BaseHub.Config):
                # Pure hub map: extractors are in the hub
                is_hub_based = True
                extractors_in_hub = True
                width = min(width, instance.hub_width)
                height = min(height, instance.hub_height)

    area = width * height

    # Estimate average distance to extractors
    # Key: use extractors_in_hub, not is_hub_based
    estimated_distance = _estimate_avg_extractor_distance(
        width, height, building_coverage, distribution_type, extractors_in_hub
    )

    # Estimate extractor counts
    if extractors_in_hub:
        # Hub maps: exactly 1 of each extractor in corners
        extractor_counts = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}
    else:
        # Arena maps: estimate from building coverage
        # Typical: ~1% coverage, ~25% of buildings are extractors of some type
        total_buildings = max(4, int(area * building_coverage))
        extractors_per_type = max(1, total_buildings // 8)  # 4 extractor types + chargers
        extractor_counts = {
            "carbon": extractors_per_type,
            "oxygen": extractors_per_type,
            "germanium": extractors_per_type,
            "silicon": extractors_per_type,
        }

    return SpatialAnalysis(
        map_width=width,
        map_height=height,
        map_area=area,
        building_coverage=building_coverage,
        distribution_type=distribution_type,
        estimated_avg_distance=estimated_distance,
        is_hub_based=is_hub_based,
        extractors_in_hub=extractors_in_hub,
        extractor_counts=extractor_counts,
    )


def _estimate_avg_extractor_distance(
    width: int, height: int, coverage: float, distribution: str, extractors_in_hub: bool
) -> float:
    """Estimate average distance to extractors based on distribution.

    Calibrated against rigorous pathfinding validation:
    - Hub maps (extractors_in_hub): ~8 tiles avg distance
    - 100x100 arenas: ~30 tiles avg distance
    - 200x200 arenas: ~25-30 tiles avg distance (more extractors, similar density)

    Key insight: Arena distances converge around 25-30 regardless of size
    when coverage is typical (0.01). This is because more extractors are
    placed on larger maps, keeping density roughly constant.

    The greedy TSP path visits nearest extractors first, so actual travel
    is less than naive "map_size/3" would suggest.
    """
    if extractors_in_hub:
        # Extractors are in the hub corners, ~6-8 tiles from center
        return 8.0

    # For arenas, distances are relatively stable due to coverage scaling
    # Base: ~25-30 for typical arenas regardless of size
    # Adjust slightly for distribution type

    base_distance = 28.0  # Empirically calibrated from rigorous validation

    if distribution == "uniform":
        dist_factor = 1.0
    elif distribution == "normal":
        # Clustered toward center - shorter if you're near center
        dist_factor = 0.8
    elif distribution == "bimodal":
        # Two clusters
        dist_factor = 0.9
    elif distribution == "exponential":
        # Concentrated at origin
        dist_factor = 0.85
    else:
        dist_factor = 1.0

    return base_distance * dist_factor


# =============================================================================
# Exploration Analysis
# =============================================================================

# Standard agent count for normalized comparisons
STANDARD_NUM_AGENTS = 4


def _analyze_exploration(
    mission: Mission,
    spatial: SpatialAnalysis,
    resources: ResourceAnalysis,
) -> ExplorationAnalysis:
    """Analyze exploration requirements using optimal exploration model.

    Even an oracle agent must *discover* extractors before using them.
    This models the minimum steps needed to have "seen" all required extractor types.

    Uses a greedy covering approach:
    - Agent starts at center (near assembler)
    - Vision radius of 5 tiles
    - Visits locations to maximize discovery

    For multi-agent: exploration scales nearly linearly with agent count
    (4 agents explore ~3.5x faster accounting for overlap).
    """
    vision_radius = 5
    num_agents = STANDARD_NUM_AGENTS
    area = spatial.map_area

    # How many tiles does one agent "see" per position?
    # Vision is a diamond of radius 5: area ≈ 2 * r^2
    tiles_per_view = 2 * vision_radius * vision_radius  # ~50 tiles

    # What fraction of map needs exploring to find all extractor types?
    if spatial.extractors_in_hub:
        # Hub is small, extractors are predictable in corners
        # Need to explore ~60% of hub to find all 4 corners
        exploration_fraction = 0.6
    else:
        # Arena: extractors are scattered uniformly
        # Need to explore enough to statistically find at least 1 of each type
        # With n extractors of each type in area A, expected search to find one: A/n
        # For 4 types, multiply by coverage overlap factor

        total_extractors = sum(spatial.extractor_counts.values())
        if total_extractors > 0:
            # Probability of finding all 4 types requires significant coverage
            # Use coupon collector approximation: E[steps] ≈ n * ln(n) for n types
            # For 4 types: ~5.5 * area_per_extractor
            avg_extractors_per_type = total_extractors / 4
            if avg_extractors_per_type > 0:
                # Fraction of map to cover to expect finding 1 of each type
                # ~2.5 / density for 4 types (coupon collector)
                exploration_fraction = min(0.8, 2.5 / avg_extractors_per_type)
            else:
                exploration_fraction = 0.8
        else:
            exploration_fraction = 0.8

    # Total tiles to explore
    tiles_to_explore = area * exploration_fraction

    # Steps for single agent to explore (each step reveals ~tiles_per_view new tiles)
    # But adjacent moves overlap, so effective new tiles per step is less
    # Assume ~60% efficiency due to overlap and backtracking
    efficiency = 0.6
    effective_tiles_per_step = tiles_per_view * efficiency

    single_agent_steps = int(tiles_to_explore / max(1, effective_tiles_per_step))

    # Multi-agent exploration speedup
    # 4 agents explore ~3.5x faster (not 4x due to potential overlap in random exploration)
    # Optimal coordination would be closer to 4x
    multi_agent_speedup = num_agents * 0.85  # 85% efficiency per additional agent
    multi_agent_steps = int(single_agent_steps / multi_agent_speedup)

    return ExplorationAnalysis(
        vision_radius=vision_radius,
        num_agents=num_agents,
        single_agent_discovery=single_agent_steps,
        multi_agent_discovery=multi_agent_steps,
        map_exploration_fraction=exploration_fraction,
        discovery_efficiency=efficiency,
    )


# =============================================================================
# Conflict Detection
# =============================================================================


def _detect_conflicts(
    mission: Mission,
    energy: EnergyAnalysis,
    resources: ResourceAnalysis,
    recipe: RecipeAnalysis,
    initial: InitialResourceAnalysis,
) -> list[str]:
    """Detect impossible or problematic configurations."""
    conflicts: list[str] = []

    # Energy death: no regen, no chargers, high move cost
    if energy.regen_per_step == 0 and not energy.charger_available:
        if energy.sustainable_steps < 20:
            conflicts.append("Energy death: zero regen, no chargers, will die in <20 steps")

    # All extractors clipped with no unclip capability
    all_clipped = (
        mission.carbon_extractor.start_clipped
        and mission.oxygen_extractor.start_clipped
        and mission.germanium_extractor.start_clipped
        and mission.silicon_extractor.start_clipped
    )
    if all_clipped and mission.clip_period == 0:
        conflicts.append("All extractors clipped with no clip_period for automatic unclipping")

    # Check if extractors have enough uses
    for resource in ["carbon", "oxygen", "germanium", "silicon"]:
        extractor = getattr(mission, f"{resource}_extractor", None)
        if extractor and extractor.max_uses > 0:
            needed_visits = initial.effective_extraction_need.get(resource, 0)
            output = resources.extractor_output.get(resource, 1)
            visits_needed = math.ceil(needed_visits / output) if output > 0 else needed_visits
            if visits_needed > extractor.max_uses:
                conflicts.append(
                    f"{resource} extractor max_uses ({extractor.max_uses}) < visits needed ({visits_needed})"
                )

    # Recipe requires more agents than available
    num_cogs = mission.num_cogs if mission.num_cogs is not None else mission.site.min_cogs
    if recipe.min_agents_required > num_cogs:
        conflicts.append(f"Recipe requires {recipe.min_agents_required} agents but only {num_cogs} available")

    # Cargo capacity check
    if recipe.cheapest_heart_cost:
        total_cargo = sum(
            v for k, v in recipe.cheapest_heart_cost.items() if k in ("carbon", "oxygen", "germanium", "silicon")
        )
        if total_cargo > mission.cargo_capacity:
            conflicts.append(f"Heart recipe ({total_cargo} cargo) exceeds cargo_capacity ({mission.cargo_capacity})")

    return conflicts


# =============================================================================
# Step Estimation
# =============================================================================


def _estimate_steps_to_heart(
    mission: Mission,
    energy: EnergyAnalysis,
    resources: ResourceAnalysis,
    recipe: RecipeAnalysis,
    initial: InitialResourceAnalysis | None,
    spatial: SpatialAnalysis | None,
    min_visits: dict[str, int],
) -> int:
    """Estimate minimum MOVEMENT steps to produce first heart.

    This returns pure movement steps. Energy overhead (waiting/chargers)
    is added separately by _calculate_total_steps.

    Calibrated against rigorous greedy TSP validation:
    - Hub maps: ~4 steps per extractor visit (extractors nearby)
    - Arena maps: First visit ~avg_distance, subsequent visits much shorter
      because greedy finds nearest unvisited extractor

    With many extractors (50+ per type), subsequent visits average ~10-15 steps,
    not the full avg_distance. The greedy path is very efficient.
    """
    if initial is not None and initial.first_heart_covered:
        # Just need to walk to assembler
        return int(spatial.estimated_avg_distance * 2) if spatial else 10

    total_visits = sum(min_visits.values())
    avg_distance = spatial.estimated_avg_distance if spatial else 10

    if spatial and spatial.extractors_in_hub:
        # Hub: extractors are close together, ~4-6 steps between them
        steps_per_visit = 4.0
        final_trip = 4.0  # To chest + assembler
    else:
        # Arena: first visit is far, subsequent are closer (greedy nearest neighbor)
        # Empirically: ~12 steps per visit on dense arena maps
        # Scale slightly with avg_distance for sparse maps
        steps_per_visit = min(avg_distance * 0.4, 15.0)  # Cap at 15
        final_trip = avg_distance * 0.3  # Trip back to hub

    movement_steps = int(total_visits * steps_per_visit + final_trip)

    return max(10, movement_steps)


# =============================================================================
# Scoring (Ratio-based, unbounded)
# =============================================================================


def _calculate_spatial_complexity(spatial: SpatialAnalysis) -> float:
    """Calculate spatial complexity score (0-1)."""
    # Larger maps = more complex
    area_factor = min(1.0, spatial.map_area / 40000)  # 200x200 = 1.0

    # Lower coverage = harder to find resources
    coverage_factor = 1.0 - min(1.0, spatial.building_coverage * 20)  # 0.05 coverage = 0

    # Non-uniform distributions can be harder
    dist_factor = 0.0
    if spatial.distribution_type in ("bimodal", "exponential"):
        dist_factor = 0.2

    # Hub-based maps are easier
    hub_factor = -0.3 if spatial.is_hub_based else 0.0

    complexity = (area_factor * 0.4 + coverage_factor * 0.4 + dist_factor + hub_factor)
    return max(0.0, min(1.0, complexity))


def _calculate_energy_difficulty(
    energy: EnergyAnalysis,
    estimated_steps: int,
    resources: ResourceAnalysis,
) -> float:
    """Calculate energy difficulty as a ratio.

    Key insight: If regen > 0, agents can always wait to recharge between steps.
    If regen = 0 but charger available, agents are at HIGH RISK - any suboptimal
    pathing can strand them with no way to recover.
    Only truly impossible if regen = 0 AND no chargers (finite energy, no recovery).

    Returns:
        1.0 = trivial (energy positive or can wait-recharge)
        2.0+ = significant time cost due to waiting/recharging
        5.0+ = charger-dependent with zero regen (high risk)
        inf = impossible (zero regen AND no chargers)
    """
    if energy.energy_positive:
        # Energy positive = trivial energy management
        return 1.0

    can_wait_recharge = energy.regen_per_step > 0

    # Calculate energy costs
    movement_energy = estimated_steps * energy.move_cost
    extraction_energy = resources.extraction_energy_cost
    total_energy_needed = movement_energy + extraction_energy

    if not can_wait_recharge and not energy.charger_available:
        # TRULY IMPOSSIBLE: No regen, no chargers = finite energy pool
        max_energy_ever = energy.capacity
        if total_energy_needed > max_energy_ever:
            return float("inf")
        # Even if theoretically possible, it's extremely constrained
        energy_ratio = total_energy_needed / max_energy_ever
        if energy_ratio > 0.8:
            return float("inf")  # Too tight, effectively impossible
        # Very high difficulty even if technically possible
        return 10 ** energy_ratio  # 0.5 ratio = 3.2x, 0.8 ratio = 6.3x

    if not can_wait_recharge and energy.charger_available:
        # ZERO REGEN + CHARGER DEPENDENT: High risk scenario
        # Any wrong move could strand you with no recovery possible
        # This is fundamentally harder than having any positive regen

        steps_per_charge = energy.capacity // energy.move_cost
        charger_trip_cost = 20  # steps to reach charger and return
        charger_trip_energy = charger_trip_cost * energy.move_cost

        # Can we even afford to reach the charger?
        if charger_trip_energy > energy.capacity * 0.8:
            # Charger trip costs almost all our energy - extremely risky
            return 10.0

        effective_range = steps_per_charge - charger_trip_cost
        if effective_range <= 0:
            # Can't even complete one useful action between charges
            return float("inf")

        # Calculate how many charger visits needed
        charger_visits = max(1, estimated_steps // effective_range)

        # Base difficulty: must succeed at every charger visit or die
        # Each visit is a risk point - compound the risk
        # Formula: base * (1 + risk_per_visit)^visits
        risk_per_visit = 0.3  # 30% "risk factor" per mandatory charger visit
        risk_multiplier = (1 + risk_per_visit) ** min(charger_visits, 10)  # Cap at 10 visits

        # Frequency penalty
        if effective_range < 20:
            frequency_penalty = 3.0  # Must charge very often
        elif effective_range < 50:
            frequency_penalty = 2.0  # Moderate charging
        else:
            frequency_penalty = 1.5  # Can go a while between charges

        return frequency_penalty * risk_multiplier

    # POSITIVE REGEN: Can always wait to recover - much safer
    # Calculate time cost
    net_cost_per_step = energy.move_cost - energy.regen_per_step

    if net_cost_per_step <= 0:
        # Already handled by energy_positive check, but safety
        return 1.0

    # Wait ratio: how many steps of waiting per step of movement
    wait_ratio = net_cost_per_step / energy.regen_per_step

    # Time multiplier: 1.0 = no waiting, 2.0 = wait as long as you move
    time_multiplier = 1.0 + wait_ratio

    # Factor in charger availability (reduces waiting)
    if energy.charger_available:
        time_multiplier = max(1.0, time_multiplier * 0.5)

    # Convert time multiplier to difficulty
    # Scale: 1.0 time = 1.0 difficulty, 4.0 time = 2.0 difficulty
    return 1.0 + math.log2(max(1.0, time_multiplier)) * 0.5


def _calculate_extraction_difficulty(
    resources: ResourceAnalysis,
    initial: InitialResourceAnalysis,
    energy: EnergyAnalysis,
) -> float:
    """Calculate extraction difficulty as a ratio.

    Considers:
    - Number of extractor visits needed
    - Energy cost of silicon extraction
    - Whether initial inventory reduces needs

    Returns:
        1.0 = trivial (first heart covered or minimal extraction)
        2.0+ = significant extraction required
        inf = extraction energy exceeds energy capacity
    """
    if initial.first_heart_covered:
        return 1.0

    # Check if silicon extraction is even possible
    silicon_energy_cost = resources.extraction_energy_cost
    if silicon_energy_cost > 0:
        # Can we afford the energy for silicon extraction?
        if not energy.energy_positive:
            max_energy_for_extraction = energy.capacity  # Assume we can dedicate full capacity
            if silicon_energy_cost > max_energy_for_extraction * 2:
                # Need more than 2x our capacity for extraction alone = very hard
                if not energy.charger_available:
                    return float("inf")

    # Calculate effective visits needed
    effective_visits = sum(
        max(0, math.ceil(need / resources.extractor_output.get(res, 1)))
        for res, need in initial.effective_extraction_need.items()
        if resources.extractor_output.get(res, 1) > 0
    )

    # Baseline: 4 visits (one per resource type) = 1.0 difficulty
    # Scale logarithmically: 8 visits = 1.5x, 16 visits = 2x, etc.
    baseline_visits = 4
    if effective_visits <= baseline_visits:
        visit_factor = 1.0
    else:
        visit_factor = 1.0 + math.log2(effective_visits / baseline_visits)

    # Energy cost factor for silicon
    if silicon_energy_cost > 0 and energy.capacity > 0:
        energy_factor = 1.0 + (silicon_energy_cost / energy.capacity) * 0.5
    else:
        energy_factor = 1.0

    return visit_factor * energy_factor


def _calculate_recipe_difficulty(recipe: RecipeAnalysis) -> float:
    """Calculate recipe difficulty as a ratio.

    Returns:
        1.0 = single agent, simple recipe
        1.5 = multi-agent coordination needed
        2.0+ = complex vibe synchronization required
    """
    base = 1.0

    # Coordination adds difficulty
    # coordination_difficulty is 0-1, map to 1.0-2.0 multiplier
    coordination_factor = 1.0 + recipe.coordination_difficulty

    # Not all recipes accessible adds some difficulty
    if not recipe.all_recipes_accessible:
        coordination_factor *= 1.1

    # More agents required = harder coordination
    if recipe.min_agents_required > 1:
        coordination_factor *= 1.0 + (recipe.min_agents_required - 1) * 0.2

    return base * coordination_factor


def _calculate_spatial_difficulty(spatial: SpatialAnalysis | None) -> float:
    """Calculate spatial difficulty as a ratio.

    Returns:
        1.0 = small hub-based map
        2.0 = medium map with good coverage
        4.0+ = large sparse map
    """
    if spatial is None:
        return 1.5  # Unknown = slightly elevated

    base = 1.0

    # Map size factor: 100x100 = baseline
    # Smaller = easier, larger = harder
    baseline_area = 10000  # 100x100
    area_ratio = spatial.map_area / baseline_area
    if area_ratio <= 1.0:
        size_factor = 1.0
    else:
        # Log scale: 400x400 (16x area) = 2x difficulty
        size_factor = 1.0 + math.log2(area_ratio) * 0.25

    # Coverage factor: lower coverage = harder to find buildings
    # 0.01 coverage = baseline, lower = harder
    baseline_coverage = 0.01
    if spatial.building_coverage >= baseline_coverage:
        coverage_factor = 1.0
    elif spatial.building_coverage > 0:
        coverage_factor = baseline_coverage / spatial.building_coverage
    else:
        coverage_factor = 10.0  # No buildings = very hard

    # Hub-based maps are predictable = easier
    if spatial.is_hub_based:
        hub_factor = 0.5  # 50% easier
    else:
        hub_factor = 1.0

    # Distribution type
    dist_factor = 1.0
    if spatial.distribution_type in ("bimodal", "exponential"):
        dist_factor = 1.2  # Clustered resources = slightly harder on average

    return base * size_factor * coverage_factor * hub_factor * dist_factor

