"""
Game Statistics Analysis Module

This module provides comprehensive analysis of mettagrid replay statistics, including:
- Agent behavioral analysis with statistical methods
- Resource flow and production/consumption pattern analysis
- Combat interaction matrices for multi-agent dynamics
- Building efficiency scoring for economic strategy analysis
- Strategic phase detection identifying behavioral transitions
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AgentStats:
    """Container for parsed agent statistics"""

    agent_id: int
    total_actions: Dict[str, float]
    action_success_rates: Dict[str, float]
    resource_flows: Dict[str, Dict[str, float]]  # {item: {gained/lost: amount}}
    movement_patterns: Dict[str, float]
    combat_stats: Dict[str, float]
    building_interactions: Dict[str, float]
    efficiency_metrics: Dict[str, float]


@dataclass
class BuildingStats:
    """Container for parsed building/converter statistics"""

    building_id: int
    type_id: int
    type_name: str
    location: Tuple[int, int]
    production_efficiency: Dict[str, float]
    resource_flows: Dict[str, Dict[str, float]]  # {item: {produced/consumed: amount}}
    operational_stats: Dict[str, float]
    bottleneck_analysis: Dict[str, float]


@dataclass
class GameStatsAnalysis:
    """Complete game statistics analysis results"""

    agent_stats: List[AgentStats]
    building_stats: List[BuildingStats]
    comparative_analysis: Dict[str, Any]
    resource_flow_matrix: Dict[str, Any]
    combat_interaction_matrix: Dict[str, Any]
    building_efficiency_scores: Dict[str, float]
    strategic_phases: List[Dict[str, Any]]
    global_metrics: Dict[str, float]


class StatsExtractor:
    """Extracts and parses mettagrid episode statistics from replay data"""

    def __init__(self):
        self.action_prefixes = {"action.", "movement.", "status.", "box.", "conversions.", "cooldown.", "blocked."}
        self.resource_suffixes = {".gained", ".lost", ".produced", ".consumed", ".put", ".get", ".added", ".removed"}

    def extract_from_replay_data(self, replay_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract episode statistics from replay data.

        Args:
            replay_data: Parsed replay JSON data

        Returns:
            Dictionary containing structured episode stats

        Raises:
            ValueError: If replay data doesn't contain required statistics
        """
        # Simulate episode stats extraction since we don't have access to the C++ method
        # In real implementation, this would call mettagrid's get_episode_stats()

        # Check if replay contains the required statistical data
        if not self._validate_replay_stats(replay_data):
            raise ValueError("Replay data missing required statistical information")

        # For now, return a structured stats format based on what we know the C++ system provides
        episode_stats = {
            "game": self._extract_global_stats(replay_data),
            "agent": self._extract_agent_stats(replay_data),
            "converter": self._extract_building_stats(replay_data),
        }

        return episode_stats

    def _validate_replay_stats(self, replay_data: Dict[str, Any]) -> bool:
        """Validate that replay data contains statistical information"""
        # Check for objects format or grid_objects format
        has_objects = "objects" in replay_data or "grid_objects" in replay_data
        has_agents = False

        if "objects" in replay_data:
            # Check for agent objects with statistical data
            for obj in replay_data["objects"]:
                if obj.get("agent_id") is not None:
                    has_agents = True
                    break
        elif "grid_objects" in replay_data:
            # Check for agent objects
            for obj in replay_data["grid_objects"]:
                if obj.get("type") == 0:  # Agent type
                    has_agents = True
                    break

        return has_objects and has_agents

    def _extract_global_stats(self, replay_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract game-level statistics"""
        # Get actual data to calculate meaningful metrics
        total_agents = replay_data.get("num_agents", 0)
        episode_length = replay_data.get("max_steps", 0)

        # Calculate resource scarcity based on available resource sources vs agents
        resource_buildings = 0
        agent_count = 0

        # Count resource-producing buildings and agents
        if "grid_objects" in replay_data:
            for obj in replay_data["grid_objects"]:
                obj_type = obj.get("type", 0)
                # Count agents (type 0)
                if obj_type == 0:
                    agent_count += 1
                # Count resource buildings: mines (2,3,4), generators (5,6,7)
                elif obj_type in [2, 3, 4, 5, 6, 7]:
                    resource_buildings += 1

        # Calculate scarcity index: ratio of agents to resource sources
        # Higher values indicate more resource scarcity
        resource_scarcity = agent_count / max(resource_buildings, 1) if resource_buildings > 0 else float(agent_count)

        return {
            "episode_length": episode_length,
            "total_agents": total_agents,
            "resource_scarcity_index": resource_scarcity,
            "cooperation_index": 0.5,  # Would need interaction analysis
            "competition_intensity": min(resource_scarcity / 2.0, 1.0),  # Normalized scarcity
        }

    def _extract_agent_stats(self, replay_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract per-agent statistics from replay data"""
        agent_stats = []
        action_names = replay_data.get("action_names", [])
        inventory_items = replay_data.get("inventory_items", [])

        # Get agent objects based on format
        if "objects" in replay_data:
            agents = [obj for obj in replay_data["objects"] if obj.get("agent_id") is not None]
        else:
            agents = [obj for obj in replay_data.get("grid_objects", []) if obj.get("type") == 0]

        for i, agent in enumerate(agents):
            # Extract basic stats from agent data structure
            stats = {
                "agent_id": agent.get("agent_id", i),
                "total_reward": self._extract_final_reward(agent),
                "action_counts": self._count_agent_actions(agent, action_names),
                "resource_transactions": self._extract_resource_flows(agent, inventory_items, is_building=False),
                "movement_stats": self._extract_movement_patterns(agent),
                "interaction_stats": {},  # Placeholder for combat/social stats
            }
            agent_stats.append(stats)

        return agent_stats

    def _extract_building_stats(self, replay_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract per-building/converter statistics"""
        building_stats = []
        inventory_items = replay_data.get("inventory_items", [])
        # Support both type_names and object_types fields
        type_names = replay_data.get("type_names", replay_data.get("object_types", []))

        # Extract building objects (non-agent objects)
        if "objects" in replay_data:
            buildings = [
                obj for obj in replay_data["objects"] if obj.get("agent_id") is None and obj.get("type_id", 0) > 1
            ]
        else:
            buildings = [
                obj for obj in replay_data.get("grid_objects", []) if obj.get("type", 0) > 1
            ]  # Exclude agents (type=0) and walls (type=1)

        for i, building in enumerate(buildings):
            type_id = building.get("type_id", building.get("type", 0))
            type_name = type_names[type_id] if 0 <= type_id < len(type_names) else f"type_{type_id}"

            stats = {
                "building_id": building.get("id", i),
                "type_id": type_id,
                "type_name": type_name,
                "location": self._extract_location(building),
                "production_stats": {},  # Could be enhanced with conversion data
                "resource_flows": self._extract_resource_flows(
                    building, inventory_items, is_building=True
                ),  # Extract actual resource flows!
                "operational_stats": {},
                "utilization_rate": 0.0,
                "bottleneck_score": 0.0,
            }
            building_stats.append(stats)

        return building_stats

    def _extract_final_reward(self, agent: Dict[str, Any]) -> float:
        """Extract final reward from agent data"""
        if "total_reward" in agent:
            total_reward = agent["total_reward"]
            # Handle temporal array format: [[step, value], ...]
            if isinstance(total_reward, list) and total_reward:
                final_entry = total_reward[-1]
                return final_entry[1] if isinstance(final_entry, list) and len(final_entry) >= 2 else final_entry
            # Handle scalar format (current yudhister_replay.json format)
            elif isinstance(total_reward, (int, float)):
                return float(total_reward)
        return 0.0

    def _count_agent_actions(self, agent: Dict[str, Any], action_names: Optional[List[str]] = None) -> Dict[str, int]:
        """Count actions taken by agent"""
        action_counts = defaultdict(int)

        # Count from action data
        action_data = agent.get("action", []) or agent.get("action_id", [])
        for action_entry in action_data:
            if isinstance(action_entry, list) and len(action_entry) >= 2:
                action_spec = action_entry[1]
                # Handle different action formats
                if isinstance(action_spec, list) and len(action_spec) >= 1:
                    # Format: [action_id, arg]
                    action_id = action_spec[0]
                    action_name = (
                        action_names[action_id]
                        if action_names and action_id < len(action_names)
                        else f"action_{action_id}"
                    )
                    action_counts[action_name] += 1
                elif isinstance(action_spec, (int, float)):
                    # Format: action_id
                    action_id = int(action_spec)
                    action_name = (
                        action_names[action_id]
                        if action_names and action_id < len(action_names)
                        else f"action_{action_id}"
                    )
                    action_counts[action_name] += 1

        return dict(action_counts)

    def _extract_resource_flows(
        self, entity: Dict[str, Any], inventory_items: Optional[List[str]] = None, is_building: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Extract resource gain/loss patterns for agents or production/consumption for buildings"""
        if is_building:
            resource_flows = defaultdict(lambda: {"produced": 0.0, "consumed": 0.0})
        else:
            resource_flows = defaultdict(lambda: {"gained": 0.0, "lost": 0.0})

        # Extract from inventory changes
        inventory = entity.get("inventory", [])
        prev_inventory = {}

        for entry in inventory:
            if len(entry) < 2:
                continue

            _, curr_inv = entry[0], entry[1]

            if isinstance(curr_inv, list):
                # Objects format: [[item_id, count], ...]
                curr_inv_dict = {item[0]: item[1] for item in curr_inv if len(item) >= 2}
            elif isinstance(curr_inv, dict):
                # Grid_objects format: {item_id_str: count}
                curr_inv_dict = {int(k): v for k, v in curr_inv.items()}
            else:
                continue

            # Calculate changes
            for item_id, count in curr_inv_dict.items():
                prev_count = prev_inventory.get(item_id, 0)
                change = count - prev_count

                # Get meaningful item name
                item_name = (
                    inventory_items[item_id]
                    if inventory_items and item_id < len(inventory_items)
                    else f"item_{item_id}"
                )

                if is_building:
                    # For buildings: increases = production, decreases = consumption
                    if change > 0:
                        resource_flows[item_name]["produced"] += change
                    elif change < 0:
                        resource_flows[item_name]["consumed"] += abs(change)
                else:
                    # For agents: increases = gained, decreases = lost
                    if change > 0:
                        resource_flows[item_name]["gained"] += change
                    elif change < 0:
                        resource_flows[item_name]["lost"] += abs(change)

            prev_inventory = curr_inv_dict

        return dict(resource_flows)

    def _extract_movement_patterns(self, agent: Dict[str, Any]) -> Dict[str, int]:
        """Extract movement pattern statistics"""
        movement_stats = defaultdict(int)

        # Count location changes
        location_data = agent.get("location", [])
        prev_location = None

        for entry in location_data:
            if len(entry) < 2:
                continue

            _, location = entry[0], entry[1]

            if prev_location is not None:
                # Calculate direction of movement
                dr = location[0] - prev_location[0]
                dc = location[1] - prev_location[1]

                if dr == 0 and dc == 0:
                    movement_stats["stationary"] += 1
                elif dr == -1 and dc == 0:
                    movement_stats["up"] += 1
                elif dr == 1 and dc == 0:
                    movement_stats["down"] += 1
                elif dr == 0 and dc == -1:
                    movement_stats["left"] += 1
                elif dr == 0 and dc == 1:
                    movement_stats["right"] += 1
                else:
                    movement_stats["diagonal"] += 1

            prev_location = location

        return dict(movement_stats)

    def _extract_location(self, building: Dict[str, Any]) -> Tuple[int, int]:
        """Extract location from building object"""
        if "location" in building:
            loc = building["location"]
            if isinstance(loc, list) and len(loc) >= 2:
                return (loc[0], loc[1])
        return (0, 0)


class BehavioralAnalysisEngine:
    """Statistical analysis engine for agent behaviors"""

    def __init__(self):
        self.min_sample_size = 5  # Minimum actions for statistical analysis

    def analyze_agent_behaviors(self, agent_stats: List[AgentStats]) -> Dict[str, Any]:
        """
        Perform comprehensive behavioral analysis on agent statistics

        Args:
            agent_stats: List of parsed agent statistics

        Returns:
            Dictionary containing behavioral analysis results

        Raises:
            ValueError: If insufficient data for analysis
        """
        if not agent_stats:
            raise ValueError("No agent statistics provided for behavioral analysis")

        analysis = {
            "efficiency_rankings": self._rank_agent_efficiency(agent_stats),
            "behavioral_clusters": self._cluster_behaviors(agent_stats),
            "strategy_identification": self._identify_strategies(agent_stats),
            "performance_correlations": self._analyze_performance_correlations(agent_stats),
            "outlier_detection": self._detect_behavioral_outliers(agent_stats),
        }

        return analysis

    def _rank_agent_efficiency(self, agent_stats: List[AgentStats]) -> List[Dict[str, Any]]:
        """Rank agents by various efficiency metrics"""
        rankings = []

        # Calculate efficiency scores
        for agent in agent_stats:
            efficiency_score = 0.0

            # Resource efficiency (gained/lost ratio)
            total_gained = sum(flows.get("gained", 0) for flows in agent.resource_flows.values())
            total_lost = sum(flows.get("lost", 0) for flows in agent.resource_flows.values())
            resource_efficiency = total_gained / max(total_lost, 1)

            # Action success rate
            total_successes = sum(agent.action_success_rates.values())
            total_actions = sum(agent.total_actions.values())
            action_efficiency = total_successes / max(total_actions, 1)

            # Combined efficiency score
            efficiency_score = (resource_efficiency * 0.6) + (action_efficiency * 0.4)

            rankings.append(
                {
                    "agent_id": agent.agent_id,
                    "efficiency_score": efficiency_score,
                    "resource_efficiency": resource_efficiency,
                    "action_efficiency": action_efficiency,
                }
            )

        # Sort by efficiency score
        rankings.sort(key=lambda x: x["efficiency_score"], reverse=True)
        return rankings

    def _cluster_behaviors(self, agent_stats: List[AgentStats]) -> Dict[str, List[int]]:
        """Cluster agents by behavioral patterns"""
        # Simple clustering based on dominant strategies
        clusters = {"resource_focused": [], "combat_focused": [], "exploration_focused": [], "balanced": []}

        for agent in agent_stats:
            # Calculate behavioral weights
            resource_weight = sum(flows.get("gained", 0) for flows in agent.resource_flows.values())
            combat_weight = sum(v for k, v in agent.combat_stats.items() if "hit" in k or "steal" in k)
            exploration_weight = sum(v for k, v in agent.movement_patterns.items())

            total_weight = resource_weight + combat_weight + exploration_weight
            if total_weight == 0:
                clusters["balanced"].append(agent.agent_id)
                continue

            # Normalize weights
            resource_pct = resource_weight / total_weight
            combat_pct = combat_weight / total_weight
            exploration_pct = exploration_weight / total_weight

            # Classify based on dominant behavior
            if resource_pct > 0.5:
                clusters["resource_focused"].append(agent.agent_id)
            elif combat_pct > 0.4:
                clusters["combat_focused"].append(agent.agent_id)
            elif exploration_pct > 0.4:
                clusters["exploration_focused"].append(agent.agent_id)
            else:
                clusters["balanced"].append(agent.agent_id)

        return clusters

    def _identify_strategies(self, agent_stats: List[AgentStats]) -> Dict[int, str]:
        """Identify strategic patterns for each agent"""
        strategies = {}

        for agent in agent_stats:
            # Analyze action patterns to identify strategies
            strategy = "unknown"

            # Check for specialist strategies
            total_resource_actions = sum(flows.get("gained", 0) for flows in agent.resource_flows.values())
            total_combat_actions = sum(v for k, v in agent.combat_stats.items() if "attack" in k)

            if total_resource_actions > total_combat_actions * 3:
                strategy = "resource_specialist"
            elif total_combat_actions > total_resource_actions * 2:
                strategy = "combat_specialist"
            elif agent.efficiency_metrics.get("cooperation_score", 0) > 0.7:
                strategy = "cooperative"
            elif agent.efficiency_metrics.get("competition_score", 0) > 0.7:
                strategy = "competitive"
            else:
                strategy = "generalist"

            strategies[agent.agent_id] = strategy

        return strategies

    def _analyze_performance_correlations(self, agent_stats: List[AgentStats]) -> Dict[str, float]:
        """Analyze correlations between different performance metrics"""
        if len(agent_stats) < 3:
            return {}

        correlations = {}

        # Extract metric arrays
        efficiency_scores = []
        resource_gains = []
        action_success_rates = []

        for agent in agent_stats:
            # Calculate composite scores
            total_gained = sum(flows.get("gained", 0) for flows in agent.resource_flows.values())
            avg_success_rate = statistics.mean(agent.action_success_rates.values()) if agent.action_success_rates else 0

            efficiency_scores.append(agent.efficiency_metrics.get("overall_efficiency", 0))
            resource_gains.append(total_gained)
            action_success_rates.append(avg_success_rate)

        # Calculate correlations (simplified Pearson correlation)
        if len(set(efficiency_scores)) > 1 and len(set(resource_gains)) > 1:
            correlations["efficiency_vs_resource_gain"] = self._pearson_correlation(efficiency_scores, resource_gains)

        if len(set(efficiency_scores)) > 1 and len(set(action_success_rates)) > 1:
            correlations["efficiency_vs_action_success"] = self._pearson_correlation(
                efficiency_scores, action_success_rates
            )

        return correlations

    def _detect_behavioral_outliers(self, agent_stats: List[AgentStats]) -> List[Dict[str, Any]]:
        """Detect agents with unusual behavioral patterns"""
        outliers = []

        if len(agent_stats) < 3:
            return outliers

        # Calculate z-scores for key metrics
        efficiency_scores = [agent.efficiency_metrics.get("overall_efficiency", 0) for agent in agent_stats]
        # Note: resource_scores could be used for additional outlier detection if needed

        if len(set(efficiency_scores)) > 1:
            eff_mean = statistics.mean(efficiency_scores)
            eff_stdev = statistics.stdev(efficiency_scores)

            for i, agent in enumerate(agent_stats):
                z_score = abs(efficiency_scores[i] - eff_mean) / max(eff_stdev, 0.01)
                if z_score > 2.0:  # Significant outlier
                    outliers.append(
                        {
                            "agent_id": agent.agent_id,
                            "metric": "efficiency",
                            "z_score": z_score,
                            "description": "Unusually high" if efficiency_scores[i] > eff_mean else "Unusually low",
                        }
                    )

        return outliers

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5

        return numerator / max(denominator, 0.001)


class ResourceFlowAnalyzer:
    """Analyzes resource production, consumption, and flow patterns"""

    def analyze_resource_flows(
        self, agent_stats: List[AgentStats], building_stats: List[BuildingStats]
    ) -> Dict[str, Any]:
        """
        Analyze resource flow patterns across agents and buildings

        Args:
            agent_stats: List of agent statistics
            building_stats: List of building statistics

        Returns:
            Comprehensive resource flow analysis

        Raises:
            ValueError: If insufficient data for flow analysis
        """
        if not agent_stats and not building_stats:
            raise ValueError("No statistics provided for resource flow analysis")

        analysis = {
            "resource_flow_matrix": self._build_flow_matrix(agent_stats, building_stats),
            "production_efficiency": self._analyze_production_efficiency(building_stats),
            "consumption_patterns": self._analyze_consumption_patterns(agent_stats),
            "bottleneck_identification": self._identify_bottlenecks(agent_stats, building_stats),
            "resource_scarcity_analysis": self._analyze_resource_scarcity(agent_stats, building_stats),
        }

        return analysis

    def _build_flow_matrix(self, agent_stats: List[AgentStats], building_stats: List[BuildingStats]) -> Dict[str, Any]:
        """Build resource flow matrix showing sources and sinks"""
        flow_matrix = {
            "sources": defaultdict(float),  # Where resources come from
            "sinks": defaultdict(float),  # Where resources go
            "net_flows": defaultdict(float),  # Net flow per resource type
        }

        # Aggregate agent resource flows
        for agent in agent_stats:
            for resource, flows in agent.resource_flows.items():
                flow_matrix["sources"][resource] += flows.get("gained", 0)
                flow_matrix["sinks"][resource] += flows.get("lost", 0)
                flow_matrix["net_flows"][resource] += flows.get("gained", 0) - flows.get("lost", 0)

        # Aggregate building production/consumption
        for building in building_stats:
            for resource, flows in building.resource_flows.items():
                flow_matrix["sources"][resource] += flows.get("produced", 0)
                flow_matrix["sinks"][resource] += flows.get("consumed", 0)

        # Convert defaultdicts to regular dicts
        return {k: dict(v) for k, v in flow_matrix.items()}

    def _analyze_production_efficiency(self, building_stats: List[BuildingStats]) -> Dict[str, float]:
        """Analyze production efficiency of buildings"""
        efficiency_scores = {}

        for building in building_stats:
            # Skip block buildings - they don't produce anything useful
            if building.type_name == "block":
                continue

            total_produced = sum(flows.get("produced", 0) for flows in building.resource_flows.values())
            total_consumed = sum(flows.get("consumed", 0) for flows in building.resource_flows.values())

            # Production efficiency = output/input ratio
            efficiency = total_produced / max(total_consumed, 1)
            x, y = building.location
            building_key = f"{building.type_name} (#{building.building_id}) x={x}, y={y}"
            efficiency_scores[building_key] = efficiency

        return efficiency_scores

    def _analyze_consumption_patterns(self, agent_stats: List[AgentStats]) -> Dict[str, Any]:
        """Analyze resource consumption patterns across agents"""
        consumption_analysis = {
            "total_consumption": defaultdict(float),
            "consumption_per_agent": defaultdict(lambda: defaultdict(float)),
            "consumption_distribution": {},
        }

        for agent in agent_stats:
            for resource, flows in agent.resource_flows.items():
                lost = flows.get("lost", 0)
                consumption_analysis["total_consumption"][resource] += lost
                consumption_analysis["consumption_per_agent"][agent.agent_id][resource] = lost

        # Calculate consumption distribution statistics
        for resource in consumption_analysis["total_consumption"]:
            consumptions = [flows[resource] for flows in consumption_analysis["consumption_per_agent"].values()]
            if consumptions:
                consumption_analysis["consumption_distribution"][resource] = {
                    "mean": statistics.mean(consumptions),
                    "stdev": statistics.stdev(consumptions) if len(consumptions) > 1 else 0,
                    "max": max(consumptions),
                    "min": min(consumptions),
                }

        # Convert defaultdicts
        return {
            "total_consumption": dict(consumption_analysis["total_consumption"]),
            "consumption_per_agent": {k: dict(v) for k, v in consumption_analysis["consumption_per_agent"].items()},
            "consumption_distribution": consumption_analysis["consumption_distribution"],
        }

    def _identify_bottlenecks(
        self, agent_stats: List[AgentStats], building_stats: List[BuildingStats]
    ) -> List[Dict[str, Any]]:
        """Identify resource flow bottlenecks"""
        bottlenecks = []

        # Analyze building bottlenecks
        for building in building_stats:
            blocked_events = building.operational_stats.get("blocked_total", 0)
            total_attempts = building.operational_stats.get("conversion_attempts", 1)

            if blocked_events / total_attempts > 0.3:  # >30% blocked
                bottlenecks.append(
                    {
                        "type": "building_bottleneck",
                        "building_id": building.building_id,
                        "blocked_rate": blocked_events / total_attempts,
                        "primary_cause": self._identify_block_cause(building),
                    }
                )

        return bottlenecks

    def _analyze_resource_scarcity(
        self, agent_stats: List[AgentStats], building_stats: List[BuildingStats]
    ) -> Dict[str, float]:
        """Analyze resource scarcity levels"""
        scarcity_index = {}

        # Collect all resources that actually exist in the data
        all_resources = set()
        for agent in agent_stats:
            all_resources.update(agent.resource_flows.keys())
        for building in building_stats:
            all_resources.update(building.resource_flows.keys())

        # Default resource list if we don't find any (fallback)
        if not all_resources:
            all_resources = {"ore_red", "ore_blue", "ore_green", "battery_red", "battery_blue", "battery_green"}

        # Calculate demand vs supply for each resource
        for resource in all_resources:
            total_demand = sum(agent.resource_flows.get(resource, {}).get("lost", 0) for agent in agent_stats)
            total_supply = sum(
                building.resource_flows.get(resource, {}).get("produced", 0) for building in building_stats
            )

            # Scarcity = demand/supply ratio (higher = more scarce)
            if total_supply > 0:
                scarcity = total_demand / total_supply
            else:
                # No supply but there is demand = maximum scarcity
                scarcity = 10.0 if total_demand > 0 else 0.0

            scarcity_index[resource] = min(scarcity, 10.0)  # Cap at 10x

        # Ensure the standard resources are included even if no data
        for standard_resource in ["ore_red", "ore_blue", "ore_green", "battery_red", "battery_blue", "battery_green"]:
            if standard_resource not in scarcity_index:
                scarcity_index[standard_resource] = 0.0

        return scarcity_index

    def _identify_block_cause(self, building: BuildingStats) -> str:
        """Identify primary cause of building blocks"""
        operational = building.operational_stats

        output_blocked = operational.get("blocked.output_full", 0)
        input_blocked = operational.get("blocked.insufficient_input", 0)

        if output_blocked > input_blocked:
            return "output_full"
        elif input_blocked > 0:
            return "insufficient_input"
        else:
            return "unknown"


class CombatInteractionAnalyzer:
    """Analyzes combat and social interactions between agents"""

    def analyze_combat_interactions(self, agent_stats: List[AgentStats]) -> Dict[str, Any]:
        """
        Analyze combat interactions and social dynamics

        Args:
            agent_stats: List of agent statistics

        Returns:
            Combat interaction analysis results

        Raises:
            ValueError: If insufficient data for combat analysis
        """
        if not agent_stats:
            raise ValueError("No agent statistics provided for combat analysis")

        analysis = {
            "interaction_matrix": self._build_interaction_matrix(agent_stats),
            "aggression_rankings": self._rank_aggression_levels(agent_stats),
            "cooperation_metrics": self._analyze_cooperation(agent_stats),
            "territorial_analysis": self._analyze_territorial_behavior(agent_stats),
            "social_network_metrics": self._calculate_social_metrics(agent_stats),
        }

        return analysis

    def _build_interaction_matrix(self, agent_stats: List[AgentStats]) -> Dict[str, Any]:
        """Build matrix of agent-to-agent interactions"""
        matrix = {
            "attacks": defaultdict(lambda: defaultdict(int)),
            "steals": defaultdict(lambda: defaultdict(int)),
            "cooperation": defaultdict(lambda: defaultdict(int)),
        }

        for agent in agent_stats:
            agent_id = agent.agent_id

            # Extract combat statistics
            for stat_name, value in agent.combat_stats.items():
                if "hit." in stat_name and value > 0:
                    target = stat_name.split("hit.")[1] if "hit." in stat_name else "unknown"
                    matrix["attacks"][agent_id][target] += value
                elif "steals." in stat_name and value > 0:
                    # Parse "group.steals.item.from.target" format
                    parts = stat_name.split(".")
                    if len(parts) >= 5 and parts[3] == "from":
                        target = parts[4]
                        matrix["steals"][agent_id][target] += value

        # Convert defaultdicts to regular dicts
        return {
            interaction_type: {str(k): dict(v) for k, v in interactions.items()}
            for interaction_type, interactions in matrix.items()
        }

    def _rank_aggression_levels(self, agent_stats: List[AgentStats]) -> List[Dict[str, Any]]:
        """Rank agents by aggression level"""
        aggression_scores = []

        for agent in agent_stats:
            # Calculate aggression score from combat stats
            total_attacks = sum(v for k, v in agent.combat_stats.items() if "hit." in k)
            total_steals = sum(v for k, v in agent.combat_stats.items() if "steals." in k)
            friendly_fire = agent.combat_stats.get("friendly_fire", 0)

            # Weight different aggressive actions
            aggression_score = (total_attacks * 2.0) + (total_steals * 1.5) + (friendly_fire * 0.5)

            aggression_scores.append(
                {
                    "agent_id": agent.agent_id,
                    "aggression_score": aggression_score,
                    "total_attacks": total_attacks,
                    "total_steals": total_steals,
                    "friendly_fire": friendly_fire,
                }
            )

        # Sort by aggression score
        aggression_scores.sort(key=lambda x: x["aggression_score"], reverse=True)
        return aggression_scores

    def _analyze_cooperation(self, agent_stats: List[AgentStats]) -> Dict[str, float]:
        """Analyze cooperation levels between agents"""
        cooperation_metrics = {
            "overall_cooperation_index": 0.0,
            "friendly_fire_rate": 0.0,
            "resource_sharing_index": 0.0,
        }

        total_interactions = 0
        total_friendly_fire = 0

        for agent in agent_stats:
            # Count friendly fire incidents
            friendly_fire = agent.combat_stats.get("friendly_fire", 0)
            total_friendly_fire += friendly_fire

            # Count total aggressive actions
            total_aggressive = sum(
                v for k, v in agent.combat_stats.items() if any(term in k for term in ["hit.", "steals.", "attack."])
            )
            total_interactions += total_aggressive

        # Calculate cooperation metrics
        if total_interactions > 0:
            cooperation_metrics["friendly_fire_rate"] = total_friendly_fire / total_interactions
            cooperation_metrics["overall_cooperation_index"] = 1.0 - (total_friendly_fire / total_interactions)

        return cooperation_metrics

    def _analyze_territorial_behavior(self, agent_stats: List[AgentStats]) -> Dict[str, Any]:
        """Analyze territorial behavior patterns"""
        territorial_analysis = {"territory_overlap": 0.0, "movement_clustering": {}, "resource_competition_zones": []}

        # Simplified territorial analysis
        # In a full implementation, this would analyze movement patterns and resource access
        # agent_territories = {}  # Reserved for future territorial mapping features

        for agent in agent_stats:
            # Estimate territory from movement patterns
            movement_diversity = len([k for k, v in agent.movement_patterns.items() if v > 0])
            territorial_analysis["movement_clustering"][agent.agent_id] = movement_diversity

        return territorial_analysis

    def _calculate_social_metrics(self, agent_stats: List[AgentStats]) -> Dict[str, float]:
        """Calculate social network metrics"""
        social_metrics = {"network_density": 0.0, "average_interactions_per_agent": 0.0, "isolation_index": 0.0}

        total_interactions = 0
        agents_with_interactions = 0

        for agent in agent_stats:
            agent_interactions = sum(
                v for k, v in agent.combat_stats.items() if any(term in k for term in ["hit.", "steals.", "hit_by."])
            )

            if agent_interactions > 0:
                agents_with_interactions += 1
                total_interactions += agent_interactions

        num_agents = len(agent_stats)
        if num_agents > 0:
            social_metrics["average_interactions_per_agent"] = total_interactions / num_agents
            social_metrics["network_density"] = agents_with_interactions / num_agents
            social_metrics["isolation_index"] = (num_agents - agents_with_interactions) / num_agents

        return social_metrics


class BuildingEfficiencyScorer:
    """Scores building efficiency and utilization"""

    def score_building_efficiency(self, building_stats: List[BuildingStats]) -> Dict[str, Any]:
        """
        Score efficiency of all buildings

        Args:
            building_stats: List of building statistics

        Returns:
            Building efficiency analysis

        Raises:
            ValueError: If no building statistics provided
        """
        if not building_stats:
            raise ValueError("No building statistics provided for efficiency scoring")

        analysis = {
            "individual_scores": self._score_individual_buildings(building_stats),
            "comparative_rankings": self._rank_buildings_by_efficiency(building_stats),
            "utilization_analysis": self._analyze_utilization(building_stats),
            "optimization_recommendations": self._generate_optimization_recommendations(building_stats),
        }

        return analysis

    def _score_individual_buildings(self, building_stats: List[BuildingStats]) -> Dict[str, Dict[str, float]]:
        """Score each building individually"""
        scores = {}

        for building in building_stats:
            building_id = f"building_{building.building_id}"

            # Calculate component scores
            production_score = self._calculate_production_score(building)
            efficiency_score = self._calculate_efficiency_score(building)
            utilization_score = self._calculate_utilization_score(building)
            reliability_score = self._calculate_reliability_score(building)

            # Combined score (weighted average)
            overall_score = (
                production_score * 0.3 + efficiency_score * 0.3 + utilization_score * 0.2 + reliability_score * 0.2
            )

            scores[building_id] = {
                "overall_score": overall_score,
                "production_score": production_score,
                "efficiency_score": efficiency_score,
                "utilization_score": utilization_score,
                "reliability_score": reliability_score,
            }

        return scores

    def _calculate_production_score(self, building: BuildingStats) -> float:
        """Calculate production output score"""
        total_produced = sum(flows.get("produced", 0) for flows in building.resource_flows.values())
        # Normalize to 0-1 scale (assuming max reasonable production of 100 units)
        return min(total_produced / 100.0, 1.0)

    def _calculate_efficiency_score(self, building: BuildingStats) -> float:
        """Calculate input/output efficiency score"""
        total_produced = sum(flows.get("produced", 0) for flows in building.resource_flows.values())
        total_consumed = sum(flows.get("consumed", 0) for flows in building.resource_flows.values())

        if total_consumed == 0:
            return 1.0 if total_produced > 0 else 0.0

        efficiency_ratio = total_produced / total_consumed
        # Normalize assuming ideal efficiency ratio of 2:1
        return min(efficiency_ratio / 2.0, 1.0)

    def _calculate_utilization_score(self, building: BuildingStats) -> float:
        """Calculate utilization/activity score"""
        total_conversions = building.operational_stats.get("conversions.completed", 0)
        blocked_conversions = building.operational_stats.get("conversions.blocked", 0)

        total_attempts = total_conversions + blocked_conversions
        if total_attempts == 0:
            return 0.0

        # Utilization = successful conversions / total attempts
        return total_conversions / total_attempts

    def _calculate_reliability_score(self, building: BuildingStats) -> float:
        """Calculate reliability/uptime score"""
        started = building.operational_stats.get("conversions.started", 0)
        completed = building.operational_stats.get("conversions.completed", 0)

        if started == 0:
            return 0.0

        # Reliability = completed conversions / started conversions
        return completed / started

    def _rank_buildings_by_efficiency(self, building_stats: List[BuildingStats]) -> List[Dict[str, Any]]:
        """Rank all buildings by overall efficiency"""
        individual_scores = self._score_individual_buildings(building_stats)

        rankings = []
        for building_id, scores in individual_scores.items():
            rankings.append(
                {
                    "building_id": building_id,
                    "overall_score": scores["overall_score"],
                    "rank": 0,  # Will be set after sorting
                }
            )

        # Sort by overall score
        rankings.sort(key=lambda x: x["overall_score"], reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1

        return rankings

    def _analyze_utilization(self, building_stats: List[BuildingStats]) -> Dict[str, Any]:
        """Analyze overall building utilization patterns"""
        if not building_stats:
            return {}

        utilization_scores = []
        production_scores = []

        for building in building_stats:
            util_score = self._calculate_utilization_score(building)
            prod_score = self._calculate_production_score(building)

            utilization_scores.append(util_score)
            production_scores.append(prod_score)

        analysis = {
            "average_utilization": statistics.mean(utilization_scores),
            "utilization_stdev": statistics.stdev(utilization_scores) if len(utilization_scores) > 1 else 0,
            "average_production": statistics.mean(production_scores),
            "underutilized_buildings": len([s for s in utilization_scores if s < 0.5]),
            "high_performance_buildings": len([s for s in production_scores if s > 0.8]),
        }

        return analysis

    def _generate_optimization_recommendations(self, building_stats: List[BuildingStats]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        individual_scores = self._score_individual_buildings(building_stats)

        for building_id, scores in individual_scores.items():
            if scores["utilization_score"] < 0.3:
                recommendations.append(f"{building_id}: Low utilization - check resource supply chains")

            if scores["efficiency_score"] < 0.4:
                recommendations.append(f"{building_id}: Poor efficiency - optimize input/output ratios")

            if scores["reliability_score"] < 0.6:
                recommendations.append(f"{building_id}: Low reliability - investigate conversion failures")

        # Global recommendations
        util_scores = [scores["utilization_score"] for scores in individual_scores.values()]
        if statistics.mean(util_scores) < 0.5:
            recommendations.append("GLOBAL: Overall low utilization suggests resource flow bottlenecks")

        return recommendations


class StrategicPhaseDetector:
    """Detects strategic phases and behavioral transitions in gameplay"""

    def detect_strategic_phases(self, agent_stats: List[AgentStats], episode_length: int) -> List[Dict[str, Any]]:
        """
        Detect strategic phases throughout the episode

        Args:
            agent_stats: List of agent statistics
            episode_length: Total episode length in steps

        Returns:
            List of detected strategic phases

        Raises:
            ValueError: If insufficient data for phase detection
        """
        if not agent_stats or episode_length <= 0:
            raise ValueError("Insufficient data for strategic phase detection")

        # Divide episode into phases for analysis
        num_phases = min(5, max(3, episode_length // 100))  # 3-5 phases based on episode length
        phase_length = episode_length // num_phases

        phases = []
        for phase_idx in range(num_phases):
            start_step = phase_idx * phase_length
            end_step = min((phase_idx + 1) * phase_length, episode_length)

            phase = self._analyze_phase(agent_stats, start_step, end_step, phase_idx + 1)
            phases.append(phase)

        # Detect transitions between phases
        self._detect_phase_transitions(phases)

        return phases

    def _analyze_phase(
        self, agent_stats: List[AgentStats], start_step: int, end_step: int, phase_number: int
    ) -> Dict[str, Any]:
        """Analyze a specific phase of the episode"""
        phase = {
            "phase_number": phase_number,
            "start_step": start_step,
            "end_step": end_step,
            "duration": end_step - start_step,
            "dominant_strategy": "unknown",
            "agent_activities": {},
            "resource_focus": {},
            "cooperation_level": 0.0,
            "competition_level": 0.0,
            "key_events": [],
        }

        # Analyze agent activities in this phase
        total_resource_actions = 0
        total_combat_actions = 0
        total_cooperation_events = 0

        for agent in agent_stats:
            agent_resource_activity = sum(
                flows.get("gained", 0) + flows.get("lost", 0) for flows in agent.resource_flows.values()
            )
            agent_combat_activity = sum(
                v for k, v in agent.combat_stats.items() if any(term in k for term in ["hit.", "attack.", "steals."])
            )

            total_resource_actions += agent_resource_activity
            total_combat_actions += agent_combat_activity

            # Track cooperation events
            friendly_fire = agent.combat_stats.get("friendly_fire", 0)
            total_cooperation_events += friendly_fire  # Inverse relationship

            phase["agent_activities"][agent.agent_id] = {
                "resource_activity": agent_resource_activity,
                "combat_activity": agent_combat_activity,
            }

        # Determine dominant strategy for this phase
        if total_resource_actions > total_combat_actions * 2:
            phase["dominant_strategy"] = "resource_focused"
        elif total_combat_actions > total_resource_actions:
            phase["dominant_strategy"] = "combat_focused"
        else:
            phase["dominant_strategy"] = "balanced"

        # Calculate cooperation and competition levels
        total_activity = total_resource_actions + total_combat_actions
        if total_activity > 0:
            phase["cooperation_level"] = max(0, 1.0 - (total_cooperation_events / total_activity))
            phase["competition_level"] = total_combat_actions / total_activity

        return phase

    def _detect_phase_transitions(self, phases: List[Dict[str, Any]]) -> None:
        """Detect and annotate transitions between phases"""
        for i in range(1, len(phases)):
            prev_phase = phases[i - 1]
            curr_phase = phases[i]

            # Detect strategy transitions
            if prev_phase["dominant_strategy"] != curr_phase["dominant_strategy"]:
                transition = {
                    "type": "strategy_shift",
                    "from": prev_phase["dominant_strategy"],
                    "to": curr_phase["dominant_strategy"],
                    "step": curr_phase["start_step"],
                }
                curr_phase["key_events"].append(transition)

            # Detect cooperation level changes
            coop_change = curr_phase["cooperation_level"] - prev_phase["cooperation_level"]
            if abs(coop_change) > 0.3:  # Significant change
                transition = {
                    "type": "cooperation_shift",
                    "change": "increase" if coop_change > 0 else "decrease",
                    "magnitude": abs(coop_change),
                    "step": curr_phase["start_step"],
                }
                curr_phase["key_events"].append(transition)
