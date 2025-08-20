"""
Optimal solution calculator for in-context learning converter chain problems.

This module computes theoretical optimal and realistic expected solution times
for environments with converter chains and sinks. It models both perfect information
(optimal) and discovery-based (realistic) agent behavior.

Key features:
- Supports arbitrary numbers of converters and sinks
- Models both tank-style (rotate+move) and cardinal (direct) movement
- Accounts for sink discovery and learning
- Calculates expected values over spawn distributions
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class InContextOptimalSolver:
    """
    Calculates optimal and expected solution times for converter chain problems.

    The solver models a grid environment where:
    - Converters spawn on edge positions (excluding corners)
    - Agents spawn in central positions (configurable)
    - Sinks can destroy items but are learned after first use
    """

    def __init__(self, grid_size: int = 6):
        """
        Initialize the solver with grid parameters.

        Args:
            grid_size: Size of the square grid (default: 6x6)
        """
        self.grid_size = grid_size
        self.edge_positions = self._get_edge_positions()
        self.central_positions = self._get_central_positions()

    def _get_edge_positions(self) -> List[Tuple[int, int]]:
        """Get all edge positions excluding corners."""
        positions = []
        for i in range(1, self.grid_size - 1):
            positions.extend(
                [
                    (0, i),  # Top edge
                    (self.grid_size - 1, i),  # Bottom edge
                    (i, 0),  # Left edge
                    (i, self.grid_size - 1),  # Right edge
                ]
            )
        return positions

    def _get_central_positions(self) -> List[Tuple[int, int]]:
        """Get central 4x4 area positions for agent spawning."""
        positions = []
        center_start = (self.grid_size - 4) // 2
        center_end = center_start + 4
        for r in range(center_start, center_end):
            for c in range(center_start, center_end):
                positions.append((r, c))
        return positions

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def tank_movement_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate tank-style movement cost (rotate + move).

        Assumes average case: 1 rotation + manhattan distance moves.
        In worst case, add 1 more rotation for turning corners.
        """
        distance = self.manhattan_distance(pos1, pos2)
        return 0 if distance == 0 else 1 + distance

    def calculate_average_distances(
        self, agent_pos: Tuple[int, int], converter_positions: List[Tuple[int, int]], use_tank_movement: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate average distances for movement cost estimation.

        Returns:
            Tuple of (avg_distance_from_agent, avg_inter_converter_distance)
        """
        movement_func = self.tank_movement_cost if use_tank_movement else self.manhattan_distance

        # Average distance from agent to converters
        if converter_positions:
            avg_from_agent = np.mean([movement_func(agent_pos, pos) for pos in converter_positions])
        else:
            avg_from_agent = 0

        # Average distance between converters
        if len(converter_positions) > 1:
            distances = []
            for i, pos1 in enumerate(converter_positions):
                for pos2 in converter_positions[i + 1 :]:
                    distances.append(movement_func(pos1, pos2))
            avg_inter = np.mean(distances) if distances else 0
        else:
            avg_inter = 0

        return float(avg_from_agent), float(avg_inter)

    def solve_optimal(
        self, agent_pos: Tuple[int, int], converter_positions: List[Tuple[int, int]], use_tank_movement: bool = True
    ) -> Dict:
        """
        Find optimal solution with perfect information.

        The agent knows the conversion sequence and follows the shortest path.
        Tries all possible starting converters to find the minimum cost path.

        Args:
            agent_pos: Starting position of the agent
            converter_positions: List of converter positions
            use_tank_movement: If True, use tank movement; else use cardinal

        Returns:
            Dict with movement_cost, action_cost, total_cost, and best starting converter
        """
        n = len(converter_positions)
        if n == 0:
            return {"movement_cost": 0, "action_cost": 0, "total_cost": 0}

        movement_func = self.tank_movement_cost if use_tank_movement else self.manhattan_distance
        best_total = float("inf")
        best_result = None

        # Try starting from each converter
        for start_idx in range(n):
            movement_cost = movement_func(agent_pos, converter_positions[start_idx])

            # Visit remaining converters in order (circular)
            current_pos = converter_positions[start_idx]
            for i in range(1, n):
                next_idx = (start_idx + i) % n
                movement_cost += movement_func(current_pos, converter_positions[next_idx])
                current_pos = converter_positions[next_idx]

            # Action costs: 2 timesteps per successful conversion (put + get)
            action_cost = 2 * n
            total_cost = movement_cost + action_cost

            if total_cost < best_total:
                best_total = total_cost
                best_result = {
                    "movement_cost": movement_cost,
                    "action_cost": action_cost,
                    "total_cost": total_cost,
                    "start_converter": start_idx + 1,  # 1-indexed for readability
                }

        assert best_result is not None
        return best_result

    def compute_cycle_movement_cost(
        self, converter_positions: List[Tuple[int, int]], use_tank_movement: bool = True
    ) -> float:
        """
        Compute movement cost for one complete cycle through the converter sequence.
        Complete cycle: conv[0] → conv[1] → ... → conv[n-1] → conv[0] (wrap-around)

        This represents the optimal path once the sequence is known.
        """
        n = len(converter_positions)
        if n <= 1:
            return 0.0

        movement_func = self.tank_movement_cost if use_tank_movement else self.manhattan_distance
        total = 0.0
        for i in range(n):
            a = converter_positions[i]
            b = converter_positions[(i + 1) % n]
            total += movement_func(a, b)
        return float(total)

    def solve_realistic(
        self,
        agent_pos: Tuple[int, int],
        converter_positions: List[Tuple[int, int]],
        use_tank_movement: bool = True,
        num_sinks: int = 0,
    ) -> Dict:
        """
        Calculate expected cost with discovery process using clean probability theory.

        Models a learning agent that:
        - Must discover the conversion sequence through trial and error
        - Experiences chain restarts when hitting sinks
        - Uses expected value calculations without arbitrary heuristics

        Args:
            agent_pos: Starting position of the agent
            converter_positions: List of converter positions
            use_tank_movement: If True, use tank movement; else use cardinal
            num_sinks: Number of sinks in the environment

        Returns:
            Dict with movement_cost, action_cost, failed_attempts,
            sinks_discovered, and total_cost
        """
        n = len(converter_positions)
        if n == 0:
            return {"movement_cost": 0, "action_cost": 0, "failed_attempts": 0, "sinks_discovered": 0, "total_cost": 0}

        # Calculate average distances
        avg_from_agent, avg_inter = self.calculate_average_distances(agent_pos, converter_positions, use_tank_movement)

        # Total objects in environment
        total_objects = n + num_sinks

        # Calculate expected attempts to complete the chain
        if num_sinks == 0:
            # Without sinks: finding n specific converters in sequence
            # Expected attempts = n + (n-1) + ... + 1 = n(n+1)/2
            expected_attempts = n * (n + 1) / 2
        else:
            # With sinks: at each step we need ONE SPECIFIC converter
            # among the remaining objects (converters + sinks - already found)
            expected_attempts = 0.0
            for k in range(n):
                remaining_objects = total_objects - k
                # Only ONE specific converter works at each step (not remaining_converters!)
                expected_attempts += remaining_objects

        # Calculate expected sink discoveries (learning agent avoids after first hit)
        expected_sink_hits = 0.0
        if num_sinks > 0:
            sinks_remaining = float(num_sinks)

            for k in range(n):
                objects_remaining = total_objects - k - expected_sink_hits

                if objects_remaining > 1 and sinks_remaining > 0:
                    # Probability of hitting a sink before finding the specific converter
                    p_sink_per_trial = sinks_remaining / objects_remaining
                    # Expected attempts for this converter = objects_remaining
                    # But only (objects_remaining - 1) are "wrong" attempts
                    expected_hits_this_step = p_sink_per_trial * (objects_remaining - 1)
                    expected_sink_hits += min(expected_hits_this_step, sinks_remaining)
                    sinks_remaining -= expected_hits_this_step

        # Movement costs
        movement_cost = avg_from_agent  # Initial movement
        if expected_attempts > 1:
            movement_cost += (expected_attempts - 1) * avg_inter  # Inter-object movements

        # Add restart movement costs for sink hits
        if expected_sink_hits > 0:
            restart_movement = expected_sink_hits * avg_from_agent
            movement_cost += restart_movement

        # Calculate action costs
        successful_conversions = n
        failed_attempts = expected_attempts - n - expected_sink_hits
        action_cost = (
            successful_conversions * 2  # Put + get for each converter
            + failed_attempts  # Failed attempts
            + expected_sink_hits * 3  # Sink hit + restart actions
        )

        total_cost = movement_cost + action_cost

        return {
            "movement_cost": movement_cost,
            "action_cost": action_cost,
            "failed_attempts": failed_attempts,
            "sinks_discovered": expected_sink_hits,  # For backwards compatibility
            "total_cost": total_cost,
        }

    def solve_random_policy(
        self,
        agent_pos: Tuple[int, int],
        converter_positions: List[Tuple[int, int]],
        use_tank_movement: bool = True,
        num_sinks: int = 0,
    ) -> Dict:
        """
        Random policy with no memory - agent never learns from mistakes.

        Key behaviors:
        - Picks uniformly from all objects (excluding current location)
        - Sinks cause full restart to state 0
        - No learning: can hit same sink repeatedly

        Uses Markov chain analysis to calculate expected completion time.
        """
        n = len(converter_positions)
        m = int(num_sinks)
        if n == 0:
            return {"movement_cost": 0.0, "action_cost": 0.0, "attempts": 0.0, "total_cost": 0.0}

        # Markov chain setup
        if n + m == 0:
            return {"movement_cost": 0.0, "action_cost": 0.0, "attempts": 0.0, "total_cost": 0.0}

        # Transition probabilities
        p0 = 1.0 / (n + m)  # From state 0: probability of correct converter

        if n + m > 1:
            p = 1.0 / (n + m - 1)  # From state k>0: probability of correct next
            s = m / (n + m - 1)  # From state k>0: probability of sink
        else:
            p = 0.0
            s = 0.0

        # Solve system of linear equations for expected attempts
        # E[k] = A[k] + B[k]*E[0] decomposition
        A = [0.0] * (n + 1)
        B = [0.0] * (n + 1)

        # Terminal state
        A[n] = 0.0
        B[n] = 0.0

        # Backward recursion
        if p + s > 1e-10:
            for k in range(n - 1, 0, -1):
                A[k] = (1.0 + p * A[k + 1]) / (p + s)
                B[k] = (p * B[k + 1] + s) / (p + s)
        else:
            for k in range(1, n):
                A[k] = 1e6
                B[k] = 0.0

        # Solve for E[0]
        if abs(1.0 - B[1]) > 1e-10 and p0 > 1e-10:
            E0 = (1.0 + p0 * A[1]) / (p0 * (1.0 - B[1]))
        else:
            E0 = 1e6

        attempts = E0

        # Calculate costs
        avg_from_agent, avg_inter = self.calculate_average_distances(agent_pos, converter_positions, use_tank_movement)
        movement_cost = float(avg_from_agent + max(0.0, attempts - 1.0) * avg_inter)
        action_cost = float(attempts + n)  # Failed attempts + successful conversions
        total_cost = movement_cost + action_cost
        return {
            "movement_cost": movement_cost,
            "action_cost": action_cost,
            "attempts": attempts,
            "total_cost": total_cost,
        }

    def calculate_expected_over_spawns(
        self,
        converter_positions: List[Tuple[int, int]],
        use_tank_movement: bool = True,
        spawn_positions: Optional[List[Tuple[int, int]]] = None,
        num_sinks: int = 0,
    ) -> Dict:
        """
        Calculate expected values over different agent spawn positions.

        Args:
            converter_positions: List of converter positions
            use_tank_movement: If True, use tank movement; else use cardinal
            spawn_positions: List of possible agent spawn positions
                           (defaults to central 4x4 area)
            num_sinks: Number of sinks in the environment

        Returns:
            Dict with statistics (mean, std, min, max) for optimal and realistic solutions
        """
        if spawn_positions is None:
            spawn_positions = self.central_positions

        optimal_results = []
        realistic_results = []
        sinks_discovered_results = []

        for agent_pos in spawn_positions:
            optimal = self.solve_optimal(agent_pos, converter_positions, use_tank_movement)
            realistic = self.solve_realistic(agent_pos, converter_positions, use_tank_movement, num_sinks)

            optimal_results.append(optimal["total_cost"])
            realistic_results.append(realistic["total_cost"])
            sinks_discovered_results.append(realistic["sinks_discovered"])

        return {
            "optimal_mean": np.mean(optimal_results),
            "optimal_std": np.std(optimal_results),
            "optimal_min": np.min(optimal_results),
            "optimal_max": np.max(optimal_results),
            "realistic_mean": np.mean(realistic_results),
            "realistic_std": np.std(realistic_results),
            "realistic_min": np.min(realistic_results),
            "realistic_max": np.max(realistic_results),
            "sinks_discovered_mean": np.mean(sinks_discovered_results),
            "movement_type": "tank" if use_tank_movement else "cardinal",
        }

    def calculate_random_over_spawns(
        self,
        converter_positions: List[Tuple[int, int]],
        use_tank_movement: bool = True,
        spawn_positions: Optional[List[Tuple[int, int]]] = None,
        num_sinks: int = 0,
    ) -> Dict:
        """
        Expected cost for fully random policy over spawn distribution.
        """
        if spawn_positions is None:
            spawn_positions = self.central_positions

        totals = []
        for agent_pos in spawn_positions:
            r = self.solve_random_policy(agent_pos, converter_positions, use_tank_movement, num_sinks)
            totals.append(r["total_cost"])
        return {
            "random_mean": float(np.mean(totals)) if totals else 0.0,
            "random_std": float(np.std(totals)) if totals else 0.0,
            "movement_type": "tank" if use_tank_movement else "cardinal",
        }

    def expected_reward_random(
        self,
        agent_pos: Tuple[int, int],
        converter_positions: List[Tuple[int, int]],
        use_tank_movement: bool = True,
        num_sinks: int = 0,
        total_steps: int = 256,
    ) -> Dict:
        """
        Calculate expected reward under random policy within step budget.
        Unlike the smart policy, random doesn't improve after first completion.
        """
        n = len(converter_positions)
        if n == 0:
            return {"completions": 0.0, "reward": 0.0}

        # Get expected cost for one completion under random policy
        random_result = self.solve_random_policy(agent_pos, converter_positions, use_tank_movement, num_sinks)
        cost_per_completion = random_result["total_cost"]

        if cost_per_completion <= 0 or cost_per_completion >= 1e6:
            return {"completions": 0.0, "reward": 0.0}

        # Simple model: how many completions fit in budget?
        expected_completions = total_steps / cost_per_completion

        return {
            "completions": expected_completions,
            "reward": expected_completions,  # 1 reward per completion
        }

    def expected_reward_random_over_spawns(
        self,
        converter_positions: List[Tuple[int, int]],
        use_tank_movement: bool = True,
        spawn_positions: Optional[List[Tuple[int, int]]] = None,
        num_sinks: int = 0,
        total_steps: int = 256,
    ) -> Dict:
        """
        Expected reward under random policy averaged over spawn positions.
        """
        if spawn_positions is None:
            spawn_positions = self.central_positions

        rewards = []
        for agent_pos in spawn_positions:
            result = self.expected_reward_random(
                agent_pos, converter_positions, use_tank_movement, num_sinks, total_steps
            )
            rewards.append(result["reward"])

        return {
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "movement_type": "tank" if use_tank_movement else "cardinal",
        }

    def expected_reward(
        self,
        agent_pos: Tuple[int, int],
        converter_positions: List[Tuple[int, int]],
        steps_available: int,
        use_tank_movement: bool = True,
        num_sinks: int = 0,
        reward_per_completion: float = 1.0,
    ) -> Dict:
        """
        Expected reward with finite budget of steps, assuming:
        - First completion follows discovery model (realistic)
        - Subsequent completions repeat optimally as a steady-state cycle
        - Next cycle starts at the end of the chain (wrap-around included)
        """
        n = len(converter_positions)
        if steps_available <= 0 or n == 0:
            return {"expected_completions": 0.0, "expected_reward": 0.0, "leftover_steps": float(steps_available)}

        # First completion (discovery)
        first = self.solve_realistic(agent_pos, converter_positions, use_tank_movement, num_sinks)
        first_cost = first["total_cost"]

        # Per-cycle optimal steady-state cost (movement + actions)
        cycle_movement = self.compute_cycle_movement_cost(converter_positions, use_tank_movement)
        cycle_actions = 2 * n
        cycle_cost = cycle_movement + cycle_actions

        # Compute expected completions within budget
        if first_cost > steps_available:
            completions = 0
            leftover = float(steps_available)
        else:
            remaining = steps_available - first_cost
            extra = int(remaining // cycle_cost) if cycle_cost > 0 else 0
            completions = 1 + extra
            leftover = float(remaining - extra * cycle_cost)

        return {
            "expected_completions": float(completions),
            "expected_reward": float(completions) * reward_per_completion,
            "leftover_steps": leftover,
            "first_completion_cost": float(first_cost),
            "cycle_cost": float(cycle_cost),
        }

    def expected_reward_over_spawns(
        self,
        converter_positions: List[Tuple[int, int]],
        steps_available: int,
        use_tank_movement: bool = True,
        spawn_positions: Optional[List[Tuple[int, int]]] = None,
        num_sinks: int = 0,
        reward_per_completion: float = 1.0,
    ) -> Dict:
        """
        Average expected completions/reward over a set of spawn positions.
        """
        if spawn_positions is None:
            spawn_positions = self.central_positions

        completions = []
        rewards = []

        for agent_pos in spawn_positions:
            er = self.expected_reward(
                agent_pos,
                converter_positions,
                steps_available,
                use_tank_movement=use_tank_movement,
                num_sinks=num_sinks,
                reward_per_completion=reward_per_completion,
            )
            completions.append(er["expected_completions"])
            rewards.append(er["expected_reward"])

        return {
            "expected_completions_mean": float(np.mean(completions)) if completions else 0.0,
            "expected_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "movement_type": "tank" if use_tank_movement else "cardinal",
        }


def analyze_configuration(
    num_converters: int,
    num_sinks: int = 0,
    grid_size: int = 6,
    num_samples: int = 100,
) -> Dict:
    """
    Analyze a configuration with given converters and sinks.

    Samples random converter placements and calculates expected performance
    for both movement types.

    Args:
        num_converters: Number of converters in the chain
        num_sinks: Number of sinks that can destroy items
        grid_size: Size of the grid
        num_samples: Number of random samples to average over

    Returns:
        Dict with results for both tank and cardinal movement
    """
    solver = InContextOptimalSolver(grid_size)

    # Can't have more objects than edge positions
    max_objects = len(solver.edge_positions)
    if num_converters + num_sinks > max_objects:
        raise ValueError(f"Too many objects ({num_converters + num_sinks}) for {max_objects} edge positions")

    all_results = {"tank": [], "cardinal": []}

    for _ in range(num_samples):
        # Randomly place converters
        indices = np.random.choice(len(solver.edge_positions), num_converters, replace=False)
        converter_positions = [solver.edge_positions[i] for i in indices]

        # Calculate for both movement types
        for use_tank, key in [(True, "tank"), (False, "cardinal")]:
            result = solver.calculate_expected_over_spawns(
                converter_positions, use_tank_movement=use_tank, num_sinks=num_sinks
            )
            all_results[key].append(result)

    # Aggregate results
    final_results = {}
    for movement_type in ["tank", "cardinal"]:
        results = all_results[movement_type]
        final_results[movement_type] = {
            "optimal_mean": np.mean([r["optimal_mean"] for r in results]),
            "optimal_std": np.mean([r["optimal_std"] for r in results]),
            "realistic_mean": np.mean([r["realistic_mean"] for r in results]),
            "realistic_std": np.mean([r["realistic_std"] for r in results]),
            "sinks_discovered_mean": np.mean([r["sinks_discovered_mean"] for r in results]),
        }

    return final_results


def print_analysis(
    num_converters: int,
    num_sinks: int = 0,
    grid_size: int = 6,
    steps_available: Optional[int] = None,
    samples: int = 100,
    reward_per_completion: float = 1.0,
):
    """
    Print a formatted analysis for a given configuration.

    Args:
        num_converters: Number of converters in the chain
        num_sinks: Number of sinks that can destroy items
        grid_size: Size of the grid
    """
    print(f"\n{'=' * 60}")
    print(f"Configuration: {num_converters} converters, {num_sinks} sinks on {grid_size}x{grid_size} grid")
    print(f"{'=' * 60}")

    results = analyze_configuration(num_converters, num_sinks, grid_size, num_samples=samples)

    for movement_type in ["tank", "cardinal"]:
        r = results[movement_type]
        print(f"\n{movement_type.capitalize()} Movement:")
        print(f"  Optimal:   {r['optimal_mean']:6.1f} ± {r['optimal_std']:.1f} steps")
        print(f"  Realistic: {r['realistic_mean']:6.1f} ± {r['realistic_std']:.1f} steps")
        if num_sinks > 0:
            print(f"    Sinks discovered: {r['sinks_discovered_mean']:.2f} on average")
        # Random policy baseline
        solver = InContextOptimalSolver(grid_size)
        # sample one placement for readability
        idx = np.random.choice(len(solver.edge_positions), num_converters, replace=False)
        conv_pos = [solver.edge_positions[i] for i in idx]
        rand_stats = solver.calculate_random_over_spawns(
            conv_pos, use_tank_movement=(movement_type == "tank"), num_sinks=num_sinks
        )
        print(f"  Random:    {rand_stats['random_mean']:6.1f} ± {rand_stats['random_std']:.1f} steps (baseline)")

    # Show impact of sinks (using same converter positions for fair comparison)
    if num_sinks > 0:
        solver = InContextOptimalSolver(grid_size)

        # Calculate impact using the same converter positions
        impact_tank = []
        impact_cardinal = []

        for _ in range(min(samples, 20)):  # Use fewer samples for impact calculation
            indices = np.random.choice(len(solver.edge_positions), num_converters, replace=False)
            converter_positions = [solver.edge_positions[i] for i in indices]

            # Calculate with and without sinks using SAME positions
            tank_with = solver.calculate_expected_over_spawns(
                converter_positions, use_tank_movement=True, num_sinks=num_sinks
            )
            tank_without = solver.calculate_expected_over_spawns(
                converter_positions, use_tank_movement=True, num_sinks=0
            )
            cardinal_with = solver.calculate_expected_over_spawns(
                converter_positions, use_tank_movement=False, num_sinks=num_sinks
            )
            cardinal_without = solver.calculate_expected_over_spawns(
                converter_positions, use_tank_movement=False, num_sinks=0
            )

            # Calculate percentage increase
            if tank_without["realistic_mean"] > 0:
                impact_tank.append(
                    (tank_with["realistic_mean"] - tank_without["realistic_mean"])
                    / tank_without["realistic_mean"]
                    * 100
                )
            if cardinal_without["realistic_mean"] > 0:
                impact_cardinal.append(
                    (cardinal_with["realistic_mean"] - cardinal_without["realistic_mean"])
                    / cardinal_without["realistic_mean"]
                    * 100
                )

        print(f"\nImpact of {num_sinks} sink(s):")
        if impact_tank:
            print(f"  Tank: +{np.mean(impact_tank):.1f}% time increase")
        if impact_cardinal:
            print(f"  Cardinal: +{np.mean(impact_cardinal):.1f}% time increase")

    # Expected reward section (optional)
    if steps_available is not None and steps_available > 0:
        solver = InContextOptimalSolver(grid_size)
        # sample placements similarly to analyze_configuration
        max_objects = len(solver.edge_positions)
        if num_converters + num_sinks <= max_objects:
            tank_rewards = []
            card_rewards = []
            for _ in range(samples):
                indices = np.random.choice(len(solver.edge_positions), num_converters, replace=False)
                converter_positions = [solver.edge_positions[i] for i in indices]
                tank = solver.expected_reward_over_spawns(
                    converter_positions,
                    steps_available=steps_available,
                    use_tank_movement=True,
                    num_sinks=num_sinks,
                    reward_per_completion=reward_per_completion,
                )
                card = solver.expected_reward_over_spawns(
                    converter_positions,
                    steps_available=steps_available,
                    use_tank_movement=False,
                    num_sinks=num_sinks,
                    reward_per_completion=reward_per_completion,
                )
                tank_rewards.append(tank)
                card_rewards.append(card)

            def _mean(xs: List[float]) -> float:
                return float(np.mean(xs)) if xs else 0.0

            # Also calculate random policy rewards
            tank_random_rewards = []
            card_random_rewards = []
            for _ in range(min(10, samples)):  # Use fewer samples for random since it's expensive
                indices = np.random.choice(len(solver.edge_positions), num_converters, replace=False)
                converter_positions = [solver.edge_positions[i] for i in indices]
                tank_rand = solver.expected_reward_random_over_spawns(
                    converter_positions,
                    total_steps=steps_available,
                    use_tank_movement=True,
                    num_sinks=num_sinks,
                )
                card_rand = solver.expected_reward_random_over_spawns(
                    converter_positions,
                    total_steps=steps_available,
                    use_tank_movement=False,
                    num_sinks=num_sinks,
                )
                tank_random_rewards.append(tank_rand["reward_mean"])
                card_random_rewards.append(card_rand["reward_mean"])

            print(f"\nExpected reward with budget: {steps_available} steps")
            print(
                f"  Tank     - Smart: {_mean([r['expected_completions_mean'] for r in tank_rewards]):.2f} completions, "
                f"Random: {_mean(tank_random_rewards):.2f} completions"
            )
            print(
                f"  Cardinal - Smart: {_mean([r['expected_completions_mean'] for r in card_rewards]):.2f} completions, "
                f"Random: {_mean(card_random_rewards):.2f} completions"
            )


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in value.split(",")]


def main():
    """
    Command-line interface for the optimal solver.

    Examples:
        # Run default demo
        python incontext_optimal_solver.py

        # Analyze specific configuration
        python incontext_optimal_solver.py --converters 3 --sinks 2

        # Analyze multiple configurations
        python incontext_optimal_solver.py --converters 2,3,4 --sinks 0,1,2

        # Output as JSON
        python incontext_optimal_solver.py --converters 3 --sinks 1 --json

        # Custom grid size and samples
        python incontext_optimal_solver.py --converters 4 --sinks 2 --grid-size 8 --samples 500
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyze optimal solutions for in-context learning converter chains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--converters", "-c", type=parse_int_list, help="Number of converters (comma-separated for multiple)"
    )

    parser.add_argument(
        "--sinks",
        "-s",
        type=parse_int_list,
        default=[0],
        help="Number of sinks (comma-separated for multiple, default: 0)",
    )

    parser.add_argument("--grid-size", "-g", type=int, default=6, help="Grid size (default: 6)")

    parser.add_argument(
        "--samples", "-n", type=int, default=100, help="Number of random samples for averaging (default: 100)"
    )

    parser.add_argument("--json", "-j", action="store_true", help="Output results as JSON")

    parser.add_argument("--compare-movement", action="store_true", help="Show detailed movement comparison")

    parser.add_argument("--demo", action="store_true", help="Run demonstration with default configurations")

    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Compute expected reward under this step budget (default: 256)",
    )

    args = parser.parse_args()

    # args.steps already defaults to 256 to report expected reward by default

    # If no converters specified, run demo
    if args.converters is None or args.demo:
        print("In-Context Learning Optimal Solution Analysis")
        print("=" * 60)

        # Default demo configurations
        configurations = [
            (2, 0),  # 2 converters, no sinks
            (2, 1),  # 2 converters, 1 sink
            (3, 0),  # 3 converters, no sinks
            (3, 2),  # 3 converters, 2 sinks
            (4, 2),  # 4 converters, 2 sinks
            (5, 3),  # 5 converters, 3 sinks
        ]

        for n_conv, n_sinks in configurations:
            print_analysis(n_conv, n_sinks, args.grid_size, steps_available=args.steps, samples=args.samples)

        print("\n" + "=" * 60)
        print("Analysis complete!")
        return

    # Process command-line arguments
    # Ensure lists are same length or broadcast single values
    if len(args.sinks) == 1 and len(args.converters) > 1:
        args.sinks = args.sinks * len(args.converters)
    elif len(args.converters) == 1 and len(args.sinks) > 1:
        args.converters = args.converters * len(args.sinks)
    elif len(args.converters) != len(args.sinks):
        parser.error("Converters and sinks lists must have same length or one must be single value")

    results = {}

    for n_conv, n_sinks in zip(args.converters, args.sinks, strict=False):
        config_key = f"{n_conv}_converters_{n_sinks}_sinks"

        try:
            result = analyze_configuration(n_conv, n_sinks, args.grid_size, args.samples)
            results[config_key] = result

            if not args.json:
                print_analysis(
                    n_conv,
                    n_sinks,
                    args.grid_size,
                    steps_available=args.steps,
                    samples=args.samples,
                )

                if args.compare_movement:
                    print("\nMovement Comparison:")
                    tank = result["tank"]
                    cardinal = result["cardinal"]

                    opt_savings = (tank["optimal_mean"] - cardinal["optimal_mean"]) / tank["optimal_mean"] * 100
                    real_savings = (tank["realistic_mean"] - cardinal["realistic_mean"]) / tank["realistic_mean"] * 100

                    print(f"  Cardinal vs Tank (Optimal):   -{opt_savings:.1f}% time")
                    print(f"  Cardinal vs Tank (Realistic): -{real_savings:.1f}% time")

        except ValueError as e:
            if args.json:
                results[config_key] = {"error": str(e)}
            else:
                print(f"\nError for {n_conv} converters, {n_sinks} sinks: {e}")

    if args.json:
        # Add metadata
        results["_metadata"] = {
            "grid_size": args.grid_size,
            "samples": args.samples,
            "configurations": len(results) - 1,
        }
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
