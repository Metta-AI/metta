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

    def calculate_sink_discovery(self, n_converters: int, n_sinks: int) -> float:
        """
        Calculate expected number of sinks discovered during chain completion.

        Uses probability theory to estimate how many sinks will be found
        while searching for converters in sequence.
        """
        if n_sinks == 0 or n_converters == 0:
            return 0

        total_objects = n_converters + n_sinks
        expected_discovered = 0
        unknown_sinks = n_sinks
        unknown_objects = total_objects

        for step in range(n_converters):
            remaining_converters = n_converters - step
            if unknown_sinks > 0 and unknown_objects > 0:
                prob_sink = unknown_sinks / unknown_objects
                expected_attempts = unknown_objects / remaining_converters
                sinks_in_step = prob_sink * (expected_attempts - 1)
                expected_discovered += sinks_in_step
                unknown_sinks -= sinks_in_step
                unknown_objects = remaining_converters - 1 + unknown_sinks

        return min(expected_discovered, n_sinks)  # Can't discover more than exist

    def solve_realistic(
        self,
        agent_pos: Tuple[int, int],
        converter_positions: List[Tuple[int, int]],
        use_tank_movement: bool = True,
        num_sinks: int = 0,
    ) -> Dict:
        """
        Calculate expected cost with discovery process.

        Models a learning agent that:
        - Must discover the conversion sequence through trial and error
        - Learns and avoids sinks after losing an item to them
        - Searches efficiently among remaining unknown objects

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

        # Calculate expected sink discoveries
        expected_sinks_discovered = self.calculate_sink_discovery(n, num_sinks)

        # Initialize cost accumulators
        movement_cost = 0
        failed_attempts = 0

        # Model the discovery process
        total_objects = n + num_sinks
        discovered_sinks = 0

        # Phase 1: Find first converter
        if num_sinks == 0:
            expected_first_attempts = (n + 1) / 2
        else:
            expected_first_attempts = total_objects / n
            # Assume ~30% of sinks discovered in first phase
            discovered_in_phase1 = min(expected_sinks_discovered * 0.3, num_sinks * 0.3)
            discovered_sinks = discovered_in_phase1

        movement_cost += avg_from_agent * expected_first_attempts
        failed_attempts += expected_first_attempts - 1

        # Phase 2: Find remaining converters in sequence
        for step in range(1, n):
            remaining_converters = n - step
            remaining_unknown_sinks = max(0, num_sinks - discovered_sinks)
            effective_unknown_objects = remaining_converters + remaining_unknown_sinks

            if remaining_unknown_sinks == 0:
                # All sinks discovered, search only converters
                expected_attempts = (remaining_converters + 1) / 2
            else:
                # Still have unknown sinks in the mix
                expected_attempts = effective_unknown_objects / remaining_converters
                # Discover more sinks as we go (distribute remaining 70%)
                if n > 1:
                    discovered_in_step = min(remaining_unknown_sinks * 0.2, expected_sinks_discovered * (0.7 / (n - 1)))
                    discovered_sinks += discovered_in_step

            movement_cost += avg_inter * expected_attempts
            failed_attempts += expected_attempts - 1

        # Cost of restarting after each sink discovery
        sink_restart_cost = expected_sinks_discovered * (
            avg_from_agent + 2  # Move to first converter + get first item
        )
        movement_cost += sink_restart_cost

        # Calculate total action costs
        successful_conversions = n
        action_cost = (
            successful_conversions * 2  # Put + get for each conversion
            + failed_attempts * 1  # Failed put attempts
            + expected_sinks_discovered * 1  # Putting items into sinks
            + expected_sinks_discovered * 2  # Restarting chain after sink loss
        )

        total_cost = movement_cost + action_cost

        return {
            "movement_cost": movement_cost,
            "action_cost": action_cost,
            "failed_attempts": failed_attempts,
            "sinks_discovered": expected_sinks_discovered,
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


def print_analysis(num_converters: int, num_sinks: int = 0, grid_size: int = 6):
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

    results = analyze_configuration(num_converters, num_sinks, grid_size)

    for movement_type in ["tank", "cardinal"]:
        r = results[movement_type]
        print(f"\n{movement_type.capitalize()} Movement:")
        print(f"  Optimal:   {r['optimal_mean']:6.1f} ± {r['optimal_std']:.1f} steps")
        print(f"  Realistic: {r['realistic_mean']:6.1f} ± {r['realistic_std']:.1f} steps")
        if num_sinks > 0:
            print(f"    Sinks discovered: {r['sinks_discovered_mean']:.2f} on average")

    # Show impact of sinks
    if num_sinks > 0:
        # Get baseline without sinks
        baseline = analyze_configuration(num_converters, 0, grid_size)
        print(f"\nImpact of {num_sinks} sink(s):")
        for movement_type in ["tank", "cardinal"]:
            increase = (
                (results[movement_type]["realistic_mean"] - baseline[movement_type]["realistic_mean"])
                / baseline[movement_type]["realistic_mean"]
                * 100
            )
            print(f"  {movement_type.capitalize()}: +{increase:.1f}% time increase")


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

    args = parser.parse_args()

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
            print_analysis(n_conv, n_sinks, args.grid_size)

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
                print_analysis(n_conv, n_sinks, args.grid_size)

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
