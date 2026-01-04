#!/usr/bin/env python3
"""Debug harness for diagnosing CoGsGuard agent behaviors.

This module provides tools to:
- Step through simulation manually
- Inspect agent state, observations, and internal policy state
- Compare actual simulation state vs internal agent tracking
- Identify coordinate mismatches, stuck agents, and other issues

Usage:
    from cogames.policy.scripted_agent.cogsguard.debug_agent import DebugHarness

    harness = DebugHarness.from_recipe("recipes.experiment.cogsguard")
    harness.step(10)  # Run 10 steps
    harness.print_agent_summary()
    harness.diagnose_stuck_agents()
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

# Ensure we can import from project root
sys.path.insert(0, ".")


@dataclass
class AgentDebugInfo:
    """Debug information for a single agent."""

    agent_id: int
    # Internal state (from policy)
    internal_pos: tuple[int, int] | None = None
    role: str | None = None
    phase: str | None = None
    cargo: int = 0
    cargo_capacity: int = 4
    has_gear: bool = False
    current_vibe: str | None = None
    last_action: str | None = None
    # Actual state (from simulation)
    actual_inventory: dict[str, int] = field(default_factory=dict)
    # Navigation
    target_position: tuple[int, int] | None = None
    assembler_pos: tuple[int, int] | None = None
    # History
    position_history: list[tuple[int, int]] = field(default_factory=list)
    stuck_count: int = 0


class DebugHarness:
    """Debug harness for CoGsGuard agent diagnostics."""

    def __init__(
        self,
        env_cfg: Any,
        policy: Any,
        agent_policies: list[Any],
        rollout: Any,
    ):
        self.env_cfg = env_cfg
        self.policy = policy
        self.agent_policies = agent_policies
        self.rollout = rollout
        self.sim = rollout._sim
        self.agents = rollout._agents
        self.num_agents = len(self.agents)
        self.step_count = 0

        # Track debug info per agent
        self.agent_info: dict[int, AgentDebugInfo] = {i: AgentDebugInfo(agent_id=i) for i in range(self.num_agents)}

        # Callback hooks
        self.on_step_callbacks: list[Callable[["DebugHarness"], None]] = []

    @classmethod
    def from_recipe(
        cls,
        recipe_module: str = "recipes.experiment.cogsguard",
        num_agents: int = 10,
        max_steps: int = 1000,
        seed: int = 42,
        policy_uri: str = "metta://policy/cogsguard?scrambler=1&miner=4",
    ) -> "DebugHarness":
        """Create debug harness from a recipe module.

        Args:
            recipe_module: Module path to recipe (e.g., "recipes.experiment.cogsguard")
            num_agents: Number of agents
            max_steps: Maximum simulation steps
            seed: Random seed
            policy_uri: Policy URI with role counts (e.g., "metta://policy/cogsguard?miner=4&scrambler=1")
        """
        import importlib

        from mettagrid.policy.loader import initialize_or_load_policy
        from mettagrid.policy.policy_env_interface import PolicyEnvInterface
        from mettagrid.simulator.rollout import Rollout
        from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

        # Import recipe and get make_env
        recipe = importlib.import_module(recipe_module)
        make_env = recipe.make_env

        # Create environment config
        env_cfg = make_env(num_agents=num_agents, max_steps=max_steps)
        policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

        # Load the cogsguard policy with role counts from URI
        policy_spec = policy_spec_from_uri(policy_uri)
        multi_policy = initialize_or_load_policy(policy_env_info, policy_spec)

        # Create per-agent policies
        agent_policies = [multi_policy.agent_policy(i) for i in range(num_agents)]

        # Create rollout
        rollout = Rollout(
            config=env_cfg,
            policies=agent_policies,
            render_mode=None,
            seed=seed,
        )

        return cls(env_cfg, multi_policy, agent_policies, rollout)

    def get_agent_state(self, agent_idx: int) -> Any | None:
        """Get internal policy state for an agent."""
        agent_policy = self.agent_policies[agent_idx]
        if hasattr(agent_policy, "_state"):
            return agent_policy._state
        return None

    def step(self, n: int = 1) -> None:
        """Execute n simulation steps."""
        for _ in range(n):
            self.rollout.step()
            self.step_count += 1
            self._update_debug_info()
            for callback in self.on_step_callbacks:
                callback(self)

    def _update_debug_info(self) -> None:
        """Update debug info for all agents after a step."""
        for i in range(self.num_agents):
            info = self.agent_info[i]
            state = self.get_agent_state(i)

            if state:
                pos = (state.row, state.col)
                info.internal_pos = pos
                info.role = state.role.value if hasattr(state.role, "value") else str(state.role)
                info.phase = state.phase.value if hasattr(state.phase, "value") else str(state.phase)
                info.cargo = state.total_cargo
                info.cargo_capacity = state.cargo_capacity
                info.has_gear = state.has_gear()
                info.current_vibe = state.current_vibe
                info.last_action = state.last_action.name if state.last_action else None
                info.target_position = state.target_position
                info.assembler_pos = state.stations.get("assembler")

                # Track position history
                info.position_history.append(pos)
                if len(info.position_history) > 30:
                    info.position_history.pop(0)

                # Detect stuck
                if len(info.position_history) >= 10:
                    recent = info.position_history[-10:]
                    if all(p == recent[0] for p in recent):
                        info.stuck_count += 1
                    else:
                        info.stuck_count = 0

            # Get actual inventory from simulation
            info.actual_inventory = dict(self.agents[i].inventory)

    def get_actual_agent_position(self, agent_idx: int) -> tuple[int, int] | None:
        """Get actual agent position from simulation.

        Returns (row, col) in simulation coordinates.
        """
        agent = self.agents[agent_idx]
        if hasattr(agent, "location"):
            loc = agent.location
            # location is typically (col, row) so convert to (row, col)
            return (loc[1], loc[0])
        elif hasattr(agent, "r") and hasattr(agent, "c"):
            return (agent.r, agent.c)
        return None

    def verify_position_tracking(self, verbose: bool = True) -> dict[int, dict]:
        """Verify that internal position tracking matches simulation.

        Since internal positions are RELATIVE to starting position,
        we track deltas from the starting position and compare movement.

        Returns dict of agent_id -> {issues found}
        """
        results: dict[int, dict] = {}

        if verbose:
            print(f"\n=== Position Tracking Verification (step {self.step_count}) ===")

        for i in range(self.num_agents):
            state = self.get_agent_state(i)
            if not state:
                continue

            # Get internal position (relative coords, centered at ~100)
            internal_pos = (state.row, state.col)

            # Get actual simulation position
            actual_pos = self.get_actual_agent_position(i)

            # Get what action was intended vs executed
            intended = state.last_action.name if state.last_action else "none"
            executed = state.last_action_executed if hasattr(state, "last_action_executed") else "unknown"

            agent_result = {
                "internal_pos": internal_pos,
                "actual_pos": actual_pos,
                "intended_action": intended,
                "executed_action": executed,
                "mismatch": intended != executed if executed != "unknown" else None,
            }

            # Check if internal position is within expected grid bounds
            # Internal coords centered at ~100, so valid range is roughly 0-200
            if internal_pos[0] < 0 or internal_pos[0] >= 200 or internal_pos[1] < 0 or internal_pos[1] >= 200:
                agent_result["out_of_bounds"] = True

            results[i] = agent_result

            if verbose:
                mismatch_str = ""
                if agent_result.get("mismatch"):
                    mismatch_str = f" [MISMATCH: intended={intended}, executed={executed}]"
                print(
                    f"Agent {i}: internal={internal_pos}, actual_sim={actual_pos}, last_action={intended}{mismatch_str}"
                )

        return results

    def track_position_drift(self, num_steps: int = 50, verbose: bool = True) -> None:
        """Run simulation and track if positions drift from expected.

        This helps identify if internal position tracking diverges from reality.
        """
        if verbose:
            print(f"\n=== Position Drift Tracking ({num_steps} steps) ===")

        # Record starting positions for each agent
        start_internal: dict[int, tuple[int, int]] = {}
        start_actual: dict[int, tuple[int, int] | None] = {}

        for i in range(self.num_agents):
            state = self.get_agent_state(i)
            if state:
                start_internal[i] = (state.row, state.col)
            start_actual[i] = self.get_actual_agent_position(i)

        # Track movement counts
        move_counts: dict[int, dict[str, int]] = {i: {"intended": 0, "executed": 0} for i in range(self.num_agents)}
        mismatches: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(self.num_agents)}

        # Step through simulation
        for _step in range(num_steps):
            self.step(1)

            for i in range(self.num_agents):
                state = self.get_agent_state(i)
                if not state:
                    continue

                intended = state.last_action.name if state.last_action else "noop"
                executed = getattr(state, "last_action_executed", None) or "unknown"

                # Count moves
                if intended.startswith("move_"):
                    move_counts[i]["intended"] += 1
                if executed and executed.startswith("move_"):
                    move_counts[i]["executed"] += 1

                # Track mismatches
                if intended != executed and executed != "unknown":
                    mismatches[i].append((self.step_count, intended, executed))

        # Report results
        if verbose:
            print("\n--- Results ---")
            for i in range(self.num_agents):
                state = self.get_agent_state(i)
                if not state:
                    continue

                end_internal = (state.row, state.col)

                # Calculate movement delta
                internal_delta = (
                    end_internal[0] - start_internal[i][0],
                    end_internal[1] - start_internal[i][1],
                )

                # Note: actual_delta could be computed here if needed for debugging
                # but we focus on internal tracking consistency

                mismatch_count = len(mismatches[i])
                moves = move_counts[i]

                print(
                    f"Agent {i}: internal_delta={internal_delta}, "
                    f"moves(intended={moves['intended']}, executed={moves['executed']}), "
                    f"action_mismatches={mismatch_count}"
                )

                if mismatch_count > 0 and verbose:
                    print(f"  First 5 mismatches: {mismatches[i][:5]}")

    def get_grid_objects(self) -> dict[int, dict]:
        """Get all grid objects from simulation."""
        return self.sim.grid_objects()

    def get_objects_by_type(self, type_name: str) -> list[dict]:
        """Get all objects of a specific type."""
        return [obj for obj in self.get_grid_objects().values() if obj.get("type_name") == type_name]

    def get_object_types(self) -> dict[str, int]:
        """Get count of each object type in simulation."""
        types: dict[str, int] = {}
        for obj in self.get_grid_objects().values():
            t = obj.get("type_name", "unknown")
            types[t] = types.get(t, 0) + 1
        return dict(sorted(types.items()))

    def find_assemblers(self) -> list[tuple[int, int]]:
        """Find actual assembler positions in simulation."""
        positions = []
        for obj in self.get_grid_objects().values():
            type_name = obj.get("type_name", "")
            if "assembler" in type_name.lower() or "nexus" in type_name.lower():
                # Use location tuple or r,c
                loc = obj.get("location")
                if loc:
                    positions.append((loc[1], loc[0]))  # Convert (col, row) to (row, col)
                else:
                    positions.append((obj.get("r", 0), obj.get("c", 0)))
        return positions

    def print_agent_summary(self, agent_ids: list[int] | None = None) -> None:
        """Print summary of agent states."""
        if agent_ids is None:
            agent_ids = list(range(self.num_agents))

        print(f"\n=== Agent Summary (step {self.step_count}) ===")
        for i in agent_ids:
            info = self.agent_info[i]
            gear_str = "GEAR" if info.has_gear else "NO_GEAR"
            stuck_str = f" [STUCK x{info.stuck_count}]" if info.stuck_count > 0 else ""
            print(
                f"Agent {i}: pos={info.internal_pos} role={info.role} phase={info.phase} "
                f"cargo={info.cargo}/{info.cargo_capacity} {gear_str}{stuck_str}"
            )

    def print_simulation_info(self) -> None:
        """Print simulation state info."""
        print(f"\n=== Simulation Info (step {self.step_count}) ===")
        print(f"Object types: {self.get_object_types()}")
        print(f"Assemblers at: {self.find_assemblers()}")

    def diagnose_stuck_agents(self, threshold: int = 5) -> list[int]:
        """Find and diagnose stuck agents.

        Args:
            threshold: Number of consecutive stuck steps to consider an agent stuck

        Returns:
            List of stuck agent IDs
        """
        stuck = []
        for i in range(self.num_agents):
            info = self.agent_info[i]
            if info.stuck_count >= threshold:
                stuck.append(i)
                self._diagnose_agent(i)
        return stuck

    def _diagnose_agent(self, agent_id: int) -> None:
        """Print detailed diagnosis for an agent."""
        info = self.agent_info[agent_id]
        state = self.get_agent_state(agent_id)

        print(f"\n=== STUCK AGENT {agent_id} (step {self.step_count}) ===")
        print(f"Internal Position: {info.internal_pos}")
        print(f"Role: {info.role}, Phase: {info.phase}")
        print(f"Cargo: {info.cargo}/{info.cargo_capacity}, Gear: {info.has_gear}")
        print(f"Current vibe: {info.current_vibe}")
        print(f"Last action: {info.last_action}")
        print(f"Actual inventory: {info.actual_inventory}")

        if state:
            print(f"\nStored assembler location: {info.assembler_pos}")

            # Find actual assemblers
            actual_assemblers = self.find_assemblers()
            print(f"Actual assemblers in sim: {actual_assemblers}")

            # Check nearby structures from state
            if info.internal_pos and hasattr(state, "structures"):
                print("\nNearby structures (within 10 cells):")
                for pos, struct in state.structures.items():
                    dist = abs(pos[0] - info.internal_pos[0]) + abs(pos[1] - info.internal_pos[1])
                    if dist <= 10:
                        stype = struct.structure_type
                        struct_type = stype.value if hasattr(stype, "value") else str(stype)
                        print(f"  {struct_type} at {pos}: inv={struct.inventory_amount}, dist={dist}")

            # Check agent occupancy
            if hasattr(state, "agent_occupancy") and state.agent_occupancy:
                print(f"\nAgent occupancy: {list(state.agent_occupancy)[:10]}")

        print(f"\nRecent positions: {info.position_history[-10:]}")

    def diagnose_coordinate_system(self) -> None:
        """Diagnose coordinate system issues.

        NOTE: Internal coordinates are RELATIVE to each agent's starting position.
        They will NOT match absolute simulation coordinates, and that's expected.

        This diagnosis checks if the internal coordinate system is consistent:
        - Are agents tracking their positions correctly?
        - Can agents navigate to their believed assembler position?
        """
        print("\n=== Coordinate System Diagnosis ===")
        print("Note: Internal coords are relative to starting position (centered at ~100,100)")
        print("      Simulation coords are absolute (different coordinate system)")

        # Get actual assembler positions (for reference only)
        actual_assemblers = self.find_assemblers()
        print(f"\nActual assembler in simulation (absolute coords): {actual_assemblers}")

        # Check what agents believe (internal coords)
        print("\nAgent beliefs (internal relative coords):")
        for i in range(self.num_agents):
            info = self.agent_info[i]
            if info.assembler_pos:
                if info.internal_pos:
                    dx = abs(info.internal_pos[0] - info.assembler_pos[0])
                    dy = abs(info.internal_pos[1] - info.assembler_pos[1])
                    dist_to_assembler = dx + dy
                else:
                    dist_to_assembler = "?"
                print(
                    f"  Agent {i}: at {info.internal_pos}, assembler at {info.assembler_pos}, dist={dist_to_assembler}"
                )

        # Check for potential issues
        print("\nConsistency check:")
        issues = []
        for i in range(self.num_agents):
            info = self.agent_info[i]
            state = self.get_agent_state(i)

            if state and info.internal_pos:
                # Check if agent has been stuck trying to deposit (full cargo, near assembler)
                if info.assembler_pos and info.cargo >= info.cargo_capacity:
                    dx = abs(info.internal_pos[0] - info.assembler_pos[0])
                    dy = abs(info.internal_pos[1] - info.assembler_pos[1])
                    dist = dx + dy
                    if dist <= 2 and info.stuck_count > 20:
                        issues.append(i)
                        print(
                            f"  Agent {i}: STUCK near assembler with full cargo! "
                            f"(dist={dist}, stuck={info.stuck_count})"
                        )

        if issues:
            print(f"\n*** POTENTIAL ISSUE: Agents {issues} stuck near assembler ***")
            print("This could indicate deposits are failing or navigation issues.")
        else:
            print("  No obvious consistency issues detected.")

    def run_until(self, condition: Callable[["DebugHarness"], bool], max_steps: int = 1000) -> int:
        """Run simulation until condition is met or max_steps reached.

        Args:
            condition: Function that returns True when we should stop
            max_steps: Maximum steps to run

        Returns:
            Number of steps executed
        """
        steps = 0
        while steps < max_steps and not condition(self):
            self.step()
            steps += 1
        return steps

    def run_until_stuck(self, threshold: int = 10, max_steps: int = 500) -> list[int]:
        """Run until any agent is stuck for threshold steps.

        Returns:
            List of stuck agent IDs
        """

        def any_stuck(h: DebugHarness) -> bool:
            return any(info.stuck_count >= threshold for info in h.agent_info.values())

        self.run_until(any_stuck, max_steps)
        return self.diagnose_stuck_agents(threshold)


def main():
    """Example usage of debug harness."""
    print("Creating debug harness from cogsguard recipe...")
    harness = DebugHarness.from_recipe(num_agents=10, max_steps=1000, seed=42)

    print("\nSimulation info at start:")
    harness.print_simulation_info()

    print("\nRunning 50 steps...")
    harness.step(50)
    harness.print_agent_summary()

    print("\nRunning until agents get stuck (or 200 steps)...")
    stuck = harness.run_until_stuck(threshold=10, max_steps=200)

    if stuck:
        print(f"\nFound {len(stuck)} stuck agents. Running coordinate diagnosis...")
        harness.diagnose_coordinate_system()
    else:
        print("\nNo stuck agents found.")
        harness.print_agent_summary()


if __name__ == "__main__":
    main()
