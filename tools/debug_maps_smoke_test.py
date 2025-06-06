#!/usr/bin/env -S uv run
"""
Smoke test suite for debug maps.

This script tests basic agent performance on the debug maps to ensure:
1. Maps load correctly without errors
2. Agents can navigate and complete basic tasks
3. Performance metrics are within acceptable ranges
4. No runtime errors occur during simulation

Usage:
    python tools/debug_maps_smoke_test.py [--policy-uri POLICY_URI] [--quick]
"""

import argparse
import logging
import sys
import time
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment


class DebugMapSmokeTestResults:
    """Results from running smoke tests on debug maps."""

    def __init__(self):
        self.map_results: Dict[str, Dict] = {}
        self.overall_success = True
        self.errors: List[str] = []

    def add_map_result(
        self, map_name: str, success: bool, avg_reward: float, avg_steps: int, completion_rate: float, errors: List[str]
    ):
        """Add results for a specific map."""
        self.map_results[map_name] = {
            "success": success,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "completion_rate": completion_rate,
            "errors": errors,
        }
        if not success:
            self.overall_success = False
            self.errors.extend(errors)

    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "=" * 60)
        print("DEBUG MAPS SMOKE TEST SUMMARY")
        print("=" * 60)

        if self.overall_success:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")

        print(f"\nTested {len(self.map_results)} maps:")

        for map_name, result in self.map_results.items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {map_name}:")
            print(f"   Avg Reward: {result['avg_reward']:.3f}")
            print(f"   Avg Steps: {result['avg_steps']}")
            print(f"   Completion Rate: {result['completion_rate']:.1%}")
            if result["errors"]:
                print(f"   Errors: {', '.join(result['errors'])}")

        if self.errors:
            print("\nOverall Errors:")
            for error in self.errors:
                print(f"  • {error}")


def test_single_map(
    map_name: str,
    env_config: str,
    policy_store: PolicyStore,
    policy_pr,
    cfg: DictConfig,
    logger: logging.Logger,
    quick_test: bool = False,
) -> Tuple[bool, Dict]:
    """
    Test a single debug map and return results.

    Args:
        map_name: Name of the map being tested
        env_config: Path to the environment configuration
        policy_store: PolicyStore instance
        policy_pr: Policy representation
        cfg: Hydra configuration
        logger: Logger instance
        quick_test: If True, run minimal tests for speed

    Returns:
        Tuple of (success, results_dict)
    """
    logger.info(f"Testing map: {map_name}")

    try:
        # Configure simulation for this map
        sim_config = {
            "name": f"debug_{map_name}",
            "max_time_s": 15 if quick_test else 30,
            "num_episodes": 1 if quick_test else 3,
            "simulations": {map_name: {"env": env_config}},
        }

        # Create simulation suite
        sim_suite_config = SimulationSuiteConfig.model_validate(sim_config)

        # Run simulation
        start_time = time.time()
        sim = SimulationSuite(
            config=sim_suite_config,
            policy_pr=policy_pr,
            policy_store=policy_store,
            replay_dir=f"/tmp/debug_smoke_test/{map_name}",
            stats_dir=f"/tmp/debug_smoke_test/{map_name}/stats",
            device=cfg.device,
            vectorization=cfg.vectorization,
        )

        results = sim.simulate()
        execution_time = time.time() - start_time

        # Extract performance metrics
        stats_db = results.stats_db

        # Get reward metrics
        reward_query = """
        SELECT AVG(value) as avg_reward, COUNT(*) as episodes
        FROM agent_metrics
        WHERE metric = 'reward'
        """
        reward_df = stats_db.query(reward_query)
        avg_reward = float(reward_df.iloc[0]["avg_reward"]) if len(reward_df) > 0 else 0.0
        episode_count = int(reward_df.iloc[0]["episodes"]) if len(reward_df) > 0 else 0

        # Get steps metrics
        steps_query = """
        SELECT AVG(value) as avg_steps
        FROM agent_metrics
        WHERE metric = 'steps'
        """
        steps_df = stats_db.query(steps_query)
        avg_steps = int(steps_df.iloc[0]["avg_steps"]) if len(steps_df) > 0 else 0

        # Calculate completion rate (episodes that achieved positive reward)
        completion_query = """
        SELECT COUNT(*) as completed_episodes
        FROM agent_metrics
        WHERE metric = 'reward' AND value > 0
        """
        completion_df = stats_db.query(completion_query)
        completed = int(completion_df.iloc[0]["completed_episodes"]) if len(completion_df) > 0 else 0
        completion_rate = completed / episode_count if episode_count > 0 else 0.0

        # Define success criteria
        min_reward = 0.1  # Minimum average reward
        max_avg_steps = 200  # Maximum average steps (efficiency check)
        min_completion_rate = 0.0  # At least some episodes should complete
        max_execution_time = 60  # Maximum execution time in seconds

        success = (
            avg_reward >= min_reward
            and avg_steps <= max_avg_steps
            and completion_rate >= min_completion_rate
            and execution_time <= max_execution_time
            and episode_count > 0
        )

        errors = []
        if avg_reward < min_reward:
            errors.append(f"Low reward: {avg_reward:.3f} < {min_reward}")
        if avg_steps > max_avg_steps:
            errors.append(f"Too many steps: {avg_steps} > {max_avg_steps}")
        if completion_rate < min_completion_rate:
            errors.append(f"Low completion rate: {completion_rate:.1%} < {min_completion_rate:.1%}")
        if execution_time > max_execution_time:
            errors.append(f"Slow execution: {execution_time:.1f}s > {max_execution_time}s")
        if episode_count == 0:
            errors.append("No episodes completed")

        logger.info(
            f"Map {map_name} - Success: {success}, Reward: {avg_reward:.3f}, "
            f"Steps: {avg_steps}, Completion: {completion_rate:.1%}, "
            f"Time: {execution_time:.1f}s"
        )

        return success, {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "completion_rate": completion_rate,
            "execution_time": execution_time,
            "episode_count": episode_count,
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"Error testing map {map_name}: {e}")
        return False, {
            "avg_reward": 0.0,
            "avg_steps": 0,
            "completion_rate": 0.0,
            "execution_time": 0.0,
            "episode_count": 0,
            "errors": [f"Exception: {str(e)}"],
        }


def run_debug_maps_smoke_test(policy_uri: str, quick_test: bool = False) -> DebugMapSmokeTestResults:
    """
    Run smoke tests on all debug maps.

    Args:
        policy_uri: URI of the policy to test
        quick_test: If True, run minimal tests for speed

    Returns:
        DebugMapSmokeTestResults with test outcomes
    """
    # Set up logging
    logger = setup_mettagrid_logger("debug_maps_smoke_test")

    # Load configuration
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="sim_job", overrides=[f"policy_uri={policy_uri}", "run=debug_smoke_test"])

    setup_mettagrid_environment(cfg)

    # Initialize policy store
    policy_store = PolicyStore(cfg, None)

    # Get policy
    policy_prs = policy_store.policies(policy_uri, "latest", n=1, metric="navigation_score")
    if not policy_prs:
        raise ValueError(f"No policies found for URI: {policy_uri}")

    policy_pr = policy_prs[0]
    logger.info(f"Testing with policy: {policy_pr.uri}")

    # Define debug maps to test
    debug_maps = {
        "mixed_objects": "env/mettagrid/debug/evals/debug_mixed_objects",
        "resource_collection": "env/mettagrid/debug/evals/debug_resource_collection",
        "simple_obstacles": "env/mettagrid/debug/evals/debug_simple_obstacles",
        "tiny_two_altars": "env/mettagrid/debug/evals/debug_tiny_two_altars",
    }

    # Run tests
    results = DebugMapSmokeTestResults()

    for map_name, env_config in debug_maps.items():
        success, map_results = test_single_map(map_name, env_config, policy_store, policy_pr, cfg, logger, quick_test)

        results.add_map_result(
            map_name=map_name,
            success=success,
            avg_reward=map_results["avg_reward"],
            avg_steps=map_results["avg_steps"],
            completion_rate=map_results["completion_rate"],
            errors=map_results["errors"],
        )

    return results


def main():
    """Main entry point for the smoke test script."""
    parser = argparse.ArgumentParser(description="Smoke test for debug maps")
    parser.add_argument(
        "--policy-uri", default="training_regular_envset", help="Policy URI to test (default: training_regular_envset)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests (1 episode per map, shorter time limits)")

    args = parser.parse_args()

    try:
        results = run_debug_maps_smoke_test(args.policy_uri, args.quick)
        results.print_summary()

        # Exit with appropriate code
        sys.exit(0 if results.overall_success else 1)

    except Exception as e:
        print(f"❌ Smoke test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
