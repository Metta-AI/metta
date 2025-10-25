"""Quick validation of Phase 2 improvements (A* pathfinding + cooldown waiting)."""

import logging
import sys

from cogames.cogs_vs_clips.exploration_experiments import (
    Experiment1Mission,
    Experiment2Mission,
    Experiment3Mission,
)
from cogames.policy.scripted_agent_outpost import Hyperparameters, ScriptedAgentPolicy
from mettagrid import MettaGridEnv

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("cogames.policy.scripted_agent_outpost").setLevel(logging.INFO)


def test_experiment(mission_cls, max_steps=1500):
    """Test a single experiment and return results."""
    mission_instance = mission_cls()
    mission_name = f"{mission_instance.site.name}.{mission_instance.name}"

    print(f"\nTesting {mission_name}...")

    mission = mission_instance.instantiate(mission_instance.site.map_builder, num_cogs=1)
    env_config = mission.make_env()
    env = MettaGridEnv(env_config)

    policy = ScriptedAgentPolicy(env)
    agent_policies = [policy.agent_policy(i) for i in range(env.c_env.num_agents)]
    impl = policy._impl

    obs, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        actions = [ap.step(obs[i]) for i, ap in enumerate(agent_policies)]
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += sum(rewards)

        if total_reward > 0:
            print(f"✓ SUCCESS at step {step+1}! Reward={total_reward:.2f}")
            break

        if terminated[0] or truncated[0]:
            break

    success = total_reward > 0
    state = agent_policies[0]._state

    print(
        f"  {'✓' if success else '✗'} {mission_name}: "
        f"Reward={total_reward:.2f}, Steps={step+1}, "
        f"Hearts={state.hearts_assembled}"
    )

    if not success:
        print(
            f"     Final: C={state.carbon}, O={state.oxygen}, G={state.germanium}, "
            f"Si={state.silicon}, E={state.energy}"
        )

    return mission_name, total_reward, success, step + 1


def main():
    print("=" * 80)
    print("PHASE 2 IMPROVEMENTS - Quick Validation")
    print("Testing critical fixes:")
    print("  1. A* pathfinding for long distances")
    print("  2. Cooldown waiting for resource timing")
    print("=" * 80)

    results = []

    # Test Exp 1 (baseline - should still work)
    print("\n" + "=" * 80)
    print("EXP1: Baseline (30x30) - Should succeed")
    print("=" * 80)
    results.append(test_experiment(Experiment1Mission))

    # Test Exp 2 (80x80 maze - WAS FAILING, should now succeed with A*)
    print("\n" + "=" * 80)
    print("EXP2: Oxygen Abundance (80x80 maze) - KEY TEST for A* pathfinding")
    print("=" * 80)
    results.append(test_experiment(Experiment2Mission, max_steps=2000))

    # Test Exp 3 (low efficiency - WAS FAILING, should now succeed with cooldown waiting)
    print("\n" + "=" * 80)
    print("EXP3: Low Efficiency (50x50) - KEY TEST for cooldown waiting")
    print("=" * 80)
    results.append(test_experiment(Experiment3Mission, max_steps=2000))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful_runs = sum(1 for _, _, s, _ in results if s)
    total_runs = len(results)
    success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0

    print(f"Success Rate: {successful_runs}/{total_runs} ({success_rate:.1f}%)")
    print()
    for name, reward, success, steps in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: Reward={reward:.2f}, Steps={steps}")

    print("\n" + "=" * 80)
    if successful_runs == total_runs:
        print("🎉 ALL TESTS PASSED! Phase 2 improvements are working!")
    elif successful_runs > 1:
        print("⚠️  PARTIAL SUCCESS - Some improvements working, needs more work")
    else:
        print("❌ TESTS FAILED - Need to debug")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()

