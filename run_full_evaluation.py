"""Full evaluation with simplified hyperparameters."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "cogames" / "src"))

from cogames.cogs_vs_clips.difficulty_variants import EASY, MEDIUM, HARD, apply_difficulty
from cogames.cogs_vs_clips.exploration_experiments import Experiment1Mission, Experiment2Mission
from cogames.cogs_vs_clips.eval_missions import (
    OxygenBottleneck, GermaniumRush, SiliconWorkbench,
    CarbonDesert, SingleUseWorld, SlowOxygen, HighRegenSprint,
    SparseBalanced, GermaniumClutch
)
from cogames.policy.scripted_agent import MettaGridEnv, ScriptedAgentPolicy, Hyperparameters

# All test configurations
test_configs = [
    # Exploration experiments with difficulties
    ("EXP1-EASY", Experiment1Mission, EASY, 800),
    ("EXP1-MEDIUM", Experiment1Mission, MEDIUM, 1000),
    ("EXP1-HARD", Experiment1Mission, HARD, 1300),
    ("EXP2-EASY", Experiment2Mission, EASY, 800),
    ("EXP2-MEDIUM", Experiment2Mission, MEDIUM, 1000),
    ("EXP2-HARD", Experiment2Mission, HARD, 1300),
    # Eval missions (no difficulty variants)
    ("OXYGEN_BOTTLENECK", OxygenBottleneck, None, 800),
    ("GERMANIUM_RUSH", GermaniumRush, None, 800),
    ("SILICON_WORKBENCH", SiliconWorkbench, None, 800),
    ("CARBON_DESERT", CarbonDesert, None, 800),
    ("SINGLE_USE_WORLD", SingleUseWorld, None, 800),
    ("SLOW_OXYGEN", SlowOxygen, None, 800),
    ("HIGH_REGEN_SPRINT", HighRegenSprint, None, 800),
    ("SPARSE_BALANCED", SparseBalanced, None, 800),
    ("GERMANIUM_CLUTCH", GermaniumClutch, None, 800),
]

# All 5 simplified hyperparameter presets
strategies = {
    "explorer": Hyperparameters(
        strategy_type="explorer_first",
        exploration_phase_steps=100,
        min_energy_for_silicon=70,
    ),
    "greedy": Hyperparameters(
        strategy_type="greedy_opportunistic",
        exploration_phase_steps=0,
        min_energy_for_silicon=70,
    ),
    "efficiency": Hyperparameters(
        strategy_type="efficiency_learner",
        exploration_phase_steps=0,
        min_energy_for_silicon=70,
    ),
    "explorer_aggressive": Hyperparameters(
        strategy_type="explorer_first",
        exploration_phase_steps=100,
        min_energy_for_silicon=60,  # Gather silicon earlier
    ),
    "explorer_conservative": Hyperparameters(
        strategy_type="explorer_first",
        exploration_phase_steps=100,
        min_energy_for_silicon=85,  # Wait for high energy
    ),
}

print("=" * 80)
print("FULL EVALUATION: Simplified Hyperparameters (3 params, 5 presets)")
print("=" * 80)
print(f"Testing {len(test_configs)} environments × {len(strategies)} strategies = {len(test_configs) * len(strategies)} total runs")
print("=" * 80)

all_results = {}
completed = 0
total = len(test_configs) * len(strategies)

for env_name, mission_class, difficulty, steps in test_configs:
    print(f"\n[{completed}/{total}] {env_name}")
    env_results = {}

    for strat_name, hyperparams in strategies.items():
        try:
            mission = mission_class()
            if difficulty:
                apply_difficulty(mission, difficulty)
            mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
            env_config = mission.make_env()
            env_config.game.max_steps = steps
            env = MettaGridEnv(env_config)
            policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

            obs, info = env.reset()
            policy.reset(obs, info)
            ap = policy.agent_policy(0)

            for _ in range(steps):
                obs, _, d, t, _ = env.step([ap.step(obs[0])])
                if d[0] or t[0]:
                    break

            hearts = ap._state.hearts_assembled
            env_results[strat_name] = hearts

            completed += 1
            success_marker = "✅" if hearts >= 1 else "❌"
            print(f"  [{completed}/{total}] {strat_name:20s}: {hearts} hearts {success_marker}")
        except Exception as e:
            env_results[strat_name] = -1
            completed += 1
            print(f"  [{completed}/{total}] {strat_name:20s}: ERROR - {str(e)[:50]}")

    all_results[env_name] = env_results

# Save results
with open("full_evaluation_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 80)
print("SUMMARY BY STRATEGY")
print("=" * 80)

for strat_name in strategies.keys():
    successes = sum(1 for env_results in all_results.values()
                   if isinstance(env_results.get(strat_name), int) and env_results[strat_name] >= 1)
    total_envs = len(all_results)
    total_hearts = sum(env_results.get(strat_name, 0) for env_results in all_results.values()
                      if isinstance(env_results.get(strat_name), int))
    print(f"{strat_name:20s}: {successes:2d}/{total_envs} envs ({successes/total_envs*100:5.1f}%), {total_hearts:3d} total hearts")

print("\n" + "=" * 80)
print("SUMMARY BY ENVIRONMENT")
print("=" * 80)

for env_name, env_results in all_results.items():
    successful_strats = [s for s, h in env_results.items() if isinstance(h, int) and h >= 1]
    best_hearts = max((h for h in env_results.values() if isinstance(h, int)), default=0)

    if successful_strats:
        print(f"{env_name:20s}: ✅ {len(successful_strats)}/5 strategies succeeded (best: {best_hearts} hearts)")
    else:
        print(f"{env_name:20s}: ❌ No strategies succeeded")

print("\n✅ Full evaluation complete! Results saved to full_evaluation_results.json")

