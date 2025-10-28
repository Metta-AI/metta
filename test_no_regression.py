"""Test that adaptive exploration doesn't break working environments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "cogames" / "src"))

from cogames.cogs_vs_clips.eval_missions import (
    OxygenBottleneck, GermaniumRush, SiliconWorkbench, CarbonDesert,
    SlowOxygen, HighRegenSprint, SparseBalanced
)
from cogames.policy.scripted_agent import MettaGridEnv, ScriptedAgentPolicy, Hyperparameters

# Test environments that were working before
working_envs = [
    ("OXYGEN_BOTTLENECK", OxygenBottleneck),
    ("GERMANIUM_RUSH", GermaniumRush),
    ("SILICON_WORKBENCH", SiliconWorkbench),
    ("CARBON_DESERT", CarbonDesert),
]

print("=" * 80)
print("REGRESSION TEST: Checking working environments still work")
print("=" * 80)

results = {}

for env_name, mission_class in working_envs:
    mission = mission_class()
    mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
    env_config = mission.make_env()
    env_config.game.max_steps = 800
    env = MettaGridEnv(env_config)

    hyperparams = Hyperparameters(strategy_type="explorer_first", exploration_phase_steps=100, min_energy_for_silicon=70)
    policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

    obs, info = env.reset()
    policy.reset(obs, info)
    ap = policy.agent_policy(0)

    for step in range(800):
        obs, _, d, t, _ = env.step([ap.step(obs[0])])
        if d[0] or t[0]:
            break

    final_state = ap._state
    hearts = final_state.hearts_assembled
    results[env_name] = hearts

    status = "✅" if hearts >= 2 else "❌"
    print(f"{status} {env_name}: {hearts} hearts")

print("\n" + "=" * 80)
total = len(results)
passed = sum(1 for h in results.values() if h >= 2)
print(f"RESULT: {passed}/{total} environments still working")

if passed == total:
    print("✅ NO REGRESSION: All working environments still work!")
elif passed >= total - 1:
    print("⚠️  MINOR REGRESSION: 1 environment affected")
else:
    print("❌ MAJOR REGRESSION: Multiple environments broken!")

