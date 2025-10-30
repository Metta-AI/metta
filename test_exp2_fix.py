"""Test if adaptive exploration fixes EXP2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "cogames" / "src"))

from cogames.cogs_vs_clips.difficulty_variants import EASY, apply_difficulty
from cogames.cogs_vs_clips.exploration_experiments import Experiment2Mission
from cogames.policy.scripted_agent import MettaGridEnv, ScriptedAgentPolicy, Hyperparameters

mission = Experiment2Mission()
apply_difficulty(mission, EASY)
mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
env_config = mission.make_env()
env_config.game.max_steps = 800
env = MettaGridEnv(env_config)

hyperparams = Hyperparameters(strategy_type="explorer_first", exploration_phase_steps=100, min_energy_for_silicon=70)
policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

obs, info = env.reset()
policy.reset(obs, info)
ap = policy.agent_policy(0)
impl = policy._impl

print(f"EXP2-EASY with Adaptive Exploration")
print(f"Map: {impl._map_height}x{impl._map_width} = {impl._map_height * impl._map_width} cells")
print(f"Adaptive exploration steps: {impl._adaptive_exploration_steps}")
print("=" * 80)

positions = []
for step in range(800):
    state = ap._state
    positions.append((state.agent_row, state.agent_col))

    obs, _, d, t, _ = env.step([ap.step(obs[0])])

    if d[0] or t[0]:
        break

final_state = ap._state
unique_pos = len(set(positions))
coverage = (unique_pos / (impl._map_height * impl._map_width)) * 100

print(f"\nFinal Results:")
print(f"  Hearts: {final_state.hearts_assembled}")
print(f"  Steps: {final_state.step_count}/800")
print(f"  Coverage: {unique_pos} cells ({coverage:.1f}%)")
print(f"  Resources: Ge={final_state.germanium}/5, Si={final_state.silicon}/10, C={final_state.carbon}/5, O={final_state.oxygen}/5")
print(f"  Extractors: Ge={len(impl.extractor_memory.get_by_type('germanium'))}, "
      f"Si={len(impl.extractor_memory.get_by_type('silicon'))}, "
      f"C={len(impl.extractor_memory.get_by_type('carbon'))}, "
      f"O={len(impl.extractor_memory.get_by_type('oxygen'))}")

if coverage > 5:
    print(f"\n✅ IMPROVED: Coverage increased from 3.5% to {coverage:.1f}%")
else:
    print(f"\n⚠️  Still low coverage: {coverage:.1f}%")

if final_state.hearts_assembled > 0:
    print(f"✅ SUCCESS: Assembled {final_state.hearts_assembled} hearts!")

