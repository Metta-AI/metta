"""Test if assembly works with 3/4 resources."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "cogames" / "src"))

from cogames.cogs_vs_clips.eval_missions import GermaniumClutch
from cogames.policy.scripted_agent import MettaGridEnv, ScriptedAgentPolicy, Hyperparameters

mission = GermaniumClutch()
mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
env_config = mission.make_env()
env_config.game.max_steps = 800
env = MettaGridEnv(env_config)

hyperparams = Hyperparameters(strategy_type="efficiency_learner", exploration_phase_steps=100, min_energy_for_silicon=70)
policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

obs, info = env.reset()
policy.reset(obs, info)
ap = policy.agent_policy(0)

print("GERMANIUM_CLUTCH with efficiency_learner")
print("Expected: Ge=4/5, Si=50/10, C=20/5, O=20/5")
print("Should assemble with 3/4 resources if germanium marked unobtainable")
print("=" * 80)

for step in range(800):
    state = ap._state

    # Log when resources are marked unobtainable
    if step % 100 == 0 and step > 0:
        print(f"\nStep {step}:")
        print(f"  Resources: Ge={state.germanium}/5, Si={state.silicon}/10, C={state.carbon}/5, O={state.oxygen}/5")
        print(f"  Unobtainable: {state.unobtainable_resources}")
        print(f"  Phase: {state.current_phase.name}")
        print(f"  Hearts: {state.hearts_assembled}")

    obs, _, d, t, _ = env.step([ap.step(obs[0])])

    if d[0] or t[0]:
        break

final_state = ap._state

print(f"\n{'=' * 80}")
print(f"FINAL RESULTS")
print(f"{'=' * 80}")
print(f"Hearts: {final_state.hearts_assembled}")
print(f"Resources: Ge={final_state.germanium}/5, Si={final_state.silicon}/10, C={final_state.carbon}/5, O={final_state.oxygen}/5")
print(f"Unobtainable: {final_state.unobtainable_resources}")
print(f"Phase: {final_state.current_phase.name}")

if final_state.hearts_assembled > 0:
    print(f"\n✅ SUCCESS: Assembled {final_state.hearts_assembled} hearts with 3/4 resources!")
elif len(final_state.unobtainable_resources) > 0:
    print(f"\n⚠️  Marked resources as unobtainable but didn't assemble")
else:
    print(f"\n❌ FAILED: No resources marked unobtainable, no hearts assembled")

