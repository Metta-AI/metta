"""Debug resource gathering tracking timing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "cogames" / "src"))

from cogames.cogs_vs_clips.eval_missions import GermaniumClutch
from cogames.policy.scripted_agent import MettaGridEnv, ScriptedAgentPolicy, Hyperparameters

mission = GermaniumClutch()
mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
env_config = mission.make_env()
env_config.game.max_steps = 400
env = MettaGridEnv(env_config)

hyperparams = Hyperparameters(strategy_type="efficiency_learner", exploration_phase_steps=100, min_energy_for_silicon=70)
policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

obs, info = env.reset()
policy.reset(obs, info)
ap = policy.agent_policy(0)

for step in range(400):
    state = ap._state

    if step % 100 == 0 and step > 0:
        if "germanium" in state.resource_gathering_start:
            start = state.resource_gathering_start["germanium"]
            total_time = state.step_count - start
            tracked = state.resource_progress_tracking.get("germanium", "NOT SET")
            current = state.germanium
            print(f"\nStep {step}:")
            print(f"  Germanium: {current}/5")
            print(f"  Gathering start: {start}")
            print(f"  Total time trying: {total_time}")
            print(f"  Tracked start amount: {tracked}")
            print(f"  Progress: {current - tracked if tracked != 'NOT SET' else 'N/A'}")
            print(f"  Unobtainable: {state.unobtainable_resources}")
        else:
            print(f"\nStep {step}: Germanium tracking not initialized yet")

    obs, _, d, t, _ = env.step([ap.step(obs[0])])

    if d[0] or t[0]:
        break

print(f"\nFinal germanium: {state.germanium}")
print(f"Unobtainable: {state.unobtainable_resources}")

