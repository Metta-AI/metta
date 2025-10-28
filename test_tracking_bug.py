"""Debug tracking initialization bug."""
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
env_config.game.max_steps = 300
env = MettaGridEnv(env_config)

hyperparams = Hyperparameters(strategy_type="explorer_first", exploration_phase_steps=100, min_energy_for_silicon=70)
policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

obs, info = env.reset()
policy.reset(obs, info)
ap = policy.agent_policy(0)

print("Tracking silicon collection and phase visits...")

last_phase = None
for step in range(300):
    state = ap._state

    if state.current_phase != last_phase and state.current_phase.name.startswith("GATHER_"):
        resource = state.current_phase.name.replace("GATHER_", "").lower()
        visit_count = state.phase_visit_count.get(resource, 0)
        tracked_amount = state.resource_progress_tracking.get(resource, "NOT SET")
        current_amount = getattr(state, resource)
        print(f"Step {step}: {state.current_phase.name} (visit #{visit_count})")
        print(f"  Current {resource}: {current_amount}")
        print(f"  Tracked start: {tracked_amount}")
        if tracked_amount != "NOT SET":
            progress = current_amount - tracked_amount
            print(f"  Progress: {progress}")
        last_phase = state.current_phase

    obs, _, d, t, _ = env.step([ap.step(obs[0])])

    if d[0] or t[0]:
        break

print(f"\nFinal silicon: {state.silicon}")
print(f"Unobtainable: {state.unobtainable_resources}")

