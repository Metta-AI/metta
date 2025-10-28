"""Debug extractor depletion detection."""
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
impl = policy._impl

for step in range(400):
    state = ap._state

    if step % 50 == 0 and step > 0:
        ge_extractors = impl.extractor_memory.get_by_type("germanium")
        print(f"\nStep {step}:")
        print(f"  Germanium: {state.germanium}/5")
        print(f"  Phase: {state.current_phase.name}")
        print(f"  Extractors found: {len(ge_extractors)}")
        for i, ext in enumerate(ge_extractors):
            print(f"    Extractor {i}: depleted={ext.is_depleted()}, uses_left={ext.estimated_uses_left:.2f}, harvests={ext.total_harvests}")

    obs, _, d, t, _ = env.step([ap.step(obs[0])])

    if d[0] or t[0]:
        break

print(f"\nFinal germanium: {state.germanium}")
print(f"Unobtainable: {state.unobtainable_resources}")

