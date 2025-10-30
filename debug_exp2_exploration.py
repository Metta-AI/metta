"""Debug why EXP2 has such low exploration coverage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "cogames" / "src"))

from cogames.cogs_vs_clips.difficulty_variants import EASY, apply_difficulty
from cogames.cogs_vs_clips.exploration_experiments import Experiment1Mission, Experiment2Mission
from cogames.policy.scripted_agent import MettaGridEnv, ScriptedAgentPolicy, Hyperparameters

print("=" * 80)
print("COMPARING EXP1 vs EXP2 EXPLORATION")
print("=" * 80)

for exp_name, mission_class in [("EXP1-EASY", Experiment1Mission), ("EXP2-EASY", Experiment2Mission)]:
    print(f"\n{'=' * 80}")
    print(f"{exp_name}")
    print('=' * 80)

    mission = mission_class()
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
    impl = policy._impl

    print(f"\nMap size: {impl._map_height}x{impl._map_width} = {impl._map_height * impl._map_width} cells")

    # Track exploration
    positions = []
    phases = []
    frontiers_found = []

    for step in range(300):
        state = ap._state
        positions.append((state.agent_row, state.agent_col))
        phases.append(state.current_phase.name)

        # Log exploration phase details
        if step % 50 == 0 and step > 0:
            unique_pos = len(set(positions))
            coverage = (unique_pos / (impl._map_height * impl._map_width)) * 100
            print(f"\nStep {step}:")
            print(f"  Position: ({state.agent_row}, {state.agent_col})")
            print(f"  Phase: {state.current_phase.name}")
            print(f"  Coverage: {unique_pos} cells ({coverage:.1f}%)")
            print(f"  Energy: {state.energy}")

            # Check frontiers
            if state.current_phase.name == "EXPLORE":
                # Count frontiers
                frontier_count = 0
                for r in range(impl._map_height):
                    for c in range(impl._map_width):
                        if impl._occ[r][c] == impl.OCC_FREE:
                            for nr, nc in impl._neighbors4(r, c):
                                if impl._occ[nr][nc] == impl.OCC_UNKNOWN:
                                    frontier_count += 1
                                    break
                print(f"  Frontiers available: {frontier_count}")

        obs, _, d, t, _ = env.step([ap.step(obs[0])])

        if d[0] or t[0]:
            break

    # Final stats
    unique_pos = len(set(positions))
    coverage = (unique_pos / (impl._map_height * impl._map_width)) * 100

    print(f"\n{'=' * 80}")
    print(f"FINAL STATS")
    print('=' * 80)
    print(f"Steps: {len(positions)}")
    print(f"Unique positions: {unique_pos}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"Phases seen: {set(phases)}")
    print(f"Time in EXPLORE: {phases.count('EXPLORE')} steps")
    print(f"Time in RECHARGE: {phases.count('RECHARGE')} steps")

    # Check if agent is stuck
    if len(positions) > 100:
        last_100 = positions[-100:]
        unique_last_100 = len(set(last_100))
        print(f"Movement in last 100 steps: {unique_last_100} unique positions")
        if unique_last_100 < 10:
            print(f"  ⚠️  Agent appears STUCK (only {unique_last_100} positions in last 100 steps)")

print("\n" + "=" * 80)

