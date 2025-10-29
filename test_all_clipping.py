#!/usr/bin/env python3
"""Test all clipping missions to see current performance."""

from cogames.cli.mission import get_mission
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv

CLIPPING_MISSIONS = [
    "machina_eval.clip_oxygen",
    "machina_eval.clip_carbon",
    "machina_eval.clip_germanium",
    "machina_eval.clip_silicon",
]

def test_mission(mission_name: str, max_steps: int = 1000):
    """Test a single mission and return results."""
    print(f"\n{'='*80}")
    print(f"Testing: {mission_name}")
    print(f"{'='*80}")

    _, env_cfg, _ = get_mission(mission_name, cogs=1)
    env = MettaGridEnv(env_cfg=env_cfg)

    # Create policy
    policy = ScriptedAgentPolicy(env)
    impl = policy._impl
    agent = policy.agent_policy(0)

    # Print loaded recipes
    print(f"Loaded recipes: {impl._unclip_recipes}")

    # Reset
    obs, info = env.reset(seed=42)
    agent.reset()

    # Run episode
    total_reward = 0
    phase_changes = []
    last_phase = None

    for step in range(max_steps):
        action = agent.step(obs[0])  # Pass single agent's observation
        result = env.step([action])

        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False

        total_reward += reward[0] if hasattr(reward, '__len__') else reward

        # Track phase changes
        current_phase = impl.phase_controller.current().name
        if current_phase != last_phase:
            # Get inventory from observation
            carbon = impl._read_int_feature(obs[0], "inv:carbon")
            oxygen = impl._read_int_feature(obs[0], "inv:oxygen")
            germanium = impl._read_int_feature(obs[0], "inv:germanium")
            silicon = impl._read_int_feature(obs[0], "inv:silicon")
            energy = impl._read_int_feature(obs[0], "inv:energy")
            decoder = impl._read_int_feature(obs[0], "inv:decoder")
            modulator = impl._read_int_feature(obs[0], "inv:modulator")
            resonator = impl._read_int_feature(obs[0], "inv:resonator")
            scrambler = impl._read_int_feature(obs[0], "inv:scrambler")

            phase_changes.append({
                'step': step,
                'phase': current_phase,
                'carbon': carbon,
                'oxygen': oxygen,
                'germanium': germanium,
                'silicon': silicon,
                'energy': energy,
                'decoder': decoder,
                'modulator': modulator,
                'resonator': resonator,
                'scrambler': scrambler,
            })
            last_phase = current_phase

        if done or truncated:
            print(f"Episode ended at step {step}")
            break

    # Print phase changes
    print(f"\nPhase changes ({len(phase_changes)}):")
    for pc in phase_changes[:20]:  # Show first 20
        print(f"  Step {pc['step']:4d}: {pc['phase']:20s} | "
              f"C={pc['carbon']} O={pc['oxygen']} G={pc['germanium']} S={pc['silicon']} "
              f"E={pc['energy']:3d} | "
              f"Dec={pc['decoder']} Mod={pc['modulator']} Res={pc['resonator']} Scr={pc['scrambler']}")

    if len(phase_changes) > 20:
        print(f"  ... ({len(phase_changes) - 20} more phase changes)")

    print(f"\nTotal reward: {total_reward}")
    print(f"Success: {'YES' if total_reward >= 6 else 'NO'}")

    return {
        'mission': mission_name,
        'reward': total_reward,
        'steps': step,
        'success': total_reward >= 6,
        'phase_changes': len(phase_changes),
    }

if __name__ == "__main__":
    results = []
    for mission in CLIPPING_MISSIONS:
        try:
            result = test_mission(mission)
            results.append(result)
        except Exception as e:
            print(f"ERROR testing {mission}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'mission': mission,
                'reward': 0,
                'steps': 0,
                'success': False,
                'error': str(e),
            })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{status} {r['mission']:30s} | Reward: {r['reward']:6.1f} | Steps: {r.get('steps', 0):4d}")

    success_count = sum(1 for r in results if r['success'])
    print(f"\nSuccess rate: {success_count}/{len(results)}")

