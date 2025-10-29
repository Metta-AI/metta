#!/usr/bin/env python3

"""Test the streamlined hyperparameters to verify they create meaningful differences."""

import logging
from packages.cogames.src.cogames.cogs_vs_clips.missions import get_map
from packages.cogames.src.cogames.cogs_vs_clips.eval_missions import ClipOxygen
from packages.mettagrid.python.src.mettagrid import MettaGridEnv
from packages.cogames.src.cogames.policy.scripted_agent import ScriptedAgentPolicy
from packages.cogames.src.cogames.policy.hyperparameters_streamlined import (
    Hyperparameters,
    create_aggressive_preset,
    create_conservative_preset,
    create_impatient_preset,
    create_patient_preset,
    create_mixture_presets
)

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def run_evaluation(hyperparams, max_steps=200):
    """Run a single evaluation and return key metrics."""
    try:
        # Set up mission and environment
        map_builder = get_map("eval_clip_oxygen.map")
        mission = ClipOxygen()
        mission_instance = mission.instantiate(map_builder, num_cogs=1)
        env_cfg = mission_instance.make_env()
        env = MettaGridEnv(env_cfg)

        # Create agent with hyperparameters
        policy = ScriptedAgentPolicy(env=env, hyperparams=hyperparams)
        agent = policy.agent_policy(agent_id=0)

        obs, _ = env.reset()

        # Track metrics
        total_reward = 0
        hearts_deposited = 0
        steps_to_first_heart = None
        steps_to_unclip = None
        final_resources = {}
        phase_transitions = []
        energy_usage = []
        recharge_events = 0

        for step in range(max_steps):
            action = agent.step(obs[0])
            obs, rewards, terminals, truncations, infos = env.step([action])

            if rewards[0] != 0:
                total_reward += rewards[0]
                hearts_deposited += 1
                if steps_to_first_heart is None:
                    steps_to_first_heart = step + 1

            if hasattr(agent, '_state'):
                state = agent._state
                energy_usage.append(state.energy)

                # Track phase transitions
                if step == 0 or state.current_phase != phase_transitions[-1][1] if phase_transitions else True:
                    phase_transitions.append((step, state.current_phase.name))
                    if state.current_phase.name == "RECHARGE":
                        recharge_events += 1

                # Track unclipping
                if state.decoder > 0 and steps_to_unclip is None:
                    steps_to_unclip = step + 1

                # Final resources
                if step == max_steps - 1:
                    final_resources = {
                        'germanium': state.germanium,
                        'silicon': state.silicon,
                        'carbon': state.carbon,
                        'oxygen': state.oxygen,
                        'energy': state.energy,
                        'decoder': state.decoder
                    }

            if terminals[0] or truncations[0]:
                break

        return {
            'total_reward': total_reward,
            'hearts_deposited': hearts_deposited,
            'steps_to_first_heart': steps_to_first_heart,
            'steps_to_unclip': steps_to_unclip,
            'final_resources': final_resources,
            'phase_transitions': len(phase_transitions),
            'recharge_events': recharge_events,
            'avg_energy': sum(energy_usage) / len(energy_usage) if energy_usage else 0,
            'min_energy': min(energy_usage) if energy_usage else 0,
            'success': hearts_deposited > 0
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            'total_reward': 0,
            'hearts_deposited': 0,
            'steps_to_first_heart': None,
            'steps_to_unclip': None,
            'final_resources': {},
            'phase_transitions': 0,
            'recharge_events': 0,
            'avg_energy': 0,
            'min_energy': 0,
            'success': False,
            'error': str(e)
        }

def test_streamlined_presets():
    """Test the streamlined hyperparameter presets."""
    print("🧪 Testing Streamlined Hyperparameters")
    print("=" * 60)

    # Test different presets
    presets = [
        ("Default", Hyperparameters()),
        ("Aggressive", create_aggressive_preset()),
        ("Conservative", create_conservative_preset()),
        ("Impatient", create_impatient_preset()),
        ("Patient", create_patient_preset()),
    ]

    results = []

    for name, hyperparams in presets:
        print(f"\n--- Testing {name} Preset ---")
        print(f"  Energy: start_small={hyperparams.recharge_start_small}, start_large={hyperparams.recharge_start_large}")
        print(f"  Waiting: cooldown_leq={hyperparams.wait_if_cooldown_leq}")

        try:
            result = run_evaluation(hyperparams)
            result['preset_name'] = name
            results.append(result)

            print(f"  ✅ Success: {result['success']}")
            print(f"  Hearts: {result['hearts_deposited']}")
            print(f"  Steps to first heart: {result['steps_to_first_heart']}")
            print(f"  Steps to unclip: {result['steps_to_unclip']}")
            print(f"  Recharge events: {result['recharge_events']}")
            print(f"  Avg energy: {result['avg_energy']:.1f}")
            print(f"  Min energy: {result['min_energy']}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({'preset_name': name, 'success': False, 'error': str(e)})

    # Analyze results
    print(f"\n--- Analysis ---")

    # Check for behavioral differences
    key_metrics = ['success', 'hearts_deposited', 'steps_to_first_heart', 'recharge_events', 'avg_energy']

    print("Behavioral differences detected:")
    for metric in key_metrics:
        values = [r.get(metric, 0) for r in results if 'error' not in r]
        if len(set(values)) > 1:
            print(f"  {metric}: {min(values)} to {max(values)}")

    # Show specific differences
    print(f"\nDetailed comparison:")
    for result in results:
        if 'error' not in result:
            print(f"  {result['preset_name']}: "
                  f"hearts={result['hearts_deposited']}, "
                  f"recharge={result['recharge_events']}, "
                  f"avg_energy={result['avg_energy']:.1f}")

def test_mixture_presets():
    """Test the mixture presets."""
    print(f"\n--- Testing Mixture Presets ---")

    mixture_hps = create_mixture_presets()
    print(f"Created {len(mixture_hps)} mixture presets")

    for i, hp in enumerate(mixture_hps):
        print(f"  {i+1}. {hp.strategy_type} (seed={hp.seed})")
        print(f"     Energy: start_small={hp.recharge_start_small}, start_large={hp.recharge_start_large}")
        print(f"     Waiting: cooldown_leq={hp.wait_if_cooldown_leq}")

if __name__ == "__main__":
    test_streamlined_presets()
    test_mixture_presets()
    print(f"\n🎉 Streamlined hyperparameters test complete!")
