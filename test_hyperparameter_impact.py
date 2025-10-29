#!/usr/bin/env python3

"""Test individual hyperparameters to identify which ones create meaningful behavioral differences."""

import logging
import numpy as np
from packages.cogames.src.cogames.cogs_vs_clips.missions import get_map
from packages.cogames.src.cogames.cogs_vs_clips.eval_missions import ClipOxygen
from packages.mettagrid.python.src.mettagrid import MettaGridEnv
from packages.cogames.src.cogames.policy.scripted_agent import ScriptedAgentPolicy
from packages.cogames.src.cogames.policy.hyperparameters import Hyperparameters

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
            'avg_energy': np.mean(energy_usage) if energy_usage else 0,
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
            'avg_energy': 0,
            'min_energy': 0,
            'success': False,
            'error': str(e)
        }

def test_hyperparameter_impact(param_name, values, baseline_value=None):
    """Test the impact of varying a single hyperparameter."""
    print(f"\n{'='*60}")
    print(f"Testing {param_name}")
    print(f"{'='*60}")

    results = []

    for i, value in enumerate(values):
        print(f"\n--- Test {i+1}/{len(values)}: {param_name} = {value} ---")

        # Create hyperparameters with this value
        hyperparams = Hyperparameters()
        setattr(hyperparams, param_name, value)

        # Run evaluation
        result = run_evaluation(hyperparams)
        result['param_value'] = value
        results.append(result)

        # Print key metrics
        print(f"  Success: {result['success']}")
        print(f"  Hearts: {result['hearts_deposited']}")
        print(f"  Steps to first heart: {result['steps_to_first_heart']}")
        print(f"  Steps to unclip: {result['steps_to_unclip']}")
        print(f"  Phase transitions: {result['phase_transitions']}")
        print(f"  Avg energy: {result['avg_energy']:.1f}")
        print(f"  Min energy: {result['min_energy']}")
        if result['final_resources']:
            print(f"  Final resources: G={result['final_resources'].get('germanium', 0)} "
                  f"Si={result['final_resources'].get('silicon', 0)} "
                  f"C={result['final_resources'].get('carbon', 0)} "
                  f"O={result['final_resources'].get('oxygen', 0)}")

    # Analyze results
    print(f"\n--- Analysis for {param_name} ---")

    # Check if all results are identical
    key_metrics = ['success', 'hearts_deposited', 'steps_to_first_heart', 'phase_transitions']
    all_identical = True

    for metric in key_metrics:
        values_list = [r[metric] for r in results]
        if len(set(values_list)) > 1:
            all_identical = False
            break

    if all_identical:
        print(f"❌ {param_name}: All results identical - REMOVE")
        return False
    else:
        print(f"✅ {param_name}: Behavioral differences detected - KEEP")

        # Show variation in key metrics
        for metric in key_metrics:
            values_list = [r[metric] for r in results]
            if len(set(values_list)) > 1:
                print(f"  {metric}: {min(values_list)} to {max(values_list)}")

        return True

def main():
    """Test all hyperparameters systematically."""
    print("🧪 Testing Hyperparameter Impact")
    print("Running 10 evaluations per parameter to detect behavioral differences")

    # Define test parameters and their value ranges
    test_params = {
        # Energy Management
        'recharge_start_small': np.linspace(40, 80, 10).astype(int),
        'recharge_stop_small': np.linspace(70, 100, 10).astype(int),
        'passive_regen_per_step': np.linspace(0.5, 1.5, 10),
        'risk_buffer_moves': np.linspace(0, 20, 10).astype(int),

        # Waiting Behavior
        'will_wait_max_steps': np.linspace(1, 50, 10).astype(int),
        'wait_if_cooldown_leq': np.linspace(0, 5, 10).astype(int),
        'rotate_on_cooldown_ge': np.linspace(1, 10, 10).astype(int),
        'prefer_short_queue_radius': np.linspace(1, 20, 10).astype(int),

        # Scoring
        'distance_weight': np.linspace(0.1, 0.9, 10),
        'efficiency_weight': np.linspace(0.1, 0.9, 10),
        'depletion_threshold': np.linspace(0.1, 0.5, 10),
        'clip_avoidance_bias': np.linspace(0.0, 0.8, 10),

        # Phase Policy
        'assembler_greediness': np.linspace(0.3, 1.0, 10),
        'germ_needed_base': np.linspace(3, 8, 10).astype(int),

        # Pathfinding
        'astar_threshold': np.linspace(5, 50, 10).astype(int),
        'bfs_sweep_bias': np.linspace(0.0, 1.0, 10),
    }

    # Test each parameter
    impactful_params = []
    non_impactful_params = []

    for param_name, values in test_params.items():
        is_impactful = test_hyperparameter_impact(param_name, values)

        if is_impactful:
            impactful_params.append(param_name)
        else:
            non_impactful_params.append(param_name)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Impactful parameters ({len(impactful_params)}):")
    for param in impactful_params:
        print(f"  - {param}")

    print(f"\n❌ Non-impactful parameters ({len(non_impactful_params)}):")
    for param in non_impactful_params:
        print(f"  - {param}")

    print(f"\nRecommendation: Remove {len(non_impactful_params)} non-impactful parameters")
    print(f"Keep {len(impactful_params)} impactful parameters for behavioral diversity")

if __name__ == "__main__":
    main()
