#!/usr/bin/env python3
"""
Simple Evaluation Script for CoGames Scripted Agent

This script runs evaluations on missions with available maps and tests
different hyperparameter configurations.
"""

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from cogames.cogs_vs_clips.difficulty_variants import (
    DIFFICULTY_LEVELS,
    apply_difficulty,
    get_difficulty,
)
from cogames.cogs_vs_clips.eval_missions import EVAL_MISSIONS
from cogames.policy.hyperparameters_streamlined import Hyperparameters
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv
from cogames.cogs_vs_clips.missions import get_map
from mettagrid.map_builder.map_builder import MapBuilderConfig


def get_available_missions():
    """Get missions that have available map files."""
    available_missions = []

    for mission_class in EVAL_MISSIONS:
        try:
            mission = mission_class()
            # Try to get the map to see if it exists
            get_map(mission.map_name)
            available_missions.append(mission_class)
            print(f"✓ {mission.name}: {mission.map_name}")
        except Exception as e:
            print(f"✗ {mission_class().name}: {mission.map_name} - {e}")

    return available_missions


def create_hyperparameter_sets() -> List[Tuple[str, Hyperparameters]]:
    """Create different hyperparameter configurations for testing."""
    hyperparams = []

    # Baseline configuration
    hyperparams.append(("baseline", Hyperparameters()))

    # Conservative energy management
    hyperparams.append(("conservative", Hyperparameters(
        recharge_start_small=80,
        recharge_start_large=60,
        wait_if_cooldown_leq=1,
        seed=42
    )))

    # Aggressive energy management
    hyperparams.append(("aggressive", Hyperparameters(
        recharge_start_small=50,
        recharge_start_large=30,
        wait_if_cooldown_leq=3,
        seed=123
    )))

    # Patient waiting strategy
    hyperparams.append(("patient", Hyperparameters(
        recharge_start_small=70,
        recharge_start_large=50,
        wait_if_cooldown_leq=0,  # Always try to use immediately
        seed=456
    )))

    # Impatient waiting strategy
    hyperparams.append(("impatient", Hyperparameters(
        recharge_start_small=60,
        recharge_start_large=40,
        wait_if_cooldown_leq=5,  # Wait longer before giving up
        seed=789
    )))

    return hyperparams


def run_single_evaluation(
    mission_class,
    difficulty_name: str,
    hyperparam_name: str,
    hyperparams: Hyperparameters,
    seed: int,
    max_steps: int = 1000
) -> Dict[str, Any]:
    """Run a single evaluation and return results."""
    try:
        # Create mission instance
        mission = mission_class()

        # Apply difficulty if specified
        if difficulty_name != "default":
            difficulty = get_difficulty(difficulty_name)
            apply_difficulty(mission, difficulty)

        # Create environment
        map_builder = MapBuilderConfig()
        num_cogs = 1
        instantiated_mission = mission.instantiate(map_builder, num_cogs)
        config = instantiated_mission.make_env()
        env = MettaGridEnv(config)

        # Create agent with hyperparameters
        policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)
        agent = policy.agent_policy(0)  # Get agent policy for agent 0

        # Reset environment
        obs, info = env.reset()
        agent.reset()

        # Run episode
        total_reward = 0
        step_count = 0
        success = False
        hearts_assembled = 0
        hearts_deposited = 0

        for step in range(max_steps):
            action = agent.step(obs)
            # Environment expects actions as array for multi-agent setup
            result = env.step([action])
            if len(result) == 4:
                obs, reward, done, info = result
            elif len(result) == 5:
                obs, reward, done, truncation, info = result
            else:
                raise ValueError(f"Unexpected number of return values: {len(result)}")
            total_reward += reward
            step_count = step + 1

            # Track mission progress
            if hasattr(agent, '_state') and hasattr(agent._state, 'hearts_assembled'):
                hearts_assembled = agent._state.hearts_assembled
            if hasattr(agent, '_state') and hasattr(agent._state, 'hearts_deposited'):
                hearts_deposited = agent._state.hearts_deposited

            # Check for success (heart deposited)
            if hearts_deposited > 0:
                success = True
                break

            if done:
                break

        # Extract final state information
        final_energy = obs.get('energy', 0) if isinstance(obs, dict) else 0
        final_carbon = obs.get('carbon', 0) if isinstance(obs, dict) else 0
        final_oxygen = obs.get('oxygen', 0) if isinstance(obs, dict) else 0
        final_germanium = obs.get('germanium', 0) if isinstance(obs, dict) else 0
        final_silicon = obs.get('silicon', 0) if isinstance(obs, dict) else 0

        return {
            'mission_name': mission.name,
            'difficulty': difficulty_name,
            'hyperparams': hyperparam_name,
            'seed': seed,
            'success': success,
            'total_reward': total_reward,
            'steps_taken': step_count,
            'hearts_assembled': hearts_assembled,
            'hearts_deposited': hearts_deposited,
            'final_energy': final_energy,
            'final_carbon': final_carbon,
            'final_oxygen': final_oxygen,
            'final_germanium': final_germanium,
            'final_silicon': final_silicon,
            'efficiency': total_reward / max(step_count, 1),
            'error': None
        }

    except Exception as e:
        return {
            'mission_name': mission_class().name if hasattr(mission_class, 'name') else str(mission_class),
            'difficulty': difficulty_name,
            'hyperparams': hyperparam_name,
            'seed': seed,
            'success': False,
            'total_reward': 0,
            'steps_taken': 0,
            'hearts_assembled': 0,
            'hearts_deposited': 0,
            'final_energy': 0,
            'final_carbon': 0,
            'final_oxygen': 0,
            'final_germanium': 0,
            'final_silicon': 0,
            'efficiency': 0,
            'error': str(e)
        }


def run_evaluation(
    missions: List = None,
    difficulties: List[str] = None,
    hyperparams: List[Tuple[str, Hyperparameters]] = None,
    seeds: List[int] = None,
    max_steps: int = 1000,
    output_dir: str = "evaluation_results"
) -> List[Dict[str, Any]]:
    """Run evaluation across all combinations."""

    # Default values
    if missions is None:
        missions = get_available_missions()
    if difficulties is None:
        difficulties = ["default", "easy", "medium", "hard"]
    if hyperparams is None:
        hyperparams = create_hyperparameter_sets()
    if seeds is None:
        seeds = [1, 2, 3]  # 3 seeds for statistical significance

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Calculate total number of evaluations
    total_evals = len(missions) * len(difficulties) * len(hyperparams) * len(seeds)
    print(f"Running {total_evals} total evaluations...")
    print(f"Missions: {len(missions)}")
    print(f"Difficulties: {len(difficulties)}")
    print(f"Hyperparameter sets: {len(hyperparams)}")
    print(f"Seeds per combination: {len(seeds)}")

    results = []
    completed = 0
    start_time = time.time()

    # Run evaluations
    for mission_class in missions:
        for difficulty in difficulties:
            for hyperparam_name, hyperparam_config in hyperparams:
                for seed in seeds:
                    print(f"\nRunning: {mission_class().name} | {difficulty} | {hyperparam_name} | seed={seed}")

                    # Set seed for reproducibility
                    random.seed(seed)
                    np.random.seed(seed)

                    result = run_single_evaluation(
                        mission_class=mission_class,
                        difficulty_name=difficulty,
                        hyperparam_name=hyperparam_name,
                        hyperparams=hyperparam_config,
                        seed=seed,
                        max_steps=max_steps
                    )

                    results.append(result)
                    completed += 1

                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_evals - completed) / rate if rate > 0 else 0

                    print(f"Progress: {completed}/{total_evals} ({completed/total_evals*100:.1f}%)")
                    print(f"Rate: {rate:.2f} evals/sec, ETA: {eta/60:.1f} minutes")
                    reward_val = result['total_reward']
                    if hasattr(reward_val, '__len__') and len(reward_val) > 0:
                        reward_val = reward_val[0]
                    print(f"Success: {result['success']}, Reward: {reward_val:.2f}")

    # Save results
    save_results(results, output_dir)

    return results


def save_results(results: List[Dict[str, Any]], output_dir: str):
    """Save results to CSV and JSON files."""
    timestamp = int(time.time())

    # Save detailed results as CSV
    csv_file = Path(output_dir) / f"evaluation_results_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Save summary statistics
    summary_file = Path(output_dir) / f"evaluation_summary_{timestamp}.json"
    summary = calculate_summary_statistics(results)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Detailed: {csv_file}")
    print(f"  Summary: {summary_file}")


def calculate_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from results."""
    if not results:
        return {}

    # Group by mission, difficulty, and hyperparams
    grouped = {}
    for result in results:
        key = (result['mission_name'], result['difficulty'], result['hyperparams'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Calculate statistics for each group
    summary = {}
    for key, group_results in grouped.items():
        mission_name, difficulty, hyperparams = key

        success_rate = sum(1 for r in group_results if r['success']) / len(group_results)
        avg_reward = np.mean([r['total_reward'] for r in group_results])
        avg_steps = np.mean([r['steps_taken'] for r in group_results])
        avg_efficiency = np.mean([r['efficiency'] for r in group_results])

        summary[f"{mission_name}_{difficulty}_{hyperparams}"] = {
            'mission_name': mission_name,
            'difficulty': difficulty,
            'hyperparams': hyperparams,
            'num_runs': len(group_results),
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'avg_efficiency': avg_efficiency,
            'std_reward': np.std([r['total_reward'] for r in group_results]),
            'std_steps': np.std([r['steps_taken'] for r in group_results]),
            'std_efficiency': np.std([r['efficiency'] for r in group_results])
        }

    # Overall statistics
    all_successes = [r['success'] for r in results]
    all_rewards = [r['total_reward'] for r in results]
    all_steps = [r['steps_taken'] for r in results]
    all_efficiencies = [r['efficiency'] for r in results]

    summary['overall'] = {
        'total_evaluations': len(results),
        'overall_success_rate': sum(all_successes) / len(all_successes),
        'overall_avg_reward': np.mean(all_rewards),
        'overall_avg_steps': np.mean(all_steps),
        'overall_avg_efficiency': np.mean(all_efficiencies),
        'overall_std_reward': np.std(all_rewards),
        'overall_std_steps': np.std(all_steps),
        'overall_std_efficiency': np.std(all_efficiencies)
    }

    return summary


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of results."""
    if not results:
        print("No results to summarize.")
        return

    # Overall statistics
    total_runs = len(results)
    successful_runs = sum(1 for r in results if r['success'])
    success_rate = successful_runs / total_runs

    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Success rate: {success_rate:.2%}")

    # Success rate by hyperparameter set
    print(f"\nSuccess rate by hyperparameter set:")
    hyperparam_groups = {}
    for result in results:
        hp = result['hyperparams']
        if hp not in hyperparam_groups:
            hyperparam_groups[hp] = []
        hyperparam_groups[hp].append(result)

    for hp, group in hyperparam_groups.items():
        hp_success_rate = sum(1 for r in group if r['success']) / len(group)
        print(f"  {hp}: {hp_success_rate:.2%} ({sum(1 for r in group if r['success'])}/{len(group)})")

    # Success rate by difficulty
    print(f"\nSuccess rate by difficulty:")
    difficulty_groups = {}
    for result in results:
        diff = result['difficulty']
        if diff not in difficulty_groups:
            difficulty_groups[diff] = []
        difficulty_groups[diff].append(result)

    for diff, group in difficulty_groups.items():
        diff_success_rate = sum(1 for r in group if r['success']) / len(group)
        print(f"  {diff}: {diff_success_rate:.2%} ({sum(1 for r in group if r['success'])}/{len(group)})")

    # Success rate by mission type (original vs clipped)
    print(f"\nSuccess rate by mission type:")
    original_success = 0
    original_total = 0
    clipped_success = 0
    clipped_total = 0

    for result in results:
        if result['mission_name'].startswith('clip_'):
            clipped_total += 1
            if result['success']:
                clipped_success += 1
        else:
            original_total += 1
            if result['success']:
                original_success += 1

    if original_total > 0:
        print(f"  Original missions: {original_success/original_total:.2%} ({original_success}/{original_total})")
    if clipped_total > 0:
        print(f"  Clipped missions: {clipped_success/clipped_total:.2%} ({clipped_success}/{clipped_total})")


if __name__ == "__main__":
    print("Starting evaluation...")

    # First, check available missions
    print("\nChecking available missions:")
    available_missions = get_available_missions()
    print(f"\nFound {len(available_missions)} available missions")

    # Run evaluation
    results = run_evaluation()

    # Print summary
    print_summary(results)

    print("\nEvaluation complete!")
