#!/usr/bin/env python3
"""Comprehensive evaluation of scripted agent across all missions and hyperparameters."""

import json
import time
from datetime import datetime
from pathlib import Path

from cogames.cli.mission import get_mission
from cogames.cogs_vs_clips.eval_missions import EVAL_MISSIONS, MACHINA_EVAL
from cogames.policy.hyperparameters_streamlined import Hyperparameters, create_mixture_presets
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv


def run_evaluation():
    """Run comprehensive evaluation and save results."""

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"eval_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Log file
    log_file = output_dir / "evaluation.log"
    results_file = output_dir / "results.json"
    summary_file = output_dir / "summary.txt"

    # Hyperparameter sets to test
    hyperparams_sets = [
        ("default", Hyperparameters()),
        *[(f"preset_{i}", hp) for i, hp in enumerate(create_mixture_presets())],
    ]

    # Results storage
    all_results = []

    def log(msg):
        """Log to both console and file."""
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log("=" * 80)
    log(f"COMPREHENSIVE EVALUATION - {timestamp}")
    log("=" * 80)
    log(f"Total missions: {len(EVAL_MISSIONS)}")
    log(f"Hyperparameter sets: {len(hyperparams_sets)}")
    log(f"Episodes per config: 3")
    log(f"Total evaluations: {len(EVAL_MISSIONS) * len(hyperparams_sets) * 3}")
    log("=" * 80)
    log("")

    total_evals = 0
    successful_evals = 0
    failed_evals = 0

    # Test each mission
    for mission_idx, mission_class in enumerate(EVAL_MISSIONS, 1):
        mission = mission_class()
        log(f"\n[{mission_idx}/{len(EVAL_MISSIONS)}] Mission: {mission.name}")
        log(f"  Description: {mission.description}")
        log(f"  Map: {mission.map_name}")

        # Test with each hyperparameter set
        for hp_name, hyperparams in hyperparams_sets:
            log(f"  Testing with hyperparams: {hp_name}")

            # Run 3 episodes
            for episode in range(3):
                total_evals += 1
                seed = 42 + episode

                try:
                    # Create environment using proper mission resolution
                    mission_full_name = f"machina_eval.{mission.name}"
                    _, env_cfg, _ = get_mission(mission_full_name, cogs=1)
                    env = MettaGridEnv(env_cfg=env_cfg)

                    # Create policy
                    policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)
                    agent = policy.agent_policy(0)

                    # Reset
                    obs, info = env.reset(seed=seed)
                    agent.reset()

                    # Run episode
                    start_time = time.time()
                    total_reward = 0
                    steps = 0
                    done = False

                    for step in range(1000):
                        action = agent.step(obs)
                        result = env.step([action])

                        if len(result) == 4:
                            obs, reward, done_arr, info = result
                        else:
                            obs, reward, done_arr, truncation, info = result

                        total_reward += reward[0]
                        steps = step + 1

                        if done_arr[0]:
                            done = True
                            break

                    elapsed_time = time.time() - start_time

                    # Record results
                    result_data = {
                        "mission": mission.name,
                        "hyperparams": hp_name,
                        "episode": episode,
                        "seed": seed,
                        "steps": steps,
                        "reward": float(total_reward),
                        "done": done,
                        "time_seconds": elapsed_time,
                        "success": True,
                    }

                    all_results.append(result_data)
                    successful_evals += 1

                    log(f"    Episode {episode}: reward={total_reward:.2f}, steps={steps}, done={done}, time={elapsed_time:.2f}s")

                except Exception as e:
                    failed_evals += 1
                    log(f"    Episode {episode}: FAILED - {e}")
                    result_data = {
                        "mission": mission.name,
                        "hyperparams": hp_name,
                        "episode": episode,
                        "seed": seed,
                        "error": str(e),
                        "success": False,
                    }
                    all_results.append(result_data)

        # Save intermediate results
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Generate summary
    log("\n" + "=" * 80)
    log("EVALUATION COMPLETE")
    log("=" * 80)
    log(f"Total evaluations: {total_evals}")
    log(f"Successful: {successful_evals}")
    log(f"Failed: {failed_evals}")
    log(f"Success rate: {100 * successful_evals / total_evals:.1f}%")
    log("")

    # Aggregate statistics
    log("AGGREGATE STATISTICS:")
    log("-" * 80)

    # By mission
    mission_stats = {}
    for result in all_results:
        if not result.get("success"):
            continue
        mission = result["mission"]
        if mission not in mission_stats:
            mission_stats[mission] = {"rewards": [], "steps": [], "done_count": 0}
        mission_stats[mission]["rewards"].append(result["reward"])
        mission_stats[mission]["steps"].append(result["steps"])
        if result["done"]:
            mission_stats[mission]["done_count"] += 1

    log("\nTop 10 missions by average reward:")
    sorted_missions = sorted(
        mission_stats.items(),
        key=lambda x: sum(x[1]["rewards"]) / len(x[1]["rewards"]) if x[1]["rewards"] else 0,
        reverse=True
    )[:10]

    for mission, stats in sorted_missions:
        avg_reward = sum(stats["rewards"]) / len(stats["rewards"]) if stats["rewards"] else 0
        avg_steps = sum(stats["steps"]) / len(stats["steps"]) if stats["steps"] else 0
        completion_rate = 100 * stats["done_count"] / len(stats["rewards"]) if stats["rewards"] else 0
        log(f"  {mission:40s} | Reward: {avg_reward:6.2f} | Steps: {avg_steps:6.1f} | Completion: {completion_rate:5.1f}%")

    # By hyperparameter set
    log("\nPerformance by hyperparameter set:")
    hp_stats = {}
    for result in all_results:
        if not result.get("success"):
            continue
        hp = result["hyperparams"]
        if hp not in hp_stats:
            hp_stats[hp] = {"rewards": [], "steps": [], "done_count": 0}
        hp_stats[hp]["rewards"].append(result["reward"])
        hp_stats[hp]["steps"].append(result["steps"])
        if result["done"]:
            hp_stats[hp]["done_count"] += 1

    for hp, stats in sorted(hp_stats.items()):
        avg_reward = sum(stats["rewards"]) / len(stats["rewards"]) if stats["rewards"] else 0
        avg_steps = sum(stats["steps"]) / len(stats["steps"]) if stats["steps"] else 0
        completion_rate = 100 * stats["done_count"] / len(stats["rewards"]) if stats["rewards"] else 0
        log(f"  {hp:20s} | Reward: {avg_reward:6.2f} | Steps: {avg_steps:6.1f} | Completion: {completion_rate:5.1f}%")

    # Save summary
    with open(summary_file, "w") as f:
        f.write(f"Evaluation completed at {datetime.now()}\n")
        f.write(f"Total evaluations: {total_evals}\n")
        f.write(f"Successful: {successful_evals}\n")
        f.write(f"Failed: {failed_evals}\n")
        f.write(f"Success rate: {100 * successful_evals / total_evals:.1f}%\n")

    log(f"\nResults saved to: {output_dir}")
    log(f"  - Full log: {log_file}")
    log(f"  - JSON results: {results_file}")
    log(f"  - Summary: {summary_file}")
    log("")


if __name__ == "__main__":
    run_evaluation()

