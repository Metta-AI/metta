#!/usr/bin/env python3
"""
Benchmark to compare performance of original vs optimized no_agent_interference implementations.
"""

import time
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.util.hydra import get_cfg


@dataclass
class BenchmarkResult:
    """Store benchmark results for analysis."""
    name: str
    times: List[float]
    steps_per_second: float
    mean_time: float
    std_time: float


def create_test_config(num_agents: int, map_size: int, no_agent_interference: bool):
    """Create a test configuration for benchmarking."""
    cfg = get_cfg("benchmark")
    
    cfg.game.num_agents = num_agents
    cfg.game.max_steps = 10000
    cfg.game.episode_truncates = True
    cfg.game.no_agent_interference = no_agent_interference
    cfg.game.obs_width = 7
    cfg.game.obs_height = 7
    cfg.game.track_movement_metrics = False  # Disable to focus on core movement performance
    
    # Create a map with specified size and agents
    cfg.game.map_builder = OmegaConf.create({
        "_target_": "metta.mettagrid.room.random.Random",
        "width": map_size,
        "height": map_size,
        "objects": {},
        "agents": num_agents,
        "seed": 42,
    })
    
    # Don't override actions - use the default from the config
    
    return cfg


def benchmark_configuration(
    num_agents: int,
    map_size: int,
    no_agent_interference: bool,
    num_steps: int,
    action_distribution: str = "mixed"
) -> Tuple[float, int]:
    """
    Benchmark a specific configuration.
    
    Args:
        num_agents: Number of agents in the environment
        map_size: Size of the map (width and height)
        no_agent_interference: Whether to enable the flag
        num_steps: Number of steps to run
        action_distribution: Type of actions to test ("move", "move_8way", "mixed")
    
    Returns:
        Tuple of (elapsed_time, actual_steps_taken)
    """
    cfg = create_test_config(num_agents, map_size, no_agent_interference)
    
    # Create environment
    curriculum = SingleTaskCurriculum("benchmark", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)
    
    # Get action indices
    action_names = env.action_names
    move_idx = action_names.index("move") if "move" in action_names else None
    move_8way_idx = action_names.index("move_8way") if "move_8way" in action_names else None
    noop_idx = action_names.index("noop") if "noop" in action_names else 0
    
    # Reset environment
    obs, _ = env.reset()
    
    # Prepare action selection based on distribution
    if action_distribution == "move":
        action_choices = [move_idx] if move_idx is not None else [noop_idx]
    elif action_distribution == "move_8way":
        action_choices = [move_8way_idx] if move_8way_idx is not None else [noop_idx]
    else:  # mixed
        action_choices = []
        if move_idx is not None:
            action_choices.append(move_idx)
        if move_8way_idx is not None:
            action_choices.append(move_8way_idx)
        if not action_choices:
            action_choices = [noop_idx]
    
    # Run benchmark
    start_time = time.perf_counter()
    actual_steps = 0
    
    for step in range(num_steps):
        # Create actions for all agents
        if len(action_choices) > 1:
            # Random mix of actions
            actions = np.random.choice(action_choices, size=(num_agents, 1), replace=True).astype(np.int32)
        else:
            # All agents do the same action
            actions = np.full((num_agents, 1), action_choices[0], dtype=np.int32)
        
        # For move actions, add random arguments
        if action_distribution in ["move", "mixed"]:
            # Move has arg 0 (forward) or 1 (backward)
            for i in range(num_agents):
                if actions[i, 0] == move_idx:
                    actions[i, 0] = (actions[i, 0] << 8) | np.random.randint(0, 2)
        
        if action_distribution in ["move_8way", "mixed"]:
            # Move_8way has args 0-7 for 8 directions
            for i in range(num_agents):
                if actions[i, 0] == move_8way_idx:
                    actions[i, 0] = (actions[i, 0] << 8) | np.random.randint(0, 8)
        
        obs, rewards, dones, truncated, info = env.step(actions)
        actual_steps += 1
        
        if dones.any() or truncated.any():
            obs, _ = env.reset()
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    env.close()
    
    return elapsed, actual_steps


def run_benchmark_suite():
    """Run a comprehensive benchmark suite comparing implementations."""
    print("=" * 80)
    print("BENCHMARKING NO_AGENT_INTERFERENCE OPTIMIZATION")
    print("=" * 80)
    print()
    
    # Test configurations
    test_configs = [
        # (num_agents, map_size, action_distribution, description)
        (4, 10, "move", "Small map, few agents, move only"),
        (4, 10, "move_8way", "Small map, few agents, move_8way only"),
        (16, 20, "move", "Medium map, many agents, move only"),
        (16, 20, "move_8way", "Medium map, many agents, move_8way only"),
        (16, 20, "mixed", "Medium map, many agents, mixed actions"),
        (32, 30, "mixed", "Large map, very many agents, mixed actions"),
    ]
    
    num_steps = 5000
    num_runs = 3
    
    results = []
    
    for num_agents, map_size, action_dist, description in test_configs:
        print(f"\nTesting: {description}")
        print(f"  Agents: {num_agents}, Map: {map_size}x{map_size}, Actions: {action_dist}")
        print("-" * 60)
        
        # Test with flag = False (normal collision detection)
        times_false = []
        for run in range(num_runs):
            elapsed, steps = benchmark_configuration(
                num_agents, map_size, False, num_steps, action_dist
            )
            times_false.append(elapsed)
            print(f"  Run {run+1} (interference=False): {elapsed:.3f}s ({steps} steps)")
        
        mean_false = np.mean(times_false)
        std_false = np.std(times_false)
        sps_false = num_steps / mean_false
        
        results.append(BenchmarkResult(
            name=f"{description}\n(normal collision)",
            times=times_false,
            steps_per_second=sps_false,
            mean_time=mean_false,
            std_time=std_false
        ))
        
        # Test with flag = True (ghost movement)
        times_true = []
        for run in range(num_runs):
            elapsed, steps = benchmark_configuration(
                num_agents, map_size, True, num_steps, action_dist
            )
            times_true.append(elapsed)
            print(f"  Run {run+1} (interference=True):  {elapsed:.3f}s ({steps} steps)")
        
        mean_true = np.mean(times_true)
        std_true = np.std(times_true)
        sps_true = num_steps / mean_true
        
        results.append(BenchmarkResult(
            name=f"{description}\n(ghost mode)",
            times=times_true,
            steps_per_second=sps_true,
            mean_time=mean_true,
            std_time=std_true
        ))
        
        # Calculate improvement
        improvement = ((mean_false - mean_true) / mean_false) * 100
        print(f"\n  Summary:")
        print(f"    Normal collision: {mean_false:.3f} ± {std_false:.3f}s ({sps_false:.1f} steps/s)")
        print(f"    Ghost mode:       {mean_true:.3f} ± {std_true:.3f}s ({sps_true:.1f} steps/s)")
        print(f"    Improvement:      {improvement:.1f}%")
    
    return results


def plot_results(results: List[BenchmarkResult]):
    """Create visualization of benchmark results."""
    # Group results by configuration
    configs = []
    normal_sps = []
    ghost_sps = []
    
    for i in range(0, len(results), 2):
        config_name = results[i].name.split('\n')[0]
        configs.append(config_name)
        normal_sps.append(results[i].steps_per_second)
        ghost_sps.append(results[i+1].steps_per_second)
    
    # Create bar plot
    x = np.arange(len(configs))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Steps per second comparison
    bars1 = ax1.bar(x - width/2, normal_sps, width, label='Normal Collision', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, ghost_sps, width, label='Ghost Mode (Optimized)', color='green', alpha=0.7)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Steps per Second')
    ax1.set_title('Performance Comparison: Original vs Optimized no_agent_interference')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    # Speedup ratio
    speedup = [g/n for n, g in zip(normal_sps, ghost_sps)]
    bars3 = ax2.bar(x, speedup, width*2, color='orange', alpha=0.7)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup Ratio (Ghost / Normal)')
    ax2.set_title('Speedup from Compile-Time Optimization')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No speedup')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to benchmark_results.png")
    plt.show()


def main():
    """Run the complete benchmark suite."""
    results = run_benchmark_suite()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    # Try to create visualization
    try:
        plot_results(results)
    except ImportError:
        print("\nNote: matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    # Print final summary
    print("\nFinal Summary:")
    print("-" * 60)
    
    total_improvement = []
    for i in range(0, len(results), 2):
        normal = results[i]
        ghost = results[i+1]
        improvement = ((ghost.steps_per_second - normal.steps_per_second) / normal.steps_per_second) * 100
        total_improvement.append(improvement)
        print(f"{normal.name.split(chr(10))[0]}:")
        print(f"  Performance gain: {improvement:+.1f}%")
    
    avg_improvement = np.mean(total_improvement)
    print(f"\nAverage performance gain: {avg_improvement:+.1f}%")
    
    print("\nConclusion:")
    if avg_improvement > 0:
        print(f"✓ The compile-time optimization provides an average {avg_improvement:.1f}% speedup")
    else:
        print(f"✗ The optimization did not provide a speedup (average: {avg_improvement:.1f}%)")


if __name__ == "__main__":
    main()