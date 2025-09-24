#!/usr/bin/env python3
"""
Demonstration of the Task Dependency Mock Environment.

This notebook demonstrates how to use the new task_dependency_mock_envs recipe
to simulate task chain learning dynamics without running mettagrid training.
"""

# %% [markdown]
# # Task Dependency Mock Environment Demo
#
# This demo shows how to use the new task dependency mock environment that implements
# the chain-based learning dynamics described in your research code. The system models:
#
# - Task dependency chains where parent tasks help child tasks
# - Performance dynamics with growth and forgetting
# - Task-specific noise generated from seeds
# - Integration with the existing learning progress curriculum system

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.recipes.task_dependency_mock_envs import (
    make_curriculum,
    make_mock_env_config,
    simulate_small_chain,
    simulate_high_gamma,
    simulate_high_forgetting,
)
from metta.cogworks.curriculum import Curriculum
from metta.rl.mock_dynamical_env import (
    MockDynamicalSystemSimulator,
)

# %% [markdown]
# ## Basic Simulator Usage
#
# First, let's create and test the simulator directly:

# %%
# Create a mock simulator with a small task chain
simulator = MockDynamicalSystemSimulator(
    num_tasks=5,
    num_epochs=20,
    samples_per_epoch=10,
    gamma=0.2,  # Parent contribution factor
    lambda_forget=0.1,  # Forgetting rate
    performance_threshold=0.8,
    task_seed=42,
)

print(f"Mock simulator created with {simulator.num_tasks} tasks")
print(f"Initial performance: {simulator.P}")
print("Task chain structure: 0 -> 1 -> 2 -> 3 -> 4")

# %% [markdown]
# ## Simulating Task Chain Dynamics
#
# Let's simulate the learning dynamics by sampling different tasks:

# %%
# Reset simulator and collect data
simulator.reset_epoch()
performances = [simulator.P.clone()]
sampling_counts = []

# Simulate curriculum learning by sampling tasks based on learning progress
for epoch in range(10):
    epoch_counts = torch.zeros(simulator.num_tasks)

    # Sample tasks for this epoch
    for step in range(simulator.samples_per_epoch):
        # Simple curriculum: sample from tasks with lowest performance
        task_probs = 1.0 / (simulator.P + 0.1)  # Inverse performance sampling
        task_probs = task_probs / task_probs.sum()
        task_id = torch.multinomial(task_probs, 1).item()

        reward = simulator.sample_task(task_id)
        epoch_counts[task_id] += 1

    # Complete epoch and update dynamics
    metrics = simulator.complete_epoch()
    performances.append(simulator.P.clone())
    sampling_counts.append(epoch_counts.clone())
    print(f"Epoch {epoch}: Mean performance = {simulator.P.mean():.3f}")

# %% [markdown]
# ## Visualizing Task Chain Learning
#
# Now let's visualize how the task chain learning progresses:

# %%
# Convert to numpy for plotting
performance_history = torch.stack(performances).numpy()
epochs = np.arange(len(performances))

# Plot performance over time
plt.figure(figsize=(12, 8))

# Plot 1: Task performance over time
plt.subplot(2, 2, 1)
for task_id in range(simulator.num_tasks):
    plt.plot(
        epochs,
        performance_history[:, task_id],
        label=f"Task {task_id}",
        marker="o",
        markersize=3,
    )
plt.xlabel("Epoch")
plt.ylabel("Task Performance")
plt.title("Task Performance Over Time")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Mean performance
plt.subplot(2, 2, 2)
mean_performance = performance_history.mean(axis=1)
plt.plot(epochs, mean_performance, "k-", linewidth=2, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Mean Performance")
plt.title("Overall Learning Progress")
plt.grid(True, alpha=0.3)

# Plot 3: Task dependency visualization
plt.subplot(2, 2, 3)
# Show the chain structure
task_positions = np.arange(simulator.num_tasks)
final_performances = performance_history[-1, :]
plt.bar(task_positions, final_performances, alpha=0.7)
plt.xlabel("Task ID (Chain Position)")
plt.ylabel("Final Performance")
plt.title("Final Performance by Chain Position")
for i in range(simulator.num_tasks - 1):
    plt.arrow(
        i + 0.4,
        final_performances[i],
        0.2,
        0,
        head_width=0.02,
        head_length=0.05,
        fc="red",
        ec="red",
    )
plt.grid(True, alpha=0.3)

# Plot 4: Sampling distribution
if sampling_counts:
    plt.subplot(2, 2, 4)
    sampling_matrix = torch.stack(sampling_counts).numpy()
    plt.imshow(sampling_matrix.T, aspect="auto", cmap="Blues")
    plt.xlabel("Epoch")
    plt.ylabel("Task ID")
    plt.title("Task Sampling Heatmap")
    plt.colorbar(label="Sample Count")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Recipe Configuration Examples
#
# The recipe provides several pre-configured training setups:

# %%
# Example 1: Small chain training
print("=== Small Chain Configuration ===")
small_config = make_mock_env_config(
    num_tasks=5,
    num_epochs=50,
    samples_per_epoch=25,
    gamma=0.1,
    lambda_forget=0.1,
)
print(f"Config: {small_config}")

# Example 2: High parent contribution
print("\n=== High Gamma Configuration ===")
high_gamma_config = make_mock_env_config(
    gamma=0.3,  # High parent contribution
    lambda_forget=0.05,  # Lower forgetting
)
print(f"Config: {high_gamma_config}")

# %% [markdown]
# ## Curriculum Integration
#
# The mock environment integrates with the existing curriculum system:

# %%
# Create curriculum configuration
curriculum_config = make_curriculum(
    mock_env_config=small_config,
    enable_detailed_slice_logging=True,
)

# Create curriculum instance
curriculum = Curriculum(curriculum_config, seed=42)

# Get tasks from curriculum
task1 = curriculum.get_task()
print(f"Task 1 ID: {task1._task_id}")

# Complete task and get new one
task1.complete(0.7)
curriculum.update_task_performance(task1._task_id, 0.7)

task2 = curriculum.get_task()
print(f"Task 2 ID: {task2._task_id}")

# Print curriculum stats
stats = curriculum.stats()
print(f"Curriculum stats: {list(stats.keys())}")

# %% [markdown]
# ## Simulation Configurations
#
# The recipe provides several simulation configurations ready for use:

# %%
# Small chain simulation
print("Running small chain simulation...")
results_small = simulate_small_chain(wandb_run_name="demo_small_chain")
print(f"Small chain results: {results_small['final_mean_performance']:.3f}")

# High gamma simulation
print("Running high gamma simulation...")
results_gamma = simulate_high_gamma(wandb_run_name="demo_high_gamma")
print(f"High gamma results: {results_gamma['final_mean_performance']:.3f}")

# High forgetting simulation
print("Running high forgetting simulation...")
results_forget = simulate_high_forgetting(wandb_run_name="demo_high_forgetting")
print(f"High forgetting results: {results_forget['final_mean_performance']:.3f}")

# %% [markdown]
# ## Integration with WandB Logging
#
# The simulator provides extensive metrics for logging:

# %%
# Demonstrate the metrics available for wandb logging
metrics = simulator._get_current_metrics()

print("Available metrics for wandb logging:")
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value}")
    elif isinstance(value, list) and len(value) <= 10:  # Show small lists
        print(f"  {key}: {value}")

# %% [markdown]
# ## Running with the Tool Pipeline
#
# To use this with the tool system, you can run:
#
# ```bash
# # Small chain experiment
# uv run ./tools/run.py experiments.recipes.task_dependency_mock_envs.simulate_small_chain
#
# # High gamma experiment
# uv run ./tools/run.py experiments.recipes.task_dependency_mock_envs.simulate_high_gamma
#
# # Custom configuration (direct Python)
# from experiments.recipes.task_dependency_mock_envs import simulate
# results = simulate(wandb_run_name="custom_test")
# ```
#
# The simulator will automatically log task performance metrics, completion probabilities,
# and learning progress statistics to wandb, allowing you to analyze the task dependency dynamics
# without any agent training overhead.

print("\n✅ Task Dependency Simulation Demo Complete!")
print("The simulator successfully implements the chain learning dynamics")
print("and integrates with the existing curriculum infrastructure for pure simulation.")
