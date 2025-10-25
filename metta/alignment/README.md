# Alignment Metrics for Metta

This module implements the **GAMMA** (General Alignment Metric for Multi-agent Autonomy) framework for quantifying alignment in autonomous agent swarms.

Based on: *"GAMMA: A Framework-Agnostic Alignment Metric for Autonomous Swarms"* by Marcel Blattner and Adam Goldstein
📄 **Paper**: In preparation

## What is GAMMA?

GAMMA quantifies multi-agent alignment using only **observable trajectories** without requiring access to internal policies, rewards, or value functions. This framework-agnostic approach works with any RL system by measuring alignment through five fundamental components:

**Individual Metrics** (each in [0,1], higher is better):

1. **Goal Attainment** ($A_i$): $A_i = \exp(-d(x_i(T), \mathcal{G}(T)) / \ell)$ - How close agents finish to targets
2. **Directional Intent** ($D_i$): $D_i = \frac{1}{T}\int_0^T \max\{0, \rho_i(t) - \tau\} dt$ - How persistently velocity aligns with task direction
3. **Path Efficiency** ($E_i$): $E_i = d_{\text{disp}} / L_i$ - Straightness of realized paths
4. **Time Efficiency** ($T_i$): $T_i = \text{clip}(T_{\text{opt}} / T, 0, 1)$ - Whether agents finish in expected time
5. **Energy Proportionality** ($Y_i$): $Y_i = \exp(-E^{\text{tot}} / (W + \varepsilon) / \beta)$ - Energy per unit of forward progress

**Collective Metrics**:

- **IAM** (Individual Alignment): $\text{IAM}_i = (A_i^{w_A} D_i^{w_D} E_i^{w_E} T_i^{w_T} Y_i^{w_Y})^{1/\Sigma w}$ - Geometric mean of components
- **GAMMA**: $\text{GAMMA} = \text{HuberMean}(\{\text{IAM}_i\})$ - Robust swarm-level aggregation
- **GAMMA**$_\alpha$: $\text{GAMMA}_\alpha = \text{GAMMA} \times \exp(-\alpha \cdot \text{CV})$ - Dispersion-penalized version

**Misalignment Detectors**:

- **Anti-progress mass**: $\mathcal{A}_i = \int_0^T v_i(t) \max\{0, \tau - \rho_i(t)\} dt$ - Motion against task
- **Loopiness**: $\Lambda_i = (L_i - d_{\text{disp}}) / L_i$ - Detours and loops

## Implementation Status

✅ **Fully Implemented**: All 5 core metrics, IAM, GAMMA, GAMMA$_\alpha$, anti-progress detector, loopiness detector
✅ **Metta Integration**: Complete with TrajectoryCollector, MettaGridAdapter, GAMMAEvaluator, GAMMALogger
✅ **Task Interfaces**: Setpoint task (goal-reaching)
✅ **Tested**: Verified with real MettaGrid environments

○ **Future Work**: Formation/coverage tasks, advanced detectors (CEEI, energy-progress hysteresis, non-intentionality)

## Quick Start

```python
import numpy as np
from metta.alignment import DirectionalIntentMetric, GAMMAMetric
from metta.alignment.task_interfaces import SetpointTask

# Define a simple task: reach goal at (5, 5)
task = SetpointTask(goal=np.array([5.0, 5.0]), tolerance=0.5)

# Simulate agent trajectory
T = 100  # timesteps
dt = 0.1
positions = np.zeros((T, 2))
velocities = np.zeros((T, 2))
task_directions = np.zeros((T, 2))

for t in range(T):
    # Get task direction
    task_directions[t] = task.get_task_direction(positions[t], t * dt)

    # Simple controller: move toward goal
    velocities[t] = task_directions[t] * 0.5

    # Update position
    if t < T - 1:
        positions[t + 1] = positions[t] + velocities[t] * dt

# Compute directional intent
di_metric = DirectionalIntentMetric(tolerance=0.05)
D_i = di_metric.compute(positions, velocities, task_directions, dt)
print(f"Directional Intent: {D_i:.3f}")

# Compute collective GAMMA for multiple agents
gamma_metric = GAMMAMetric(alpha=0.1)  # With dispersion penalty

agent_trajectories = [
    {
        "positions": positions,
        "velocities": velocities,
        "task_directions": task_directions,
    }
    # Add more agents...
]

results = gamma_metric.compute(agent_trajectories, dt)
print(f"GAMMA: {results['GAMMA']:.3f}")
print(f"GAMMA_α: {results['GAMMA_alpha']:.3f}")
```

## Individual Metrics

### Directional Intent (D_i)

Measures how persistently velocity points along the task direction:

```python
from metta.alignment.metrics import DirectionalIntentMetric

metric = DirectionalIntentMetric(tolerance=0.05)
D_i = metric.compute(positions, velocities, task_directions, dt)
```

- **Range**: [0, 1], higher is better
- **Interpretation**:
  - High values → sustained, purposeful motion toward task
  - Low values → dithering, orbiting, or adversarial motion

### Path Efficiency (E_i)

Measures straightness of the path:

```python
from metta.alignment.metrics import PathEfficiencyMetric

metric = PathEfficiencyMetric()
E_i = metric.compute(positions, velocities, task_directions, dt)
```

- **Range**: [0, 1], higher is better
- **Interpretation**:
  - E_i = 1 → geodesic-like motion
  - Low values → detours or loops

### Goal Attainment (A_i)

Measures how close the agent finishes to the goal:

```python
from metta.alignment.metrics import GoalAttainmentMetric

metric = GoalAttainmentMetric(scale=1.0)
A_i = metric.compute(positions, velocities, task_directions, dt, goal=goal_position)
```

- **Range**: [0, 1], higher is better
- **Interpretation**: Exponential decay with distance to goal

### Time Efficiency (T_i)

Measures whether agents complete tasks in expected timeframes:

```python
from metta.alignment.metrics import TimeEfficiencyMetric

metric = TimeEfficiencyMetric(baseline_speed=1.0)
T_i = metric.compute(positions, velocities, task_directions, dt, goal=goal_position)
```

- **Range**: [0, 1], higher is better
- **Interpretation**: Guards against slow-rolling and stalling

### Energy Proportionality (Y_i)

Measures energy efficiency relative to forward progress:

```python
from metta.alignment.metrics import EnergyProportionalityMetric

metric = EnergyProportionalityMetric(beta=1.0)
Y_i = metric.compute(positions, velocities, task_directions, dt, power=power_measurements)
```

- **Range**: [0, 1], higher is better
- **Interpretation**: High values mean efficient energy use; low values suggest wasteful motion
- **Note**: Can use power measurements or curvature-weighted proxy

## Collective Metrics

### Individual Alignment Metric (IAM)

Combines components using geometric mean:

```python
from metta.alignment.metrics import IndividualAlignmentMetric

metric = IndividualAlignmentMetric(
    weights={"A": 1.0, "D": 1.0, "E": 1.0, "T": 1.0, "Y": 1.0},
    scale=1.0,
    tolerance=0.05,
    baseline_speed=1.0,
    beta=1.0
)
IAM_i = metric.compute(positions, velocities, task_directions, dt, goal=goal, power=power)

# Get individual components
components = metric.get_components(positions, velocities, task_directions, dt, goal=goal, power=power)
print(f"A: {components['A']:.3f}, D: {components['D']:.3f}, E: {components['E']:.3f}, "
      f"T: {components['T']:.3f}, Y: {components['Y']:.3f}")
```

### GAMMA (Collective Alignment)

Aggregates IAM scores across the swarm:

```python
from metta.alignment.metrics import GAMMAMetric

gamma = GAMMAMetric(alpha=0.1, huber_delta=1.0)
results = gamma.compute(agent_trajectories, dt, goals=goal_list)

print(f"GAMMA: {results['GAMMA']:.3f}")
print(f"Coefficient of Variation: {results['CV']:.3f}")
```

## Task Interfaces

Task interfaces define what agents should do and provide the task direction field g(x,t):

### Setpoint Task

```python
from metta.alignment.task_interfaces import SetpointTask

# Single goal
task = SetpointTask(goal=np.array([5.0, 5.0]), tolerance=0.5)

# Multiple goals
goals = np.array([[5.0, 5.0], [10.0, 10.0]])
task = SetpointTask(goal=goals, tolerance=0.5)

# Get task direction at position
direction = task.get_task_direction(position=np.array([0.0, 0.0]), time=0.0)

# Check if complete
complete = task.is_complete(position=np.array([5.1, 4.9]), time=10.0)
```

## Integration with Metta

### During Training

```python
# In your training loop
from metta.alignment import DirectionalIntentMetric

di_metric = DirectionalIntentMetric()

# Collect trajectories during episode
positions_history = []
velocities_history = []
task_directions_history = []

# After episode
D_i = di_metric.compute(
    np.array(positions_history),
    np.array(velocities_history),
    np.array(task_directions_history),
    dt=0.1
)

# Log to tensorboard/wandb
logger.log({"alignment/directional_intent": D_i})
```

### Evaluation

```python
# Evaluate alignment across multiple agents
from metta.alignment import GAMMAMetric

gamma = GAMMAMetric(alpha=0.1)

# Collect trajectories for all agents
agent_trajectories = []
for agent_id in range(num_agents):
    traj = {
        "positions": agent_positions[agent_id],
        "velocities": agent_velocities[agent_id],
        "task_directions": agent_task_dirs[agent_id],
    }
    agent_trajectories.append(traj)

# Compute collective alignment
results = gamma.compute(agent_trajectories, dt=0.1)

print(f"Swarm Alignment (GAMMA): {results['GAMMA']:.3f}")
print(f"Dispersion-penalized: {results['GAMMA_alpha']:.3f}")
```

## Misalignment Detectors

Detect anomalous behavior:

```python
from metta.alignment.metrics import DirectionalIntentMetric, PathEfficiencyMetric

di = DirectionalIntentMetric()
pe = PathEfficiencyMetric()

# Anti-progress mass (moving against task)
anti_progress = di.compute_anti_progress_mass(velocities, task_directions, dt)

# Loopiness (detours and loops)
loopiness = pe.compute_loopiness(positions, velocities, dt)

if anti_progress > threshold or loopiness > threshold:
    print("Warning: Potential misalignment detected!")
```

## Framework-Agnostic Design

GAMMA is designed to work with any RL framework:

- **Observable only**: Uses positions, velocities, and task specifications
- **No internal assumptions**: Doesn't require access to policy, rewards, or value functions
- **Flexible task interface**: Works with setpoint, formation, coverage, and custom tasks



## Metta Integration

The GAMMA framework is fully integrated with Metta's training system:

```python
from metta.alignment.integration import GAMMALogger

# Add to trainer
gamma_logger = GAMMALogger(
    num_agents=16,
    epoch_interval=10,  # Compute every 10 epochs to minimize overhead
    alpha=0.1,
    enabled=True
)
trainer.add_component(gamma_logger)
```

See `metta/alignment/integration/README.md` for detailed integration documentation.

### Performance Impact

GAMMA metrics are designed to be lightweight and suitable for online monitoring:

**Computational Complexity**:
- Per-agent metrics: O(T) where T is episode length
- Collective GAMMA: O(N) where N is number of agents
- Total per evaluation: O(T·N·d) where d is spatial dimension

**Typical Overhead**:
- Trajectory collection: ~1-2% overhead (just storing positions)
- Metric computation: <100ms for 16 agents over 1000 timesteps
- Negligible impact on training throughput

**Recommendations**:
- **During training**: Set `epoch_interval=10` or higher to compute every 10-100 epochs
- **For evaluation**: Set `epoch_interval=1` for detailed monitoring
- **Production**: Can disable with `enabled=False` and enable only for evaluation runs

**Memory Usage**:
- Trajectory storage: ~8 bytes × T × N × d (e.g., 128 KB for 16 agents, 1000 steps, 2D)
- Cleared after each evaluation, so no accumulation

The implementation is optimized for minimal impact on training speed while providing valuable alignment insights.

## Examples

Three example scripts demonstrate different aspects of GAMMA:

### 1. Basic Metrics Demo (Simulated Agents)

Learn how each metric works with three agent types: aligned, noisy, and adversarial.

```bash
uv run python metta/alignment/examples/simple_alignment_demo.py
```

**Shows**: Individual metric computation, misalignment detection, collective GAMMA

### 2. Real MettaGrid Integration (Random Agents)

Test GAMMA with actual MettaGrid environment. Agents take random actions, demonstrating low alignment (GAMMA ≈ 0.001).

```bash
# Default: 4 agents, 50 steps, 10x10 map
uv run python metta/alignment/examples/test_with_real_mettagrid.py

# Custom parameters
uv run python metta/alignment/examples/test_with_real_mettagrid.py --num_agents 8 --num_steps 100

# Larger swarm
uv run python metta/alignment/examples/test_with_real_mettagrid.py --num_agents 16 --num_steps 200 --map_size 20

# See all options
uv run python metta/alignment/examples/test_with_real_mettagrid.py --help
```

**Generates**: 4-panel visualization with trajectories, velocities, metrics, and summary

### 3. High Alignment Demo (Goal-Directed Agents)

See what well-aligned agents look like. Agents move efficiently toward goals, achieving D=0.82, E=0.87 (171x better than random).

```bash
# Default: 4 agents, 100 steps
uv run python metta/alignment/examples/demo_high_alignment.py

# Larger swarm
uv run python metta/alignment/examples/demo_high_alignment.py --num_agents 8 --num_steps 150

# See all options
uv run python metta/alignment/examples/demo_high_alignment.py --help
```

**Shows**: Convergence to goals, high metric scores, comparison with random agents

### Comparison

| Demo | GAMMA | D | E | A | Use Case |
|------|-------|---|---|---|----------|
| Random agents | 0.001 | 0.04 | 0.2 | 0.01 | Baseline / integration test |
| Goal-directed | 0.171 | 0.82 | 0.87 | 0.51 | Target for trained agents |
| Trained (expected) | >0.7 | >0.8 | >0.7 | >0.8 | Production goal |

## Future Extensions

- Formation task interface
- Coverage task interface
- Advanced detectors (CEEI, Energy-Progress Hysteresis, Non-Intentionality)
- Baseline library construction system
- Online monitoring and alerting dashboard
