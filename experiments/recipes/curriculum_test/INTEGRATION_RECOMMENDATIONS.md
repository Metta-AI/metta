# Task Dependency Simulator - Integration Recommendations

## Goal
Make the task dependency simulator more comparable to real training by using the same infrastructure patterns.

## Current Architecture Issues

### 1. Direct Curriculum Manipulation
**Problem**: The simulator calls `curriculum.get_task()` and `curriculum.update_task_performance()` directly, bypassing the `CurriculumEnv` wrapper that real training uses.

**Impact**:
- Missing CurriculumEnv's stats emission patterns
- Missing automatic task management
- Stats don't flow through the same pipeline

### 2. Manual Stats Collection
**Problem**: Stats are collected manually in dictionaries rather than through info dicts.

**Impact**:
- Stats format doesn't match real training
- Can't use standard `accumulate_rollout_stats()` and `process_training_stats()`
- Manual formatting is error-prone

### 3. No Environment Abstraction
**Problem**: Simulator doesn't implement an environment interface.

**Impact**:
- Can't wrap with `CurriculumEnv`
- Can't test vectorization patterns
- Missing episode termination signals

## Recommended Refactoring

### Option A: Create a Mock Environment (Recommended)

Create a minimal PufferEnv-compatible environment that CurriculumEnv can wrap:

```python
class TaskDependencyEnv(PufferEnv):
    """Mock environment for task dependency simulation."""

    def __init__(self, simulator: TaskDependencySimulator, task_config: MettaGridConfig):
        self.simulator = simulator
        self.task_config = task_config
        self.task_id = None
        self.steps = 0
        self.max_steps_per_episode = 50

    def reset(self):
        self.steps = 0
        # Extract task ID from config
        self.task_id = int(self.task_config.label.split('_')[-1])
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        self.steps += 1

        # Sample reward from simulator
        reward = self.simulator.sample_task(self.task_id)

        # Episode done after max_steps
        done = self.steps >= self.max_steps_per_episode

        # Emit stats via info dict (matching real environments)
        info = {
            "task_dependency": {
                "performance": self.simulator.P[self.task_id].item(),
                "task_id": self.task_id,
            }
        }

        return (
            np.zeros((1,), dtype=np.float32),  # obs
            reward,  # reward
            done,  # terminal
            False,  # truncated
            info  # info dict
        )

    def get_episode_rewards(self):
        """Return episode rewards for CurriculumEnv."""
        return np.array([self.simulator.P[self.task_id].item()])

    def set_mg_config(self, config: MettaGridConfig):
        """Allow CurriculumEnv to update task config."""
        self.task_config = config
```

Then use it with CurriculumEnv:

```python
def simulate_with_curriculum_env(
    simulator: TaskDependencySimulator,
    curriculum: Curriculum,
    num_steps: int = 50000,
):
    """Simulate using CurriculumEnv wrapper (matches real training)."""

    # Create base environment
    initial_task = curriculum.get_task()
    env = TaskDependencyEnv(simulator, initial_task.get_env_cfg())

    # Wrap with CurriculumEnv (just like real training!)
    curriculum_env = CurriculumEnv(env, curriculum)

    # Simulate episodes
    stats = defaultdict(list)
    obs, info = curriculum_env.reset()

    for step in range(num_steps):
        # Take random action (doesn't matter for this simulation)
        action = 0

        obs, reward, terminal, truncated, info = curriculum_env.step(action)

        # Accumulate stats from info dicts (matching real training!)
        accumulate_rollout_stats([info], stats)

        if terminal or truncated:
            obs, info = curriculum_env.reset()

    # Process stats (matching real training!)
    processed = process_training_stats(
        raw_stats=stats,
        losses_stats={},
        experience=None,
        trainer_config=None,
    )

    return processed
```

### Option B: Use Stats Processing Infrastructure

Keep the current simulator but use the real stats processing pipeline:

```python
def simulate_with_stats_pipeline(
    simulator: TaskDependencySimulator,
    curriculum: Curriculum,
    num_epochs: int = 500,
    samples_per_epoch: int = 100,
):
    """Simulate using real stats processing pipeline."""
    from collections import defaultdict
    from metta.rl.stats import accumulate_rollout_stats, process_training_stats

    for epoch in range(num_epochs):
        # Collect info dicts (like vectorized environments would)
        infos_batch = []

        for _ in range(samples_per_epoch):
            task = curriculum.get_task()
            task_id = task._task_id % simulator.num_tasks
            reward = simulator.sample_task(task_id)

            # Create info dict (matching real environment format)
            info = {
                "task_dependency": {
                    "performance": simulator.P[task_id].item(),
                    "task_id": task_id,
                    "reward": reward,
                },
                "curriculum_stats": {
                    "pool_occupancy_gini": curriculum._algorithm.get_base_stats().get("pool_occupancy_gini", 0),
                    "pool_lp_gini": curriculum._algorithm.get_base_stats().get("pool_lp_gini", 0),
                }
            }

            # Simulate episode completion
            task.complete(reward)
            curriculum.update_task_performance(task._task_id, reward)

            infos_batch.append(info)

        # Use real stats accumulation
        rollout_stats = defaultdict(list)
        accumulate_rollout_stats(infos_batch, rollout_stats)

        # Use real stats processing
        processed_stats = process_training_stats(
            raw_stats=rollout_stats,
            losses_stats={},
            experience=MockExperience(epoch),
            trainer_config=MockTrainerConfig(),
        )

        # Log using standard format
        if wandb_run:
            wandb_run.log(processed_stats, step=epoch * samples_per_epoch)
```

### Option C: Simulate Vectorized Environments

Most realistic - simulate what happens with multiple parallel environments:

```python
def simulate_vectorized(
    simulator: TaskDependencySimulator,
    curriculum: Curriculum,
    num_envs: int = 8,
    num_steps: int = 50000,
):
    """Simulate vectorized environment behavior."""

    # Create multiple curriculum instances (like real training)
    # Note: In real training, they share the same curriculum via shared memory
    envs = [
        CurriculumEnv(
            TaskDependencyEnv(simulator, curriculum.get_task().get_env_cfg()),
            curriculum
        )
        for _ in range(num_envs)
    ]

    # Vectorized reset
    obs_list = []
    info_list = []
    for env in envs:
        obs, info = env.reset()
        obs_list.append(obs)
        info_list.append(info)

    # Rollout loop (matching real training)
    stats = defaultdict(list)

    for step in range(num_steps):
        # Vectorized step
        info_batch = []
        for env in envs:
            obs, reward, terminal, truncated, info = env.step(0)
            info_batch.append(info)

            if terminal or truncated:
                env.reset()

        # Accumulate stats from all environments
        accumulate_rollout_stats(info_batch, stats)

        # Process stats periodically (like epoch boundaries)
        if step % 1000 == 0:
            processed = process_training_stats(
                raw_stats=stats,
                losses_stats={},
                experience=MockExperience(step),
                trainer_config=MockTrainerConfig(),
            )

            # Log to wandb
            wandb.log(processed, step=step)

            # Clear stats for next epoch
            stats.clear()
```

## Benefits of Each Approach

### Option A (Mock Environment)
**Pros:**
- Most realistic - uses exact same code paths as real training
- Tests CurriculumEnv wrapper directly
- Stats automatically formatted correctly
- Easy to verify curriculum behavior matches real training

**Cons:**
- More code to write
- Requires PufferEnv interface implementation

### Option B (Stats Pipeline Only)
**Pros:**
- Minimal changes to existing code
- Uses real stats processing
- Stats format guaranteed to match

**Cons:**
- Still bypasses CurriculumEnv wrapper
- Doesn't test environment integration
- Manual info dict construction

### Option C (Vectorized Simulation)
**Pros:**
- Most realistic for production training
- Tests parallel curriculum access patterns
- Matches actual vectorized training behavior
- Can test shared memory backend

**Cons:**
- Most complex to implement
- May be overkill for simple testing
- Slower execution

## Recommendation

**Start with Option B** (Stats Pipeline) as immediate improvement:
1. Low effort, high value
2. Makes metrics immediately comparable
3. Can refactor to Option A later if needed

**Consider Option A** for comprehensive testing:
1. When you need to verify CurriculumEnv behavior
2. When testing curriculum changes
3. For regression testing

**Use Option C** for production validation:
1. Before deploying curriculum changes
2. For performance testing
3. For testing shared memory backend

## Implementation Priority

1. **Immediate** (Option B): Use `accumulate_rollout_stats` and `process_training_stats`
2. **Short-term** (Option A): Create `TaskDependencyEnv` for full integration
3. **Long-term** (Option C): Add vectorized simulation for comprehensive testing

## Example: Quick Win with Option B

Here's a minimal change to make your current simulator use the real stats pipeline:

```python
# At the top of simulate_task_dependencies()
from collections import defaultdict
from metta.rl.stats import accumulate_rollout_stats, process_training_stats

# In the epoch loop, replace manual stats collection with:
for epoch in range(num_epochs):
    infos_batch = []

    for _ in range(samples_per_epoch):
        task = curriculum.get_task()
        task_id = task._task_id % num_tasks
        reward = simulator.sample_task(task_id)
        task.complete(reward)
        curriculum.update_task_performance(task._task_id, reward)

        # Emit stats via info dict
        info = {
            "task_dependency/performance": simulator.P[task_id].item(),
            "task_dependency/reward": reward,
        }
        # Add curriculum stats from curriculum.stats()
        for key, value in curriculum.stats().items():
            info[f"curriculum_stats/{key}"] = value

        infos_batch.append(info)

    # Use real stats pipeline
    rollout_stats = defaultdict(list)
    accumulate_rollout_stats(infos_batch, rollout_stats)

    # Format for wandb using real infrastructure
    # (process_training_stats would need mock Experience/TrainerConfig)
    # For now, use the mean stats logic from process_training_stats:
    processed_stats = {}
    for k, v in rollout_stats.items():
        if "per_label_samples" in k or "tracked_task_completions" in k:
            processed_stats[k] = np.sum(v)
        else:
            processed_stats[k] = np.mean(v)

    # Log with standard metric naming
    wandb.log({
        "metric/agent_step": epoch * samples_per_epoch,
        "metric/epoch": epoch,
        **{f"env_{k}": v for k, v in processed_stats.items()}
    }, step=epoch * samples_per_epoch)
```

This small change makes your stats flow through the same pipeline as real training!

