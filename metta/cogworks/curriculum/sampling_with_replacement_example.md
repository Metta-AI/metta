# Curriculum Sampling with Replacement in Vectorized Environments

## Overview

The curriculum samples tasks **with replacement** - multiple environments can run the same task_id simultaneously. This is a key feature that enables efficient curriculum learning at scale.

## How It Works

### Task Pool is Persistent

```python
# Task pool contains 1000 tasks
curriculum._tasks = {
    8392: CurriculumTask(8392, env_cfg, ...),
    1234: CurriculumTask(1234, env_cfg, ...),
    5678: CurriculumTask(5678, env_cfg, ...),
    # ... 997 more tasks
}

# When an environment calls get_task():
task = curriculum.get_task()
# → Returns reference to task from pool
# → Task REMAINS in pool
# → Can be sampled again immediately
```

### Multiple Environments, Same Task

```python
# 256 vectorized environments
for i in range(256):
    task = curriculum.get_task()  # Each calls independently

# Possible outcome:
# Env 0:   task_id = 8392  (LP = 0.25)
# Env 1:   task_id = 8392  (LP = 0.25)  ← Same task!
# Env 2:   task_id = 1234  (LP = 0.18)
# Env 3:   task_id = 8392  (LP = 0.25)  ← Same task again!
# Env 4:   task_id = 5678  (LP = 0.12)
# ...
# Result: task_8392 appears in ~45 environments (high LP)
```

## Probability Distribution

Tasks are sampled with probabilities proportional to their learning progress scores:

```python
# Example with 5 tasks
task_scores = {
    8392: 0.25,  # High LP
    1234: 0.18,  # Medium LP
    5678: 0.12,  # Medium LP
    9999: 0.05,  # Low LP
    7777: 0.02,  # Low LP
}

total_score = 0.62

# Probabilities
P(8392) = 0.25 / 0.62 = 0.403 (40.3%)
P(1234) = 0.18 / 0.62 = 0.290 (29.0%)
P(5678) = 0.12 / 0.62 = 0.194 (19.4%)
P(9999) = 0.05 / 0.62 = 0.081 (8.1%)
P(7777) = 0.02 / 0.62 = 0.032 (3.2%)

# With 256 environments, expected distribution:
# task_8392: ~103 environments
# task_1234: ~74 environments
# task_5678: ~50 environments
# task_9999: ~21 environments
# task_7777: ~8 environments
```

## Benefits

### 1. Natural Task Distribution

High-LP tasks automatically dominate training:

```python
# Epoch 1000
Active environments: 256
Task pool: 1000 tasks

Distribution across environments:
  Top 10 LP tasks:  ~150 envs (58%)  ← Focus on learning frontier
  Next 40 LP tasks: ~80 envs (31%)   ← Secondary learning
  Remaining 950:    ~26 envs (11%)   ← Exploration
```

### 2. Efficient Batch Composition

Each training batch contains optimal task mix:

```python
# Training batch
batch_size = 256 envs × 24 agents = 6,144 agents

Task composition:
  task_8392 (high LP): 2,472 agents (40%) ← Efficient learning
  task_1234 (high LP): 1,786 agents (29%)
  task_5678 (med LP):  1,194 agents (19%)
  Others (exploration): 692 agents (12%)

# Agent gets most gradient updates on tasks where it's actively learning!
```

### 3. Automatic Load Balancing

As agent improves, distribution shifts automatically:

```python
# Initially: task_8392 has high LP
# → Appears in 103/256 environments (40%)

# After 1000 episodes: agent masters task_8392
# → LP drops to 0.05
# → Now appears in ~21/256 environments (8%)

# Meanwhile: task_5678 becomes learnable
# → LP increases from 0.12 to 0.28
# → Now appears in ~115/256 environments (45%)

# No manual reweighting needed - emerges from LP signal!
```

## Comparison: With vs Without Replacement

### Without Replacement (NOT how we work)

```python
# Hypothetical: each task can only be in one environment
task_pool = [task_1, task_2, ..., task_1000]
random.shuffle(task_pool)

env_0.task = task_pool[0]    # task_8392
env_1.task = task_pool[1]    # task_1234 (can't get 8392)
env_2.task = task_pool[2]    # task_5678 (can't get 8392 or 1234)
...

# Problems:
# 1. High-LP tasks only get 1/256 of compute (inefficient!)
# 2. Low-LP tasks forced into 1/256 of envs (wasted compute!)
# 3. Can't have more envs than tasks
# 4. No automatic focus on learning frontier
```

### With Replacement (How we actually work)

```python
# Each environment independently samples
env_0.task = curriculum.get_task()  # task_8392 (LP=0.25, P=40%)
env_1.task = curriculum.get_task()  # task_8392 (same! LP is high)
env_2.task = curriculum.get_task()  # task_1234 (LP=0.18, P=29%)
...

# Benefits:
# 1. High-LP tasks get proportional compute (efficient!)
# 2. Low-LP tasks rarely sampled (minimal waste!)
# 3. Can have unlimited environments
# 4. Automatic focus emerges from independent sampling
```

## Performance Updates

When multiple environments run the same task:

```python
# Time = 0: task_8392 in 103 environments
# All environments training on it simultaneously

# Time = 50: 15 environments complete episode with task_8392
for env_id in [3, 7, 12, 19, ...]:  # 15 completions
    curriculum.update_task_performance(
        task_id=8392,
        score=episode_scores[env_id]
    )
    # Each updates shared task tracker
    # Fast EMA and Slow EMA updated 15 times
    # LP score recalculated based on new performance

# Time = 51: New environment selects task
new_task = curriculum.get_task()
# → Sees updated LP score for task_8392
# → Probability adjusted based on latest 15 completions
# → If agent improved: LP increases → more likely to select
# → If agent plateaued: LP decreases → less likely to select
```

## Shared Memory Coordination

With multiprocessing vectorization:

```python
# 8 worker processes, 32 environments each = 256 total

# Worker 0, Env 5 completes task_8392
Worker_0.curriculum.update_task_performance(8392, score=0.65)
└─> SharedMemoryBackend writes to shared array
    └─> shared_memory[task_8392].reward_ema = 0.65
    └─> shared_memory[task_8392].lp_score = 0.22

# Worker 3, Env 127 selects next task (microseconds later)
Worker_3.curriculum.get_task()
└─> SharedMemoryBackend reads from shared array
    └─> task_8392.lp_score = 0.22  ← Sees update immediately!
    └─> Probability calculation uses latest data
    └─> All workers coordinate through shared memory

# Result: All 256 environments see consistent task pool state
```

## Example Scenario

### Setup

```python
config = CurriculumConfig(
    task_generator=BucketedTaskGenerator.Config(
        buckets={"game.num_plants": [5, 10, 15, 20, 25]}
    ),
    algorithm_config=LearningProgressConfig(
        num_active_tasks=1000,
        use_shared_memory=True,
    ),
)

# 256 vectorized environments
env = VectorizedTrainingEnvironment(
    num_envs=256,
    curriculum=curriculum,
)
```

### Training Step

```python
# Single step across all environments
obs, rewards, dones, truncs, infos = env.step(actions)

# Behind the scenes:
# - Env 0 completes task_8392 → updates shared memory
# - Env 7 completes task_8392 → updates shared memory (same task!)
# - Env 12 completes task_1234 → updates shared memory
# - Env 19 starts new episode → samples from updated pool
# - Env 23 starts new episode → sees all recent updates
# ...

# Result:
# - ~15 environments complete episodes
# - ~15 environments select new tasks
# - All selections use latest performance data
# - High-LP tasks automatically selected more often
```

### Distribution Over Time

```python
# Epoch 0: Uniform exploration
task_8392: 1 environment
task_1234: 1 environment
...all tasks equally likely...

# Epoch 100: Learning signal emerges
task_8392 (high LP): 45 environments (17%)
task_1234 (med LP): 28 environments (11%)
task_5678 (low LP): 5 environments (2%)

# Epoch 500: Focused learning
task_8392 (peak LP): 103 environments (40%)
task_1234 (high LP): 74 environments (29%)
task_5678 (mastered): 8 environments (3%)

# Epoch 1000: Agent masters task_8392
task_8392 (mastered): 21 environments (8%)
task_5678 (now high LP): 115 environments (45%)
task_7777 (new frontier): 67 environments (26%)

# The distribution automatically tracks the learning frontier!
```

## Summary

**Sampling with replacement is essential for efficient curriculum learning:**

1. ✅ **Already implemented** - no code changes needed
2. ✅ **Multiple environments** can run same task simultaneously
3. ✅ **High-LP tasks** naturally dominate training batches
4. ✅ **Low-LP tasks** minimally waste compute
5. ✅ **Automatic rebalancing** as agent learns
6. ✅ **Shared memory** coordinates across workers
7. ✅ **Scales to unlimited** environments

This design enables the curriculum to automatically focus training on the learning frontier while efficiently using all available compute! 🎯

