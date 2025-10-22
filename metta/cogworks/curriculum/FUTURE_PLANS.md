# Curriculum System: Future Plans

This document outlines potential future directions for the curriculum learning system. Each section is a placeholder for more detailed design discussions.

---

## 1. Agora: Independent Curriculum Package

**Goal**: Extract the centralized curriculum system into a standalone package called `agora` (from the Greek ἀγορά, meaning "a central place to exchange information").

**Motivation**:
- Enable use across different RL frameworks beyond Metta
- Create a general-purpose curriculum learning library
- Separate concerns: task generation, performance tracking, and sampling strategy

**Key Components to Extract**:
- `Curriculum` core with task generation and lifecycle management
- `TaskTracker` with shared memory backend for multi-process coordination
- Learning progress algorithms (bidirectional, basic)
- Task generator abstractions (single, bucketed, set)

**Design Considerations**:
- Minimal dependencies (numpy, pydantic)
- Framework-agnostic interfaces
- Easy integration hooks for custom environments
- Preserve checkpointing and state serialization

---

## 2. Two-Pool Exploration/Exploitation System

**Goal**: Replace single task pool with dual-pool architecture that dynamically balances exploration and exploitation.

**Architecture**:
```
┌─────────────────┐         ┌──────────────────┐
│  Explore Pool   │  -----> │  Exploit Pool    │
│  (New tasks)    │ promote │  (Proven tasks)  │
└─────────────────┘         └──────────────────┘
```

**Mechanism**:
- **Explore Pool**: New/untested tasks, allocated for initial evaluation
- **Exploit Pool**: Tasks proven to have high learning progress
- **Promotion**: Task with sufficient samples (>= `min_samples`) is compared to exploit pool
  - If `LP_score(explore_task) > min(LP_score(exploit_pool))`, promote and replace lowest scorer
  - Track promotion success rate across epoch
- **Dynamic Allocation**: Adjust explore/exploit sampling ratio based on promotion success
  - High promotion rate → allocate more to exploration (finding better tasks)
  - Low promotion rate → allocate more to exploitation (mining known good tasks)

**Parameters**:
- `min_promotion_samples`: Minimum completions before promotion eligibility
- `promotion_threshold`: Percentile rank needed to promote
- `allocation_alpha`: EMA smoothing for allocation adjustment
- `min_explore_ratio`, `max_explore_ratio`: Bounds on resource allocation

---

## 3. Task Dependency Toy Model

**Goal**: Model task relationships where mastering prerequisite tasks gates learning on successor tasks.

**Structure**:
```
Task A (basic) --> Task B (intermediate) --> Task C (advanced)
```

**Learning Dynamics**:
- Task performance depends on both:
  1. Direct training on that task
  2. Mastery level of prerequisite tasks
- Agent cannot learn Task C effectively until sufficient mastery of Task B
- Curriculum must discover dependency structure through learning progress signals

**Toy Environment**:
- Simple gridworld with hierarchical skills (navigate → open door → collect item)
- Performance on complex task improves only after subtasks mastered
- Ground truth dependency graph for validation

**Research Questions**:
- Can learning progress algorithms discover dependencies implicitly?
- Should curriculum track and model dependencies explicitly?
- How does two-pool system interact with task dependencies?
- What promotion strategies work best for hierarchical curricula?

**Evaluation Metrics**:
- Time to discover optimal task ordering
- Sample efficiency vs. random/fixed curriculum
- Robustness to incorrect dependency assumptions

---

## Open Questions

- How does the two-pool system interact with existing bidirectional learning progress?
- Should Agora include built-in two-pool support or keep it as an extension?
- Can we learn task dependencies from replay data or does it require explicit modeling?
- What's the right granularity for dependency modeling (task-level, generator-level, skill-level)?

---

## Next Steps

1. **Prototype Agora extraction**: Identify all framework-specific dependencies in current curriculum code
2. **Two-pool simulation**: Implement simplified version with synthetic LP scores to test allocation dynamics
3. **Toy model design**: Create minimal gridworld environment with known hierarchical structure

