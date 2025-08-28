# Phase 14 Recovery Plan: Minimal Critical Functionality Restoration

## Objective

Restore only the **essential** policy management capabilities needed for production workflows while maintaining the simplified architecture achieved in the policy cull.

## MVP Recovery Strategy

Focus on the **minimum viable functionality** rather than feature parity with the original system.

### Phase 1: Core Policy Access (Week 1)

**Goal:** Basic multi-policy evaluation and selection

#### 1.1 Enhance CheckpointManager
- Add `find_best_checkpoint(metric: str)` method
- Implement basic policy selection strategies: "latest", "best_score"
- Add simple metadata-based filtering

#### 1.2 Restore Multi-Policy Evaluation
- Fix `metta/tools/sim.py` to handle multiple policies
- Enable batch evaluation with basic result aggregation
- Support `file://` URIs with multiple checkpoint selection

**Success Criteria:**
- Can evaluate multiple checkpoints from a training run
- Can automatically select best performing checkpoint
- Can run evaluation suites against selected policies

### Phase 2: Remote Policy Access (Week 2)

**Goal:** Basic wandb policy loading

#### 2.1 Minimal Wandb Integration
- Add `wandb://` URI support to CheckpointManager
- Implement simple artifact download and caching
- Basic policy loading from wandb artifacts

#### 2.2 URI Resolution Infrastructure
- Simple URI dispatcher for `file://` and `wandb://` schemes
- Basic error handling and fallback mechanisms

**Success Criteria:**
- Can load policies from wandb artifacts
- Can evaluate policies stored in wandb
- Simple URI format: `wandb://run/run_name` or `wandb://entity/project/artifact:version`

### Phase 3: Evaluation Infrastructure (Week 3)

**Goal:** Policy performance tracking

#### 3.1 Stats Integration
- Connect evaluation results to stats database
- Basic policy performance queries in EvalStatsDB
- Simple policy comparison reports

#### 3.2 Result Aggregation
- Basic policy ranking and comparison
- Simple evaluation result exports
- Integration with existing analysis tools

**Success Criteria:**
- Can track policy performance across evaluations
- Can compare multiple policies in analysis reports
- Can export evaluation results for external analysis

## Non-Goals (Explicitly Out of Scope)

- Complex policy artifact lifecycle management
- Advanced caching and memory optimization
- Sophisticated policy versioning and lineage
- Legacy PolicyRecord/PolicyStore compatibility
- Advanced wandb integration features
- Multi-backend URI support beyond wandb
- Complex policy selection algorithms
- Advanced metadata and validation systems

## Implementation Approach

### Design Principles
1. **Extend, don't replace** - Build on existing CheckpointManager
2. **Simple interfaces** - Minimal API surface area
3. **Fail fast** - Clear error messages for unsupported features
4. **Optional dependencies** - Graceful degradation when wandb unavailable

### Key Files to Modify

#### Core Changes (Required)
- `metta/rl/checkpoint_manager.py` - Add selection and wandb support
- `metta/tools/sim.py` - Restore multi-policy evaluation
- `metta/eval/eval_service.py` - Connect to stats database

#### Supporting Changes (Minimal)
- `metta/eval/eval_stats_db.py` - Ensure policy integration works
- Add simple URI resolution utility module
- Basic wandb policy loader utility

### Risk Mitigation

- **Backwards compatibility** - Existing CheckpointManager API unchanged
- **Optional features** - All new functionality behind feature flags
- **Testing** - Each phase independently testable
- **Rollback ready** - Changes isolated to specific modules

## Success Metrics

### Phase 1 Success
- `uv run ./tools/run.py experiments.recipes.arena.evaluate policy_uri=file://./train_dir/test/checkpoints selector=best_score`
- Can evaluate top 3 checkpoints from a training run

### Phase 2 Success  
- `uv run ./tools/run.py experiments.recipes.arena.evaluate policy_uri=wandb://run/my_training_run`
- Can load and evaluate policies from wandb

### Phase 3 Success
- Can generate policy comparison reports
- Can track evaluation results in stats database

## Timeline

- **Week 1:** Core policy selection and multi-policy evaluation
- **Week 2:** Basic wandb integration and URI resolution  
- **Week 3:** Stats integration and evaluation infrastructure

**Total Effort:** 3 weeks focused development

## Decision Points

### After Phase 1
- Evaluate if multi-policy evaluation meets workflow needs
- Decide if wandb integration is required or can be deferred

### After Phase 2  
- Assess if basic wandb support is sufficient
- Determine if additional URI schemes are needed

### After Phase 3
- Review if stats integration provides needed analytics
- Plan any additional evaluation features

## Success Definition

The recovery is complete when teams can:
1. **Select best policies** from training runs automatically
2. **Evaluate multiple policies** in batch operations  
3. **Load policies from wandb** for evaluation
4. **Compare policy performance** in analysis reports

All other original functionality remains intentionally out of scope to maintain architectural simplicity.