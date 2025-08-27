# Metta AI Test Audit - Phase 2: Specific Removal & Consolidation Recommendations

## Executive Summary

This document provides detailed, actionable recommendations for reducing test cruft while maintaining meaningful coverage. Based on analysis of 130+ test files, we've identified specific files for removal, consolidation opportunities, and an implementation plan that could reduce the test suite by **25-30%** while improving focus on core functionality.

## Immediate Removal Candidates (High Impact, Low Risk)

### 1. Meta-Test Files - Delete Completely

#### `tests/test_no_xcxc.py` (29 lines) - **DELETE**
**Reason:** Scans entire codebase for "xcxc" debug strings
**Replacement:** Pre-commit hook or git hook
**Impact:** No loss of functionality testing
```bash
# Replacement command for .git/hooks/pre-commit:
grep -r "xcxc" --exclude-dir=".git" --exclude-dir=".venv" . && exit 1 || exit 0
```

#### `tests/test_tensors_have_dtype.py` (169 lines) - **DELETE** 
**Reason:** AST parser ensuring `torch.tensor()` has explicit dtype
**Replacement:** Ruff linting rule or flake8 plugin
**Impact:** Better handled by static analysis
```toml
# Add to ruff.toml:
[tool.ruff.lint.flake8-comprehensions]
allow-dict-calls-with-keyword-arguments = false
```

### 2. Commented-Out Files - Remove Immediately

#### `tests/sweep/test_sweep_init.py` (323 lines) - **DELETE**
**Status:** Entire file commented out with `# TODO(axel) #dehydration`
**Action:** Remove file completely or restore if needed
**Risk:** Zero - file is non-functional

#### `tests/sweep/test_integration_sweep_centralized.py` (partial) - **DELETE**
**Status:** Large portions commented out
**Action:** Remove commented portions, keep only active tests

#### `tests/sweep/test_sweep_config_loading.py` (partial) - **DELETE** 
**Status:** Mostly commented out
**Action:** Clean up or remove entirely

## Map Generation Test Consolidation (Major Opportunity)

### Current State Analysis
**Location:** `tests/map/scenes/` (20+ files, ~300 total lines)
**Pattern:** Most tests follow identical structure testing different scene generators

### Consolidation Strategy

#### Create Single Test File: `tests/map/scenes/test_scene_generators_consolidated.py`

**Replace these 8+ similar files:**
- `test_random.py` (29 lines) → **DELETE**
- `test_wfc.py` (21 lines) → **DELETE** 
- `test_convchain.py` (24 lines) → **DELETE**
- `test_nop.py` (9 lines) → **DELETE**
- `test_mirror.py` → **DELETE**
- `test_remove_agents.py` → **DELETE**
- Several others with minimal differentiation

**Consolidated Test Structure:**
```python
@pytest.mark.parametrize("generator_class,params,expected_props", [
    (Random, {"objects": {"altar": 3}}, lambda scene: (scene.grid == "altar").sum() == 3),
    (WFC, {"pattern": "...\n###\n..."}, lambda scene: (scene.grid == "wall").sum() > 0),
    (ConvChain, {"pattern": "##..\n#...\n###"}, lambda scene: (scene.grid == "empty").sum() > 0),
    # ... other generators
])
def test_scene_generator_basic_functionality(generator_class, params, expected_props):
    scene = render_scene(generator_class.factory(generator_class.Params(**params)), (10, 10))
    assert expected_props(scene)
```

**Estimated Reduction:** 200+ lines → 50 lines (75% reduction)

#### Keep Complex Tests Separate
**Preserve:** 
- `test_ascii.py` (17 lines) - Tests file loading
- `test_maze.py` (43 lines) - Tests algorithmic correctness 
- `test_room_grid.py` (100 lines) - Tests complex layout logic

## Distribution Testing Rationalization

### Current Problem
**Files:** `tests/map/random/test_float.py` (255 lines), `test_int.py` (140 lines)
**Issue:** Testing mathematical properties guaranteed by underlying libraries

### Specific Reductions

#### `test_float.py` - Reduce from 255 lines to ~80 lines
**Remove these test classes:**
```python
# DELETE - Tests mathematical guarantees
class TestFloatConstantDistribution:  # 40 lines testing that constants return constants
class TestFloatUniformDistribution:   # 85 lines testing uniform distribution bounds

# KEEP - Tests integration with Pydantic
class TestFloatDistributionTypes:     # 104 lines - keep for config validation
```

#### `test_int.py` - Reduce from 140 lines to ~50 lines  
**Similar approach:** Remove mathematical property testing, keep integration tests

**Estimated Total Reduction:** 395 lines → 130 lines (67% reduction)

## Buffer/Performance Test Relocations

### `mettagrid/tests/test_buffer_sharing_regression.py` (212 lines)
**Action:** Move to integration test suite, not unit tests
**New Location:** `tests/integration/test_performance_regression.py`
**Reason:** Performance tests shouldn't block development workflows

### `tests/rl/test_mps.py`
**Action:** Conditional execution only on Apple Silicon
**Reason:** Platform-specific tests cause CI failures

## Sweep/Protein Test Cleanup

### Files to Clean Up
1. **`tests/sweep/test_protein_*.py`** (5 files) - Review for commented code
2. **`tests/sweep/test_integration_sweep_pipeline.py`** - Remove unused portions
3. **`tests/sweep/test_wandb_utils.py`** - Verify still relevant to current wandb usage

## Critical Tests to Preserve (Do Not Touch)

### Core Functionality Tests
1. **`tests/test_programmatic_env_creation.py`** (233 lines) - Critical for env creation
2. **`mettagrid/tests/test_gym_env.py`** - Gymnasium integration
3. **`agent/tests/test_policy_cache.py`** - Policy management 
4. **`mettagrid/tests/test_pufferlib_integration.py`** - Core RL integration

### Environment Core Tests  
1. **`mettagrid/tests/test_global_observations.py`**
2. **`mettagrid/tests/test_diversity.py`**
3. **`mettagrid/tests/test_stats_writer.py`**

### Training/RL Core Tests
1. **`tests/rl/test_losses.py`**
2. **`tests/rl/test_kickstarter.py`** 
3. **`tests/rl/test_hyperparameter_scheduler.py`**

## Implementation Plan

### Phase 1: Safe Deletions (Week 1)
```bash
# Remove meta-tests
rm tests/test_no_xcxc.py
rm tests/test_tensors_have_dtype.py

# Remove commented-out files
rm tests/sweep/test_sweep_init.py
# Clean up other sweep files with comments

# Remove trivial scene tests
rm tests/map/scenes/test_nop.py
rm tests/map/scenes/test_random.py
rm tests/map/scenes/test_wfc.py
rm tests/map/scenes/test_convchain.py
# ... continue with similar files
```

### Phase 2: Consolidations (Week 2) 
```bash
# Create consolidated scene generator tests
# Reduce distribution testing to essentials
# Move performance tests to integration suite
```

### Phase 3: Validation (Week 3)
```bash
# Run full test suite to ensure no regressions
# Update CI/CD pipelines if needed
# Document remaining test structure
```

## Expected Impact

### Quantitative Benefits
- **Test file count:** 130+ files → ~95 files (27% reduction)
- **Total test lines:** ~8,000 → ~5,500 lines (31% reduction) 
- **CI execution time:** Estimated 15-20% reduction
- **Maintenance burden:** Significant reduction in trivial test updates

### Qualitative Benefits
- **Improved test focus:** Emphasizes behavioral testing over property testing
- **Reduced noise:** Fewer false positives from environment-dependent tests
- **Cleaner codebase:** Removes commented-out and meta-testing code
- **Better developer experience:** Faster feedback from more relevant tests

## Risk Mitigation

### Low-Risk Actions
- Deleting commented-out code
- Removing meta-tests with clear replacements
- Consolidating nearly-identical test files

### Medium-Risk Actions  
- Reducing distribution testing (verify no custom logic tested)
- Moving performance tests (ensure integration tests run in CI)

### Validation Strategy
1. **Before any deletions:** Full test run to establish baseline
2. **After each phase:** Complete test suite execution  
3. **Integration testing:** Verify end-to-end workflows still work
4. **Performance monitoring:** Ensure no critical performance regressions

## Next Steps

1. **Review and approve** specific files marked for deletion
2. **Prioritize** which consolidations provide highest value
3. **Create backup branch** before beginning deletions
4. **Execute Phase 1** safe deletions
5. **Measure impact** and proceed with subsequent phases

This plan balances aggressive cleanup with risk management, focusing on eliminating clear waste while preserving tests that verify core system behavior and integration points.