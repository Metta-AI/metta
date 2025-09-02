# Merge Conflicts Analysis: av-ks-22 → main

**Branch:** `av-ks-22`  
**Target:** `main`  
**Merge Date:** September 2, 2025  
**Total Commits to Merge:** 76 commits from main  
**Conflict Files:** 11 files with merge conflicts  

## Executive Summary

The merge from `main` into `av-ks-22` reveals significant architectural changes in the main branch, particularly around:

1. **Configuration System Migration**: Major transition from Hydra/OmegaConf YAML configs to Pydantic models
2. **Loss System Refactoring**: Complete overhaul of the loss computation system
3. **Repository Structure Changes**: Massive reorganization including map generation system migration
4. **Agent Architecture Updates**: Changes to neural network component system
5. **Training Pipeline Modernization**: New hyperparameter scheduling and loss processing

## Major Structural Changes in Main

### 1. Configuration System Overhaul

**Key Change**: Migration from Hydra/OmegaConf YAML-based configuration to Pydantic models

**Impact Areas**:
- `experiments/recipes/*.py` - All recipe files now use Pydantic configurations
- Training tools now use `./tools/run.py` instead of individual `train.py`, `sim.py`, etc.
- Configuration composition now handled programmatically in recipe functions

**Evidence from Commits**:
- Recipe functions now return `Tool` configuration objects
- Environment configs renamed from `EnvConfig` to `MettaGridConfig`
- Loss configurations moved from YAML to `LossConfig` Pydantic models

### 2. Loss System Complete Redesign

**Key Changes**:
- Reintroduction of `kickstarter.py` and advanced loss computation systems
- New hyperparameter scheduler with Redux architecture
- Prioritized sampling parameters and batch processing updates

**Files Added/Modified**:
- `metta/rl/kickstarter.py` - Re-added with modern implementation
- `metta/rl/hyperparameter_scheduler.py` - New Redux-based scheduler
- `tests/rl/test_kickstarter.py`, `tests/rl/test_losses.py` - Re-added test coverage

### 3. Repository Structure Reorganization

**Major Moves**:
- **Map Generation**: `metta/map/*` → `mettagrid/src/metta/mettagrid/mapgen/*`
- **Codebot**: `manybot/*` → `codebot/*` (complete restructuring)
- **New Components**: Home page app, MCP servers for WandB, Gitta git utilities

**New Directories**:
- `home/` - New Vite-based home page application
- `mcp_servers/wandb_dashboard/` - WandB MCP server for dashboard management
- `gitta/` - Git repository utilities
- `codebot/` - Consolidated bot tools (formerly manybot)

### 4. Agent Architecture Evolution

**Changes**:
- Neural network layer system updates in `agent/src/metta/agent/lib/`
- Component policy system refinements
- Agent testing framework improvements

### 5. Development Tooling Enhancements

**New Capabilities**:
- Home page at `home.softmax-research.net` with link management
- Enhanced CI/CD with new Docker builds
- Improved Claude Code integration with specialized agents
- Better cursor/IDE integration with rule files

## Merge Conflicts Detail Analysis

### Critical Conflicts (Require Manual Resolution)

#### 1. `experiments/recipes/arena.py`
**Conflict Type**: Configuration system migration  
**Issue**: 
- `av-ks-22` uses: `EnvConfig` and `LossConfig()`
- `main` uses: `MettaGridConfig` (renamed)
- Import statements differ fundamentally

**Resolution Strategy**: Use main's approach with `MettaGridConfig`

#### 2. `experiments/recipes/navigation.py`, `arena_basic_easy_shaped.py`, `icl_resource_chain.py`
**Conflict Type**: Same configuration migration issue  
**Pattern**: All recipe files follow same migration pattern from Hydra to Pydantic

#### 3. `metta/rl/trainer.py`
**Conflict Type**: Loss system integration  
**Issue**:
- `av-ks-22` branch: Missing loss system imports and hyperparameter scheduling
- `main` branch: Has reintroduced `kickstarter`, `Losses`, hyperparameter scheduler

**Critical Imports Added in Main**:
```python
from metta.rl.hyperparameter_scheduler import step_hyperparameters
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses, get_loss_experience_spec, process_minibatch_update
```

#### 4. `metta/rl/stats.py`
**Conflict Type**: Stats processing updates  
**Issue**: Enhanced statistics processing and batch calculation improvements

#### 5. `agent/src/metta/agent/lib/metta_layer.py`
**Conflict Type**: Neural architecture refinements  
**Issue**: Layer base class and component system updates

#### 6. `agent/tests/test_metta_agent.py`
**Conflict Type**: Test framework updates  
**Issue**: Testing methodology changes for agent architecture

### File Deletion/Addition Conflicts

#### 7. `metta/rl/kickstarter.py`
**Status**: Deleted in `av-ks-22`, Modified and re-added in `main`  
**Action Required**: Accept main's version (file should exist)

#### 8. `tests/rl/test_kickstarter.py`, `tests/rl/test_losses.py`
**Status**: Deleted in `av-ks-22`, Modified in `main`  
**Action Required**: Accept main's versions (restore test coverage)

## Changes That Will Auto-Merge Successfully

The following categories of changes from main will merge without conflicts:

### 1. New Infrastructure
- `.claude/agents/` - New specialized Claude agents
- `home/` - Complete new home page application
- `mcp_servers/` - WandB MCP server implementation
- `gitta/` - Git utilities package

### 2. Enhanced Development Tools
- Updated `.cursor/rules/` with better IDE integration
- Improved CI/CD workflows
- Enhanced Docker configurations

### 3. Map Generation System Migration
- Complete move from `metta/map/` to `mettagrid/src/metta/mettagrid/mapgen/`
- Addition of DCSS (Dungeon Crawl Stone Soup) map templates
- Enhanced procedural generation capabilities

### 4. Library and Documentation Improvements
- Enhanced research paper library with worker system
- Improved documentation and workflow guides
- Better Jupyter notebook widget integration

## Strategic Resolution Approach

### Phase 1: Accept Major Architectural Changes
1. **Configuration System**: Adopt main's Pydantic-based approach entirely
2. **Loss System**: Accept reintroduction of kickstarter and advanced loss computation
3. **Repository Structure**: Accept the map generation migration and new directories

### Phase 2: Recipe File Migration
1. Update all `experiments/recipes/*.py` files to use `MettaGridConfig` instead of `EnvConfig`
2. Ensure all loss configurations use `LossConfig()` properly
3. Verify recipe function signatures match main's expectations

### Phase 3: Training System Integration
1. Accept main's enhanced trainer with hyperparameter scheduling
2. Restore missing imports for kickstarter and loss systems
3. Update test coverage for restored components

### Phase 4: Testing and Validation
1. Run full test suite to ensure no regressions
2. Verify training pipeline functionality
3. Test recipe system with new Pydantic configurations

## Recommendations

### Immediate Actions
1. **Prioritize Configuration Migration**: The config system changes are fundamental and affect multiple systems
2. **Accept Loss System Improvements**: Main's loss system appears more comprehensive
3. **Preserve av-ks-22 Specific Features**: Identify and preserve any unique features from av-ks-22

### Risk Mitigation
1. **Backup Current State**: Ensure current av-ks-22 work is safely committed
2. **Incremental Testing**: Test each major system after conflict resolution
3. **Monitor Training Performance**: Verify no regressions in training effectiveness

## Implementation Plan

### Step 1: Resolve Core Conflicts
- Accept main's configuration system (`MettaGridConfig`)
- Restore loss system components (kickstarter, losses)
- Update trainer imports and functionality

### Step 2: Update Recipe Files
- Systematically update each recipe file to new configuration format
- Test recipe functions individually

### Step 3: Integration Testing
- Run training pipeline end-to-end
- Verify evaluation and analysis systems
- Test curriculum learning functionality

### Step 4: Validation
- Compare training performance before/after merge
- Verify all tests pass
- Confirm no functionality regressions

## Timeline Estimate

- **Conflict Resolution**: 2-4 hours
- **Testing & Validation**: 2-3 hours  
- **Performance Verification**: 1-2 hours
- **Total**: 5-9 hours for complete merge and validation

## Success Metrics

1. All merge conflicts resolved without compilation errors
2. Full test suite passes
3. Training pipeline runs successfully with new configuration system
4. No performance regression in training/evaluation
5. All unique av-ks-22 features preserved and functional

---

**Next Steps**: Begin with resolving the configuration system conflicts in recipe files, as these are foundational to the entire training pipeline.