# Task: Clean Up PR #1374 - MettaTrainer Refactoring

## Problem Statement
PR #1374 on richard-fully-functional branch has +5000/-2000 lines refactoring MettaTrainer into components. While the core refactoring is good, there are many extraneous changes that increase the diff unnecessarily.

## MVP Approach
Minimize the diff while keeping all architectural improvements by:
1. Removing cosmetic changes (imports, naming, comments)
2. Consolidating duplicated code patterns
3. Cleaning up type annotations for consistency
4. Removing unnecessary abstractions

## Implementation Plan

### Phase 1: Type Annotation Cleanup
1. Standardize on modern Python type syntax (`Type | None` instead of `Optional[Type]`)
2. Replace `Any` with specific types where possible
3. Remove unnecessary `from typing import` statements
4. Ensure consistent import ordering (stdlib, third-party, local)

### Phase 2: Remove Cosmetic Changes
1. Revert variable naming changes that don't improve clarity
   - Keep `mettagrid_env` instead of changing to `metta_grid_env`
2. Remove comments referencing old implementation ("like MettaTrainer did")
3. Standardize config variable names (`trainer_cfg` everywhere)

### Phase 3: Consolidate Duplicated Patterns
1. Create shared utilities for:
   - Device and distributed setup
   - Configuration validation
   - Stats processing pipelines
2. Remove redundant wrapper functions
3. Consolidate repeated config handling

### Phase 4: Simplify Component Structure
1. Remove `trainer_component.py` compatibility layer
2. Direct usage of functional interfaces where appropriate
3. Simplify component initialization patterns

### Phase 5: Clean Up Util Functions
1. Remove single-use utility functions
2. Consolidate related utilities
3. Ensure utilities are truly reusable

## Success Criteria
- [ ] Diff reduced by at least 30%
- [ ] All tests pass
- [ ] No functionality changes
- [ ] Core refactoring preserved
- [ ] Code more maintainable

## Implementation Updates
[To be updated during implementation]

## Notes
- Keep `run.py` and `tools/train.py` as requested (intentional duplicates)
- Preserve the component-based architecture
- Maintain backward compatibility where needed