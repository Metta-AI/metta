# Metta Codebase Dependency Summary

Generated as part of Phase 1 migration preparation.

## Overview

This document summarizes the dependencies discovered during Phase 1 analysis of the Metta codebase.

## Package Statistics

| Package | Modules | Outgoing Deps | Incoming Deps | Risk Level |
|---------|---------|---------------|---------------|------------|
| metta | 254 | 411 | 1077 | **HIGH** - Central hub |
| mettagrid | 71 | 183 | 0 | LOW - Self-contained |
| app_backend | 43 | 131 | 0 | LOW - Isolated |
| agent | 34 | 72 | 0 | LOW - Isolated |
| common | 22 | 26 | 0 | LOW - Few dependencies |
| mettascope | 2 | 10 | 4 | LOW - Small package |

## High-Risk Modules

These modules have the most connections and will require careful attention during migration:

1. **metta.common.util.config** (51 connections)
   - 0 dependencies, 51 dependents
   - Central configuration module used throughout codebase
   - Migration Impact: Phase 2 - Must maintain compatibility

2. **metta.mettagrid.curriculum.core** (36 connections)
   - 0 dependencies, 36 dependents
   - Core curriculum functionality
   - Migration Impact: Phase 3 - Part of mettagrid migration

3. **metta.rl.trainer** (32 connections)
   - 31 dependencies, 1 dependent
   - Main training orchestrator
   - Migration Impact: Phase 4 - Critical for cogworks

4. **metta.map.scene** (31 connections)
   - 4 dependencies, 27 dependents
   - Scene management for environments
   - Migration Impact: Phase 4 - Will become mapgen in cogworks

5. **metta.common.util.constants** (29 connections)
   - 0 dependencies, 29 dependents
   - Shared constants across codebase
   - Migration Impact: Phase 2 - Must be available early

## Circular Dependencies

Only 1 circular dependency group was found:
- `metta.setup.tools.local.kind` ↔ `metta.setup.local_commands`

This is a minor issue in the setup tools and won't affect the main migration.

## External Dependencies

The codebase uses 159 external packages. Key dependencies include:
- Core: numpy, torch, pydantic, hydra-core
- RL: gymnasium, pufferlib, tensordict
- Infrastructure: wandb, docker, kubernetes
- Utilities: rich, typer, pandas

## Migration Order Recommendations

Based on dependency analysis, the recommended migration order aligns with the planned phases:

### Phase 2 (Low Risk - Foundation)
- `common` package - Only 26 outgoing dependencies
- `app_backend` shared components - Isolated from core

### Phase 3 (Medium Risk - Self-contained)
- `mettagrid` - No incoming dependencies from other packages
- Can be migrated independently

### Phase 4 (High Risk - Core Integration)
- `metta` + `agent` → `cogworks`
- Contains the most interconnected modules
- Will require careful handling of 1077 incoming dependencies

### Phase 5 (Cleanup)
- Remove old structures
- Update all remaining references

## Import Validation Results

- **Valid imports**: 227 unique local imports verified
- **Invalid imports**: 35 (mostly in old wandb snapshots)
- **Total files analyzed**: 694 Python files
- **Total import statements**: ~2,500

## Key Insights

1. **Central Hub Pattern**: The `metta` package is the central hub with 1077 incoming dependencies. This makes it the highest risk for migration but also the most important to get right.

2. **Clean Boundaries**: `mettagrid`, `app_backend`, and `agent` have no incoming dependencies from other packages, making them good candidates for early migration.

3. **Configuration Critical**: `metta.common.util.config` is used by 51 modules. Any changes to this module will have widespread impact.

4. **Limited Circular Dependencies**: Only 1 circular dependency found, which is excellent for a codebase of this size.

5. **Trainer Complexity**: `metta.rl.trainer` has 31 dependencies, making it one of the most complex modules to migrate.

## Risk Mitigation Strategies

1. **Compatibility Layers**: Create import compatibility layers during each phase to allow gradual migration.

2. **Parallel Testing**: Run tests against both old and new structures during migration.

3. **Incremental Migration**: Start with low-risk, isolated packages before tackling the core.

4. **Dependency Injection**: Consider refactoring high-dependency modules to use dependency injection for easier testing.

5. **Feature Flags**: Use feature flags to switch between old and new import paths during transition.

## Tools Created

Phase 1 has delivered the following migration tools:

1. **validate_imports.py** - Validates all Python imports and identifies broken references
2. **analyze_dependencies.py** - Analyzes dependency graph and identifies high-risk modules
3. **rewrite_imports.py** - Automated import path rewriting for each migration phase
4. **migration-test.yml** - CI/CD pipeline for continuous validation during migration

## Next Steps

1. Review this dependency analysis with the team
2. Validate the migration order based on current development priorities
3. Create feature branches for Phase 2 implementation
4. Begin Phase 2 with common and backend-shared packages