# Metta Repository Migration Tools

This directory contains tools and documentation for the incremental migration of the Metta repository to a flattened structure.

## Quick Start

### 1. Clean Python Artifacts (Recommended First Step)
```bash
# See what will be removed
uv run migration/tools/clean_python_artifacts.py --dry-run

# Actually clean
uv run migration/tools/clean_python_artifacts.py --clean
```

### 2. Validate Current Imports
```bash
uv run migration/tools/validate_imports.py
```

### 3. Analyze Dependencies
```bash
uv run migration/tools/analyze_dependencies.py
```

### 4. Test Import Rewrites (Dry Run)
```bash
# Test Phase 2 (Common package)
uv run migration/tools/rewrite_imports.py --phase phase2_common --dry-run

# Test Phase 3 (MettagGrid)
uv run migration/tools/rewrite_imports.py --phase phase3_mettagrid --dry-run

# Test Phase 4 (Cogworks + Agent flattening)
uv run migration/tools/rewrite_imports.py --phase phase4_cogworks --dry-run
```

## Directory Structure

```
migration/
├── README.md                          # This file
├── PHASE1_COMPLETE.md                 # Phase 1 completion summary
├── INCREMENTAL_MIGRATION_PLAN.md     # Step-by-step migration guide
├── baselines/                         # Baseline metrics and analysis
│   ├── baseline-metrics.json         # Test count baseline
│   ├── dependency-analysis.json      # Full dependency graph
│   ├── import-validation-report.json # Import validation results
│   └── dependencies-packages.dot     # Graphviz visualization
├── docs/
│   └── DEPENDENCY_SUMMARY.md         # Human-readable dependency analysis
└── tools/
    ├── validate_imports.py            # Import validation tool
    ├── analyze_dependencies.py        # Dependency analysis tool
    ├── rewrite_imports.py            # Automated import rewriting
    └── clean_python_artifacts.py     # Python artifact cleanup

docs/REORGANIZATION_PLAN.md           # Full 5-phase reorganization plan (updated)
```

## Key Changes from Original Plan

1. **Agent stays separate**: The `agent/` package will remain standalone rather than merging into `cogworks/`
2. **Incremental approach**: We're doing the migration on the `richard-repo-org` branch incrementally
3. **Flattening focus**: Primary goal is removing `src/` directories, not full restructuring

## Current Architecture Target

```
├── cogworks/metta/cogworks/     # RL framework (metta only)
├── agent/metta/agent/           # Agent implementations (separate)
├── mettagrid/metta/mettagrid/   # Environment
├── common/metta/common/         # Shared utilities
└── backend-shared/metta/        # Backend services
```

## Migration Order (by risk)

1. **Common** (lowest risk - 26 dependencies)
2. **Agent** (low risk - no incoming deps) 
3. **MettagGrid** (medium risk - C++ components)
4. **Cogworks** (highest risk - 1077 incoming deps)

## Tools Are Non-Destructive

- All tools support `--dry-run` mode by default
- Import rewriter creates backups before changes
- Clean script shows what will be removed before acting
- Everything can be rolled back with git

## Next Steps

1. Review `INCREMENTAL_MIGRATION_PLAN.md`
2. Clean Python artifacts
3. Start with Phase 2 (Common package)
4. Test thoroughly at each step
5. Commit working states incrementally