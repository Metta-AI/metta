# Migration Impact Analysis: Directories

## Directories Being Restructured (5 packages)

These are the ONLY directories having their internal structure changed:

### 1. metta/ → cogworks/
- **Current**: `metta/src/metta/{rl,sim,eval,sweep,map}/`
- **Target**: `cogworks/metta/cogworks/{rl,sim,eval,sweep,mapgen}/`
- **Impact**: High - 1,077 incoming dependencies
- **Note**: agent components NOT moving here (staying separate)

### 2. agent/
- **Current**: `agent/src/metta/agent/`
- **Target**: `agent/metta/agent/`
- **Impact**: Low - No incoming dependencies from other packages
- **Note**: Stays as separate package, just flattened

### 3. common/
- **Current**: `common/src/metta/common/`
- **Target**: `common/metta/common/`
- **Impact**: Low - Only 26 outgoing dependencies

### 4. mettagrid/
- **Current**: `mettagrid/src/metta/mettagrid/`
- **Target**: `mettagrid/metta/mettagrid/`
- **Impact**: Medium - C++ components need CMake updates

### 5. app_backend/ → backend-shared/
- **Current**: `app_backend/src/metta/app_backend/`
- **Target**: `backend-shared/metta/backend_shared/`
- **Impact**: Low - Isolated services

## Directories Requiring Import Updates Only

These directories stay in the same location but need import statements updated:

### configs/
- **Location**: Unchanged
- **Changes**: `_target_` fields in YAML files updated to match new module paths
- **Tool**: `migration/tools/update_configs.py`
- **Example**: `_target_: metta.rl.trainer` → `_target_: metta.cogworks.rl.trainer`

### tests/
- **Location**: Unchanged
- **Changes**: Import statements updated in test files
- **Tool**: `migration/tools/rewrite_imports.py`
- **Structure**: Test organization remains identical

## Directories Completely Unchanged

These directories are NOT touched by the migration at all:

### Development Tools
- `tools/` - CLI scripts and utilities
- `devops/` - Deployment and operations
- `scripts/` - Utility scripts
- `.github/` - GitHub Actions (except new migration workflow)

### Documentation & Examples
- `docs/` - Documentation
- `recipes/` - Example recipes and tutorials
- `scenes/` - Scene definitions
- `experiments/` - Experiment tracking

### Web Interfaces
- `observatory/` - Monitoring dashboard
- `gridworks/` - Web interface
- `mettascope/` - Visualization tools
- `cogweb/` - Web components

### Data & Output
- `train_dir/` - Training outputs
- `outputs/` - General outputs
- `wandb/` - Weights & Biases logs
- `library/` - Asset library

### Other
- `manybot/` - Bot framework
- `node_modules/` - JavaScript dependencies (if present)
- Any other unlisted directories

## Import Path Changes Summary

### Phase 2 (Common)
- `from common.src.metta.common` → `from metta.common`

### Phase 3 (Agent & MettagGrid)
- `from agent.src.metta.agent` → `from metta.agent`
- `from mettagrid.src.metta.mettagrid` → `from metta.mettagrid`

### Phase 4 (Cogworks)
- `from metta.rl` → `from metta.cogworks.rl`
- `from metta.sim` → `from metta.cogworks.sim`
- `from metta.eval` → `from metta.cogworks.eval`
- `from metta.sweep` → `from metta.cogworks.sweep`
- `from metta.map` → `from metta.cogworks.mapgen`

## Files That Need Special Attention

### pyproject.toml files
- Each migrated package needs updated package paths
- Dependencies between packages need updating

### CMakeLists.txt (mettagrid only)
- Update paths from `src/metta/mettagrid` to `metta/mettagrid`

### CI/CD workflows
- May need path updates if they reference specific package structures

## Validation Checklist

Before considering migration complete:

- [ ] All imports resolve correctly
- [ ] All tests pass
- [ ] All configs have valid `_target_` fields
- [ ] Training pipeline runs end-to-end
- [ ] No files accidentally moved from preserved directories
- [ ] No new files created in preserved directories
- [ ] All preserved directories remain at exact same paths
- [ ] Documentation reflects new structure

## Quick Reference: What Changes vs What Doesn't

**Changes**:
- Internal structure of 5 packages (metta, agent, common, mettagrid, app_backend)
- Import statements throughout codebase
- Config `_target_` fields
- Package metadata (pyproject.toml)

**Does NOT Change**:
- Location of any directory except the 5 packages
- Any file outside the 5 packages (except import statements)
- Directory names (except app_backend → backend-shared)
- File names
- Configuration structure (just target values)
- Test organization
- Documentation structure
- Web interfaces
- CLI tools