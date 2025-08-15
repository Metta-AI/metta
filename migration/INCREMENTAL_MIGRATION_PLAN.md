# Incremental Migration Plan for richard-repo-org Branch

## Updated Architecture Goals

Based on our discussion, the target structure has been revised:

```
Current:
├── metta/src/metta/       # Main framework with nested src
├── agent/src/metta/agent/ # Agent implementations with nested src
├── common/src/metta/      # Shared utilities with nested src
├── mettagrid/src/metta/   # Environment with nested src
├── app_backend/src/metta/ # Backend services with nested src
├── configs/               # Hydra configs (mirrors code structure)
├── tools/                 # CLI tools (unchanged)
├── tests/                 # Test suite (unchanged)
├── [other directories]    # All preserved as-is

Target:
├── cogworks/metta/cogworks/  # RL framework (metta only, no agent)
├── agent/metta/agent/        # Agent implementations (standalone, flattened)
├── mettagrid/metta/mettagrid/# Environment (flattened)
├── common/metta/common/      # Shared utilities (flattened)
├── backend-shared/metta/     # Backend services (consolidated)
├── configs/               # Updated to match new structure
│   ├── agent/            # Points to metta.agent.*
│   ├── sim/              # Points to metta.cogworks.sim.*
│   ├── trainer/          # Points to metta.cogworks.rl.*
│   └── [others]          # Updated as needed
├── tools/                 # Unchanged
├── tests/                 # Unchanged
├── devops/               # Unchanged
├── docs/                 # Unchanged
├── recipes/              # Unchanged
├── scenes/               # Unchanged
├── scripts/              # Unchanged
├── experiments/          # Unchanged
├── [all others]          # Preserved as-is
```

**Important**: All directories not explicitly mentioned in the migration (tools/, devops/, docs/, recipes/, scenes/, scripts/, experiments/, observatory/, gridworks/, etc.) remain completely unchanged.

Key change: **agent/ remains separate** from cogworks, maintaining the clean separation that's been working well.

## Incremental Migration Steps

### Step 1: Clean Python Artifacts (5 minutes)
```bash
# Review what will be cleaned
uv run migration/tools/clean_python_artifacts.py --dry-run

# Actually clean artifacts
uv run migration/tools/clean_python_artifacts.py --clean

# Verify clean state
find . -name "__pycache__" | wc -l  # Should be 0
```

### Step 2: Phase 2 - Flatten Common Package (Low Risk)

This is the safest starting point with only 26 outgoing dependencies.

```bash
# 1. Create backup branch point
git checkout -b migration-backup-$(date +%Y%m%d)
git checkout richard-repo-org

# 2. Test import rewrites in dry-run
uv run migration/tools/rewrite_imports.py --phase phase2_common --dry-run

# 3. Manually flatten common structure
mkdir -p common/metta/common
cp -r common/src/metta/common/* common/metta/common/
# Keep src temporarily for compatibility

# 4. Update common's pyproject.toml
cat > common/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "softmax-common"
version = "0.1.0"
dependencies = []

[tool.setuptools]
packages = ["metta.common"]
package-dir = {"": "."}
EOF

# 5. Test the new structure
cd common && pip install -e . && cd ..
python -c "from metta.common import logger; print('✓ Import works')"

# 6. Run tests
uv run pytest tests/common/ -x
```

### Step 3: Phase 3 - Flatten Agent Package (Low Risk)

Agent has no incoming dependencies from other packages, making it safe to migrate early.

```bash
# 1. Flatten agent structure
mkdir -p agent/metta/agent
cp -r agent/src/metta/agent/* agent/metta/agent/
# Keep src temporarily for compatibility

# 2. Update agent's pyproject.toml
cat > agent/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "softmax-agent"
version = "0.1.0"
dependencies = ["softmax-common"]

[tool.setuptools]
packages = ["metta.agent"]
package-dir = {"": "."}
EOF

# 3. Test the new structure
cd agent && pip install -e . && cd ..
python -c "from metta.agent import MettaAgent; print('✓ Import works')"

# 4. Update imports that reference agent
uv run migration/tools/rewrite_imports.py --phase phase4_cogworks --dry-run | grep agent

# 5. Run tests
uv run pytest tests/agent/ -x
```

### Step 4: Phase 3 - Flatten MettagGrid (Medium Risk)

MettagGrid has C++ components, requiring careful handling.

```bash
# 1. Flatten mettagrid structure
mkdir -p mettagrid/metta/mettagrid
cp -r mettagrid/src/metta/mettagrid/* mettagrid/metta/mettagrid/

# 2. Update CMakeLists.txt paths
sed -i.bak 's|src/metta/mettagrid|metta/mettagrid|g' mettagrid/CMakeLists.txt

# 3. Update pyproject.toml
sed -i.bak 's|"src"|"."|g' mettagrid/pyproject.toml

# 4. Rebuild C++ extensions
cd mettagrid
metta clean
metta install core
cd ..

# 5. Test imports
python -c "from metta.mettagrid import MettaGridEnv; print('✓ Import works')"

# 6. Performance test
uv run python -c "
from metta.mettagrid import MettaGridEnv
import time
env = MettaGridEnv.from_name('navigation', num_envs=8)
start = time.perf_counter()
for _ in range(1000):
    env.step([0]*8)
print(f'✓ Performance: {1000/(time.perf_counter()-start):.0f} steps/sec')
"
```

### Step 5: Create Compatibility Layers

Before removing old src/ directories, add compatibility imports:

```bash
# Create compatibility imports in old locations
cat > common/src/metta/common/__init__.py << 'EOF'
# Compatibility layer - will be removed after full migration
import warnings
warnings.warn(
    "Importing from common/src/metta/common is deprecated. "
    "Use 'from metta.common' instead.",
    DeprecationWarning,
    stacklevel=2
)
from metta.common import *
EOF

# Repeat for agent and mettagrid
cat > agent/src/metta/agent/__init__.py << 'EOF'
# Compatibility layer - will be removed after full migration
import warnings
warnings.warn(
    "Importing from agent/src/metta/agent is deprecated. "
    "Use 'from metta.agent' instead.",
    DeprecationWarning,
    stacklevel=2
)
from metta.agent import *
EOF
```

### Step 6: Update Hydra Configs

Critical step: Update all `_target_` fields in configs to match new import paths.

```bash
# Create config update script
cat > migration/tools/update_configs.py << 'EOF'
#!/usr/bin/env uv run
import yaml
from pathlib import Path

def update_config_targets(config_dir: Path, mappings: dict):
    """Update _target_ fields in all YAML configs."""
    
    for yaml_file in config_dir.rglob("*.yaml"):
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
            
            if not config:
                continue
                
            modified = False
            
            def update_targets(obj):
                nonlocal modified
                if isinstance(obj, dict):
                    if '_target_' in obj:
                        for old, new in mappings.items():
                            if obj['_target_'].startswith(old):
                                obj['_target_'] = obj['_target_'].replace(old, new)
                                modified = True
                                print(f"  {yaml_file.name}: {old} -> {new}")
                    for value in obj.values():
                        update_targets(value)
                elif isinstance(obj, list):
                    for item in obj:
                        update_targets(item)
            
            update_targets(config)
            
            if modified:
                with open(yaml_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")

# Phase-specific mappings
phase2_mappings = {
    'common.src.metta.common': 'metta.common',
}

phase3_mappings = {
    'agent.src.metta.agent': 'metta.agent',
    'mettagrid.src.metta.mettagrid': 'metta.mettagrid',
}

phase4_mappings = {
    'metta.rl': 'metta.cogworks.rl',
    'metta.sim': 'metta.cogworks.sim',
    'metta.eval': 'metta.cogworks.eval',
    'metta.sweep': 'metta.cogworks.sweep',
    'metta.map': 'metta.cogworks.mapgen',
}

# Apply updates based on current phase
import sys
phase = sys.argv[1] if len(sys.argv) > 1 else 'phase2'

if phase == 'phase2':
    update_config_targets(Path('configs'), phase2_mappings)
elif phase == 'phase3':
    update_config_targets(Path('configs'), phase3_mappings)
elif phase == 'phase4':
    update_config_targets(Path('configs'), phase4_mappings)
else:
    print(f"Unknown phase: {phase}")
EOF

chmod +x migration/tools/update_configs.py

# Run for current phase
uv run migration/tools/update_configs.py phase2  # or phase3, phase4
```

### Step 7: Validate Everything Works

```bash
# Run comprehensive tests
uv run pytest tests/ -x --maxfail=10

# Check for any broken imports
uv run migration/tools/validate_imports.py

# Try a training run
uv run ./tools/train.py run=test_migration trainer.total_timesteps=1000 trainer.num_workers=2
```

### Step 7: Remove Old Structures (After Validation)

Only after everything is confirmed working:

```bash
# Remove old src directories
rm -rf common/src
rm -rf agent/src  
rm -rf mettagrid/src

# Remove compatibility layers
find . -path "*/src/metta/*/__init__.py" -exec grep -l "DeprecationWarning" {} \; | xargs rm

# Final validation
uv run pytest tests/ -x
```

## Rollback Procedures

At any point, if issues arise:

```bash
# Option 1: Git reset if not committed
git reset --hard HEAD
git clean -fd

# Option 2: Restore from backup branch
git checkout migration-backup-$(date +%Y%m%d)
git branch -D richard-repo-org
git checkout -b richard-repo-org

# Option 3: Restore specific package
git checkout HEAD -- common/
git checkout HEAD -- agent/
# etc.

# Clean Python artifacts after rollback
uv run migration/tools/clean_python_artifacts.py --clean
```

## Common Pitfalls to Avoid

1. **Stale .pyc files**: Always clean before testing migration
2. **Import caching**: Restart Python interpreter between tests
3. **Editable installs**: May need `pip install -e . --force-reinstall`
4. **CMake caches**: Run `metta clean` for mettagrid changes
5. **CI caches**: May need to clear GitHub Actions caches

## Success Metrics

- [ ] All tests pass with new structure
- [ ] No deprecation warnings in normal operation
- [ ] Performance benchmarks within 5% of baseline
- [ ] Clean import validation (no invalid imports)
- [ ] Successful training run with migrated packages

## Timeline Estimate

With incremental approach on richard-repo-org branch:
- Step 1 (Cleanup): 5 minutes
- Step 2 (Common): 30 minutes
- Step 3 (Agent): 30 minutes
- Step 4 (MettagGrid): 45 minutes
- Step 5 (Compatibility): 15 minutes
- Step 6 (Validation): 30 minutes
- Step 7 (Cleanup): 15 minutes

**Total: ~3 hours** for careful incremental migration

## Next Immediate Actions

1. Review this plan
2. Run cleanup script: `uv run migration/tools/clean_python_artifacts.py --clean`
3. Start with common package (lowest risk)
4. Test thoroughly at each step
5. Commit working states incrementally