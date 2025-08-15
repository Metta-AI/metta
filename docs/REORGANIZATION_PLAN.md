# Metta Repository Reorganization Plan

## Executive Summary

This document outlines a 5-phase approach to reorganizing the Metta repository from its current nested structure to a flattened, namespace-package architecture with `softmax-*` PyPI naming. The reorganization aims to improve modularity, simplify imports, and establish clear package boundaries while minimizing disruption to ongoing development.

**Timeline:** 8-10 weeks total (1.5-2 weeks per phase)

## Current State Analysis

### Repository Structure
```
Current:
├── metta/           # Main training framework (mixed concerns)
├── agent/           # Agent implementations
├── common/          # Shared utilities
├── mettagrid/       # C++/Python environment
├── app_backend/     # Backend services
├── tools/           # CLI scripts
├── tests/           # Test suite
└── configs/         # Hydra configurations

Target:
├── cogworks/        # RL framework (metta only, without agent)
├── agent/           # Agent implementations (flattened, standalone)
├── mettagrid/       # Environment (standalone)
├── common/          # Shared utilities
├── backend-shared/  # Backend services
└── tools/           # CLI scripts (unchanged location)
```

### Key Statistics
- **Total Python files:** ~450
- **Lines of code:** ~60,000
- **Import statements to update:** ~2,500
- **Test files:** 62+
- **Configuration files:** 40+

## Phase 1: Foundation & Preparation (Week 1-2)

### Objectives
- Establish baseline functionality metrics
- Create migration tooling
- Set up parallel CI/CD pipelines
- Document all current dependencies

### Tasks

#### 1.1 Baseline Establishment
```bash
# Create comprehensive test baseline
pytest tests/ --json-report --json-report-file=baseline-tests.json
coverage run -m pytest tests/
coverage report --format=json > baseline-coverage.json

# Document import graph
python -m pydeps metta --max-bacon=2 --cluster > import-graph-before.svg
```

#### 1.2 Migration Tooling
Create automated tools for:
- Import path rewriting
- Dependency graph analysis
- Package structure validation
- Import compatibility checking

**validate_imports.py:**
```python
#!/usr/bin/env python3
"""Validate all Python imports in the repository."""

import ast
import sys
from pathlib import Path
from typing import Set, List, Tuple

class ImportValidator:
    def __init__(self):
        self.valid_imports: Set[str] = set()
        self.invalid_imports: List[Tuple[Path, str, str]] = []
        
    def validate_file(self, file_path: Path) -> bool:
        """Validate all imports in a Python file."""
        try:
            with open(file_path) as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith('metta.'):
                        try:
                            __import__(node.module)
                            self.valid_imports.add(node.module)
                        except ImportError as e:
                            self.invalid_imports.append((file_path, node.module, str(e)))
            return True
        except Exception as e:
            print(f"ERROR parsing {file_path}: {e}")
            return False
    
    def validate_directory(self, directory: Path):
        """Recursively validate all Python files."""
        for py_file in directory.rglob("*.py"):
            if '.venv' not in str(py_file) and '__pycache__' not in str(py_file):
                self.validate_file(py_file)
    
    def report(self):
        """Generate validation report."""
        print(f"✓ Valid imports: {len(self.valid_imports)}")
        if self.invalid_imports:
            print(f"✗ Invalid imports: {len(self.invalid_imports)}")
            for path, module, error in self.invalid_imports[:10]:
                print(f"  {path}: {module} - {error}")

if __name__ == "__main__":
    validator = ImportValidator()
    validator.validate_directory(Path("."))
    validator.report()
    sys.exit(1 if validator.invalid_imports else 0)
```

#### 1.3 CI/CD Pipeline Setup
Add parallel testing for both old and new structures:

**.github/workflows/migration-test.yml:**
```yaml
name: Migration Testing

on: [push, pull_request]

jobs:
  test-current:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test Current Structure
        run: |
          uv sync
          pytest tests/ -x
          
  test-migrated:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test Migrated Structure
        run: |
          # Will be activated in Phase 2
          echo "Migration tests will run here"
```

### Success Criteria
- [ ] All current tests pass
- [ ] Baseline metrics documented
- [ ] Migration tools created and tested
- [ ] Team trained on migration process

### Rollback Plan
This phase is preparation only - no changes to production code.

## Phase 2: Common & Backend-Shared Migration (Week 3-4)

### Objectives
- Migrate lowest-risk, foundational packages first
- Establish new package structure pattern
- Validate packaging and installation

### Tasks

#### 2.1 Common Package Migration
```bash
# Create new structure
mkdir -p common/metta/common
mv common/src/metta/common/* common/metta/common/

# Update pyproject.toml
cat > common/pyproject.toml << EOF
[project]
name = "softmax-common"
version = "0.1.0"

[tool.setuptools]
packages = ["metta.common"]
package-dir = {"": "."}
EOF

# Test installation
cd common && pip install -e . && cd ..
python -c "from metta.common import logger"
```

#### 2.2 Backend-Shared Creation
Consolidate shared backend services:
```bash
# Create backend-shared structure
mkdir -p backend-shared/metta/backend_shared

# Move shared services
mv app_backend/src/metta/app_backend/clients/stats_client.py backend-shared/metta/backend_shared/
mv app_backend/src/metta/app_backend/auth.py backend-shared/metta/backend_shared/
# ... other shared services
```

#### 2.3 Import Compatibility Layer
Create temporary import bridges:

**common/src/metta/common/__init__.py:**
```python
# Temporary compatibility layer
import warnings
warnings.warn(
    "Importing from common/src/metta/common is deprecated. "
    "Use 'from metta.common' instead.", 
    DeprecationWarning, 
    stacklevel=2
)

# Re-export from new location
from metta.common import *
```

### Success Criteria
- [ ] softmax-common package installs correctly
- [ ] softmax-backend-shared package created
- [ ] All imports to these packages work
- [ ] No test failures related to common utilities

### Rollback Plan
```bash
# Revert structure changes
git checkout HEAD -- common/
git checkout HEAD -- backend-shared/
rm -rf backend-shared/

# Reinstall original packages
uv sync --force-reinstall
```

## Phase 3: MettagGrid Migration (Week 5-6)

### Objectives
- Migrate the self-contained environment package
- Update C++ build configurations
- Validate performance remains unchanged

### Tasks

#### 3.1 MettagGrid Structure Update
```bash
# Flatten mettagrid structure
mkdir -p mettagrid/metta/mettagrid
mv mettagrid/src/metta/mettagrid/* mettagrid/metta/mettagrid/

# Update CMakeLists.txt paths
sed -i 's|src/metta/mettagrid|metta/mettagrid|g' mettagrid/CMakeLists.txt

# Update pyproject.toml
sed -i 's|name = "metta-mettagrid"|name = "softmax-mettagrid"|' mettagrid/pyproject.toml
```

#### 3.2 Performance Validation
```python
# benchmark.py - Ensure no performance regression
import time
import numpy as np
from metta.mettagrid import MettaGridEnv

def benchmark_env(n_steps=10000):
    env = MettaGridEnv.from_name("navigation", num_envs=32)
    
    start = time.perf_counter()
    for _ in range(n_steps):
        actions = np.random.randint(0, env.action_space.n, size=32)
        obs, rewards, dones, infos = env.step(actions)
    
    elapsed = time.perf_counter() - start
    print(f"Steps per second: {n_steps * 32 / elapsed:.0f}")
    return elapsed

# Compare before and after
baseline_time = benchmark_env()  # Run before migration
new_time = benchmark_env()       # Run after migration
assert new_time < baseline_time * 1.1, "Performance regression detected"
```

### Success Criteria
- [ ] softmax-mettagrid installs correctly
- [ ] C++ extensions build successfully
- [ ] Performance benchmarks within 5% of baseline
- [ ] All environment tests pass

### Rollback Plan
```bash
# MettagGrid is relatively isolated, easy rollback
git checkout HEAD -- mettagrid/
uv sync --force-reinstall
```

## Phase 4: Cogworks & Agent Restructure (Week 7-8)

### Objectives
- Create cogworks package from metta/ components only
- Flatten agent/ structure (remove src/ directory)
- Update all internal cross-references
- Migrate complex interdependencies

### Tasks

#### 4.1 Create Cogworks Structure
```bash
# Create cogworks package (without agent)
mkdir -p cogworks/metta/cogworks/{rl,eval,sim,sweep,mapgen}

# Move metta components only
mv metta/rl/* cogworks/metta/cogworks/rl/
mv metta/eval/* cogworks/metta/cogworks/eval/
mv metta/sim/* cogworks/metta/cogworks/sim/
mv metta/sweep/* cogworks/metta/cogworks/sweep/
mv metta/map/* cogworks/metta/cogworks/mapgen/

# Flatten agent structure separately
mkdir -p agent/metta/agent
mv agent/src/metta/agent/* agent/metta/agent/
rmdir agent/src/metta/agent agent/src/metta agent/src
```

#### 4.2 Automated Import Updates
```python
#!/usr/bin/env python3
"""update_imports.py - Batch update import statements."""

import re
from pathlib import Path

# Define import mappings
IMPORT_MAPPINGS = {
    r'from metta\.rl': 'from metta.cogworks.rl',
    r'from metta\.eval': 'from metta.cogworks.eval',
    r'from metta\.sim': 'from metta.cogworks.sim',
    r'from metta\.sweep': 'from metta.cogworks.sweep',
    r'from metta\.map': 'from metta.cogworks.mapgen',
    r'from agent\.src\.metta\.agent': 'from metta.agent',
    r'import metta\.rl': 'import metta.cogworks.rl',
    r'import agent\.src\.metta\.agent': 'import metta.agent',
}

def update_file(file_path: Path):
    """Update imports in a single file."""
    content = file_path.read_text()
    original = content
    
    for pattern, replacement in IMPORT_MAPPINGS.items():
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        file_path.write_text(content)
        print(f"Updated: {file_path}")
        return True
    return False

def update_directory(directory: Path):
    """Update all Python files in directory."""
    count = 0
    for py_file in directory.rglob("*.py"):
        if '.venv' not in str(py_file):
            if update_file(py_file):
                count += 1
    print(f"Updated {count} files")

if __name__ == "__main__":
    update_directory(Path("cogworks"))
    update_directory(Path("tools"))
    update_directory(Path("tests"))
```

#### 4.3 Circular Dependency Resolution
Identify and fix circular imports:

```python
# detect_circular.py
import ast
from collections import defaultdict
from pathlib import Path

def build_import_graph(directory: Path):
    """Build a graph of all imports."""
    graph = defaultdict(set)
    
    for py_file in directory.rglob("*.py"):
        if '.venv' in str(py_file):
            continue
            
        module_name = str(py_file).replace('/', '.').replace('.py', '')
        
        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    graph[module_name].add(node.module)
        except:
            pass
    
    return graph

def find_cycles(graph):
    """Find circular dependencies."""
    # Simplified cycle detection
    cycles = []
    for module, imports in graph.items():
        for imported in imports:
            if imported in graph and module in graph[imported]:
                cycle = tuple(sorted([module, imported]))
                if cycle not in cycles:
                    cycles.append(cycle)
    return cycles

# Detect and report cycles
graph = build_import_graph(Path("cogworks"))
cycles = find_cycles(graph)
if cycles:
    print(f"Found {len(cycles)} circular dependencies:")
    for a, b in cycles[:10]:
        print(f"  {a} <-> {b}")
```

### Success Criteria
- [ ] softmax-cogworks package builds and installs
- [ ] softmax-agent package builds and installs separately
- [ ] All cogworks tests pass
- [ ] All agent tests pass
- [ ] No circular import errors
- [ ] Training pipeline works end-to-end with separate packages

### Rollback Plan
This is the most complex phase. Keep parallel structures:
```bash
# Don't delete metta/ and agent/ until fully validated
# Keep compatibility imports during transition
# Can run both old and new in parallel for testing
```

## Phase 5: Final Migration & Cleanup (Week 9-10)

### Objectives
- Migrate remaining components (tools, configs, tests)
- Remove old structure and compatibility layers
- Update all documentation
- Final validation

### Tasks

#### 5.1 Tools & Scripts Migration
```bash
# Update all tools/ scripts
python update_imports.py tools/

# Update all recipes/ scripts  
python update_imports.py recipes/

# Validate all scripts work
for script in tools/*.py; do
    echo "Testing $script"
    python $script --help || echo "FAILED: $script"
done
```

#### 5.2 Configuration Updates
```python
# update_configs.py - Update Hydra configs
import yaml
from pathlib import Path

def update_config(config_path: Path):
    """Update module references in config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update _target_ fields
    def update_targets(obj):
        if isinstance(obj, dict):
            if '_target_' in obj:
                obj['_target_'] = obj['_target_'].replace(
                    'metta.rl', 'metta.cogworks.rl'
                ).replace(
                    'metta.agent', 'metta.cogworks.agent'
                )
            for value in obj.values():
                update_targets(value)
        elif isinstance(obj, list):
            for item in obj:
                update_targets(item)
    
    update_targets(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Update all configs
for config_file in Path("configs").rglob("*.yaml"):
    update_config(config_file)
```

#### 5.3 Documentation Updates
```bash
# Update all documentation
find docs/ -name "*.md" -exec sed -i 's/from metta\.rl/from metta.cogworks.rl/g' {} \;
find docs/ -name "*.md" -exec sed -i 's/metta-agent/softmax-cogworks/g' {} \;

# Update README
sed -i 's/pip install metta/pip install softmax-cogworks/' README.md
```

#### 5.4 Old Structure Removal
```bash
# Only after full validation
rm -rf metta/rl metta/agent metta/eval metta/sim metta/sweep metta/map
rm -rf agent/
rm -rf common/src/
rm -rf mettagrid/src/

# Remove compatibility layers
find . -name "__init__.py" -exec grep -l "DeprecationWarning" {} \; | xargs rm
```

### Success Criteria
- [ ] All tests pass with new structure
- [ ] CI/CD fully migrated
- [ ] Documentation updated
- [ ] No references to old structure remain
- [ ] Performance benchmarks meet or exceed baseline

### Rollback Plan
Keep full backup before removal:
```bash
# Before removing old structure
tar -czf metta-backup-$(date +%Y%m%d).tar.gz metta/ agent/ common/src/ mettagrid/src/

# Can restore if needed
tar -xzf metta-backup-*.tar.gz
```

## Risk Matrix & Mitigation Strategies

### High-Risk Areas

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Circular imports in cogworks | High | Medium | Phase 4 detection tools, gradual migration |
| Breaking production training | Critical | Low | Parallel structures, extensive testing |
| CI/CD pipeline failures | High | Medium | Dual pipelines during transition |
| Import path confusion | Medium | High | Linting rules, clear documentation |
| Package installation issues | High | Medium | Test with clean environments |

### Potential Failure Points

#### 1. Import Path Confusion
**Problem:** Developers unsure which import to use  
**Solution:** 
- Enforce linting rule for `metta.*` imports
- Provide import cheat sheet
- Automated import correction tool

#### 2. Hidden Dependencies
**Problem:** Undocumented dependencies between modules  
**Solution:**
- Phase 1 comprehensive dependency mapping
- Gradual migration to expose issues early
- Maintain import graph visualization

#### 3. Performance Regression
**Problem:** New structure impacts performance  
**Solution:**
- Benchmark at each phase
- Profile critical paths
- Keep optimization separate from reorganization

#### 4. Configuration Breakage
**Problem:** Hydra configs reference old modules  
**Solution:**
- Automated config updating (Phase 5)
- Config validation tests
- Backwards compatibility during transition

## Success Metrics

### Quantitative Metrics
- **Test Coverage:** Maintain or improve from baseline (currently ~70%)
- **Build Time:** No more than 10% increase
- **Import Time:** No more than 5% increase
- **Package Size:** Reasonable increase (<20%) due to namespace packages

### Qualitative Metrics
- **Developer Experience:** Survey team after each phase
- **Code Clarity:** Reduced cross-package dependencies
- **Maintenance:** Easier to understand package boundaries

## Communication Plan

### Phase Kickoffs
- Team meeting before each phase
- Written summary of changes needed
- Q&A session for concerns

### Daily Updates During Migration
- Slack channel: #repo-reorg
- Daily standup item
- Blocking issues escalated immediately

### Documentation
- Migration guide for developers
- Import mapping reference
- Troubleshooting guide

## Contingency Plans

### If Phase 2 Fails
- Keep using old common/ structure
- Focus on fixing issues before proceeding
- Maximum 1 week delay

### If Phase 3 Fails  
- MettagGrid stays in current structure
- Other packages can still migrate
- Revisit after other phases complete

### If Phase 4 Fails
- Most complex phase - may need splitting
- Consider intermediate structure
- Maximum 2 week extension

### If Phase 5 Fails
- Keep compatibility layers longer
- Gradual deprecation over 2-3 months
- May need to maintain dual structures

## Post-Migration Tasks

1. **Update External Documentation**
   - GitHub README
   - Package documentation
   - Installation guides

2. **Optimize New Structure**
   - Remove unnecessary dependencies
   - Consolidate duplicate code
   - Performance optimization

3. **Team Training**
   - New structure walkthrough
   - Best practices guide
   - Q&A sessions

4. **Monitor for Issues**
   - Track import errors
   - Monitor CI/CD success rate
   - Gather developer feedback

## Appendix: Technical Details

### Namespace Package Configuration

Each package will use standard Python namespace packages:

```python
# cogworks/metta/cogworks/__init__.py
__version__ = "0.1.0"
__all__ = ["rl", "agent", "eval", "sim", "sweep", "mapgen"]

# Lazy imports to improve startup time
def __getattr__(name):
    if name == "rl":
        from . import rl
        return rl
    elif name == "agent":
        from . import agent
        return agent
    # ... etc
```

### Import Linting Rule

**.flake8 configuration:**
```ini
[flake8]
# Custom rule to enforce metta.* imports
import-order-style = custom
application-import-names = metta
import-restrictions = 
    cogworks:metta.cogworks,
    mettagrid:metta.mettagrid,
    common:metta.common,
    backend_shared:metta.backend_shared
```

### Testing Strategy

**Test Organization:**
```
tests/
├── unit/           # Fast, isolated tests
│   ├── cogworks/
│   ├── mettagrid/
│   └── common/
├── integration/    # Cross-package tests
└── e2e/           # End-to-end workflows
```

This plan provides a structured, low-risk approach to reorganizing the repository while maintaining continuous operation and providing clear rollback points at each phase.