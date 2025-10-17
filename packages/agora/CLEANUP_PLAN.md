# Agora Migration - Cleanup Plan

## Overview

After successful migration and validation, these files in `metta/cogworks/curriculum/` can be safely removed. The functionality has been migrated to the `agora` package.

---

## 🗑️ Files to Remove (After Testing)

### Core Curriculum Files (128K total)

| File | Size | Status | Migrated To |
|------|------|--------|-------------|
| `curriculum.py` | 20K | ✅ Redundant | `agora/curriculum.py` |
| `task_generator.py` | 17K | ✅ Redundant | `agora/generators/*.py` (4 files) |
| `task_tracker.py` | 18K | ✅ Redundant | `agora/tracking/tracker.py` |
| `stats.py` | 17K | ✅ Redundant | `agora/tracking/stats.py` |
| `lp_scorers.py` | 19K | ✅ Redundant | `agora/algorithms/scorers.py` |
| `learning_progress_algorithm.py` | 15K | ✅ Redundant | `agora/algorithms/learning_progress.py` |
| `shared_memory_backend.py` | 10K | ✅ Redundant | `agora/tracking/memory.py` |
| `curriculum_env.py` | 8.9K | ✅ Redundant | `agora/wrappers/puffer.py` |
| `demo.py` | 1.4K | ✅ Redundant | Example code, can remove |
| `structure.md` | <1K | ✅ Redundant | Documentation, can remove |

**Total redundant code**: ~128KB / 2,769 lines

---

## 📝 Files to Keep/Modify

### 1. `__init__.py` (3.7K) - **KEEP & MODIFY**

**Current Status**: ✅ Already updated with backward compatibility shim

**Purpose**: Provides deprecation warnings and re-exports from `agora`

**Action**: **KEEP** - This is the backward compatibility layer
- Shows focused deprecation warning when imported
- Re-exports all agora classes for backward compatibility
- Provides MettaGrid-specific helper functions

**Future**: Remove after all imports are updated (6-12 months)

---

## 🔄 Cleanup Timeline

### Phase 1: Immediate (After Tests Pass)
**Action**: None - Keep all files for backward compatibility

**Rationale**:
- Ensure smooth transition period
- Allow time for external projects to update
- Verify all functionality works via shim

### Phase 2: After 2-4 weeks (All Internal Updates Complete)
**Action**: Archive old files (don't delete yet)

```bash
# Create archive directory
mkdir -p metta/cogworks/curriculum/.archived

# Move redundant files
mv metta/cogworks/curriculum/{curriculum,task_generator,task_tracker,stats,lp_scorers,learning_progress_algorithm,shared_memory_backend,curriculum_env,demo,structure.md}.py metta/cogworks/curriculum/.archived/

# Update .gitignore
echo "metta/cogworks/curriculum/.archived/" >> .gitignore
```

**Keep only**: `__init__.py` (shim)

### Phase 3: After 3-6 months (Deprecation Period Complete)
**Action**: Remove old module entirely

```bash
# Remove entire old curriculum directory
rm -rf metta/cogworks/curriculum/
```

**Update**: Any remaining imports (should be none by this point)

---

## 📊 Cleanup Impact Analysis

### Disk Space Savings
- **Current**: ~128KB in redundant files
- **After Phase 2**: Keep only 3.7KB shim
- **Savings**: ~124KB (97% reduction)

### File Count Reduction
- **Current**: 12 files in `metta/cogworks/curriculum/`
- **After Phase 2**: 1 file (`__init__.py`)
- **After Phase 3**: 0 files (entire directory removed)

### Import Complexity
- **Current**: Two valid import paths (`metta.cogworks.curriculum` and `agora`)
- **After Phase 2**: Two paths, but old one shows warning
- **After Phase 3**: One path (`agora` only)

---

## ✅ Validation Checklist

Before removing any files, verify:

- [ ] All tests pass using `agora` imports
- [ ] No import errors in production code
- [ ] Training recipes work correctly
- [ ] Checkpointing/resuming works
- [ ] External projects notified of deprecation
- [ ] Documentation updated

---

## 🔍 How to Find Remaining Old Imports

```bash
# Search for old import pattern
grep -r "from metta.cogworks.curriculum import" --include="*.py" . 2>/dev/null | grep -v "__pycache__" | grep -v ".archived"

# Search for module-level imports
grep -r "import metta.cogworks.curriculum" --include="*.py" . 2>/dev/null | grep -v "__pycache__" | grep -v ".archived"

# Count remaining uses
grep -r "metta.cogworks.curriculum" --include="*.py" . 2>/dev/null | grep -v "__pycache__" | grep -v ".archived" | wc -l
```

Expected after migration: Only in `__init__.py` shim and test files

---

## 🚨 Files That Should NOT Be Removed

These files are NOT redundant:

### In `metta/cogworks/`
- `__init__.py` - Module marker
- Other cogworks modules unrelated to curriculum

### Anywhere else
- Test files in `tests/cogworks/curriculum/` - Keep for validation
- Recipe files in `experiments/recipes/` - Already updated imports

---

## 📋 Detailed Removal Instructions

### Step 1: Verify Tests Pass
```bash
cd /Users/bullm/Documents/GitHub/metta

# Run full test suite
uv run pytest tests/cogworks/curriculum/ -v

# Run integration tests
uv run pytest tests/integration/ -v -k curriculum

# Test training
timeout 30s uv run ./tools/run.py experiments.recipes.arena.train run=test_cleanup
```

### Step 2: Find Any Remaining Direct Imports
```bash
# This should only show the __init__.py shim
grep -r "from metta.cogworks.curriculum.curriculum import" --include="*.py" .
grep -r "from metta.cogworks.curriculum.task_generator import" --include="*.py" .
grep -r "from metta.cogworks.curriculum.task_tracker import" --include="*.py" .
```

### Step 3: Archive Old Files (After 2-4 Weeks)
```bash
# Create archive
mkdir -p metta/cogworks/curriculum/.archived
git add metta/cogworks/curriculum/.archived/.gitkeep

# Move files (keeping __init__.py)
for file in curriculum task_generator task_tracker stats lp_scorers learning_progress_algorithm shared_memory_backend curriculum_env demo structure.md; do
    if [ -f "metta/cogworks/curriculum/${file}.py" ] || [ -f "metta/cogworks/curriculum/${file}.md" ]; then
        git mv "metta/cogworks/curriculum/${file}."* "metta/cogworks/curriculum/.archived/"
    fi
done

# Commit
git add -A
git commit -m "refactor: archive old curriculum implementation (migrated to agora package)"
```

### Step 4: Complete Removal (After 3-6 Months)
```bash
# Remove entire old curriculum directory
git rm -r metta/cogworks/curriculum/

# Commit
git commit -m "refactor: remove deprecated curriculum module (fully migrated to agora)"
```

---

## 🎯 Success Metrics

After cleanup is complete:

✅ **Code Duplication**: Eliminated (0 redundant files)
✅ **Import Paths**: Simplified (1 canonical path: `agora`)
✅ **Package Independence**: `agora` is standalone
✅ **Test Coverage**: Maintained at same level
✅ **Functionality**: No regressions

---

## 📚 Related Documents

- **Migration Summary**: `packages/agora/MIGRATION_SUMMARY.md`
- **Testing Guide**: `packages/agora/READY_FOR_TESTING.md`
- **Implementation Progress**: `packages/agora/IMPLEMENTATION_PROGRESS.md`
- **Package Documentation**: `packages/agora/README.md`

---

## ⚠️ Important Notes

1. **Don't rush cleanup** - The shim ensures backward compatibility
2. **Communicate changes** - Notify team members of deprecation
3. **Monitor logs** - Watch for deprecation warnings in production
4. **Test thoroughly** - Verify each phase before proceeding
5. **Keep backups** - Git history preserves everything

---

**Current Status**: Migration complete, all files redundant but kept for compatibility
**Recommended Action**: Wait for test validation before any cleanup
**Timeline**: 3-6 month deprecation period recommended

