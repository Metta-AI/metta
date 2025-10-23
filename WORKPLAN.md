# WORKPLAN: Implement `metta format` Command

**Branch:** `feature/metta-format-multi-file-support` **Created:** 2025-10-23 **Status:** In Progress

---

## Objective

Restore root Makefile's multi-format linting/formatting functionality through the `metta` CLI by implementing
`metta format` command.

**Context:** The root Makefile was removed in commit `68127b081d`, removing the ability to format multiple file types
(Python, JSON, Markdown, Shell, TOML, YAML, C++) with a single command.

---

## Implementation Phases

### Phase 1: Create `metta format` Command ✅ COMPLETE

**Goal:** Implement core formatting functionality

- [x] Create `.claude/tasks/metta-format-command.md` task plan
- [x] Create `metta/setup/tools/formatter.py` module
  - [x] Define FORMATTERS dict with all file type mappings
  - [x] Implement format_code() function
  - [x] Add --all, --type, --check flags
  - [x] Support C++ formatting if mettagrid Makefile exists
- [x] Register format command in `metta/setup/metta_cli.py`
- [x] Test basic functionality
  - [x] `metta format` (Python only)
  - [x] `metta format --all` (all types)
  - [x] `metta format --type json`
  - [x] `metta format --check`

### Phase 2: Testing & Verification ✅ COMPLETE

**Goal:** Ensure all formatters work correctly

- [x] Test each individual formatter:
  - [x] Python (ruff format)
  - [x] JSON (devops/tools/format_json.sh)
  - [x] Markdown (devops/tools/format_md.sh)
  - [x] Shell (devops/tools/format_sh.sh)
  - [x] TOML (devops/tools/format_toml.sh)
  - [x] YAML (devops/tools/format_yml.sh)
  - [x] C++ (not available in main branch - handled gracefully)
- [x] Run `metta lint` to verify no regressions
- [x] Test error handling and edge cases

### Phase 3: Documentation ✅ COMPLETE

**Goal:** Update all documentation to reflect new command

- [x] Update `CLAUDE.md`:
  - [x] Add `metta format` to Code Quality section
  - [x] Update common commands examples
  - [x] Show all available flags and options
- [x] Update `.cursor/commands.md`:
  - [x] Add Code Formatting section
  - [x] Show usage examples with different flags
  - [x] Document supported file types and aliases
- [x] Update WORKPLAN.md with final status

### Phase 4: Finalization ✅ COMPLETE

**Goal:** Polish and prepare for PR

- [x] Run `metta lint` check - all tests pass
- [x] Format all code: `metta format --all`
- [x] Review all changes
- [x] Update WORKPLAN with final status
- [ ] Prepare PR description (ready for user)

---

## Design Decisions

### 1. Module Organization

**Decision:** Create standalone `metta/setup/tools/formatter.py` and register as sub-app **Rationale:** Follows pattern
of `pytest` and `cpptest` tools, keeps code organized

### 2. Default Behavior

**Decision:** Default to Python-only formatting **Rationale:** Backward compatible, matches user expectations from
`metta lint`

### 3. C++ Support

**Decision:** Conditionally include C++ formatting if mettagrid Makefile exists **Rationale:** Main branch may remove
mettagrid Makefile, so check dynamically

### 4. Alias Support

**Decision:** Support aliases (md→markdown, sh→shell, yml→yaml) **Rationale:** Improves UX, matches common abbreviations

---

## Success Criteria

- [x] `metta/setup/tools/formatter.py` created
- [ ] `metta format` command registered in CLI
- [ ] Default behavior: format Python only
- [ ] `--all` flag: format all file types
- [ ] `--type` flag: format specific file type
- [ ] `--check` flag: verify without modifying
- [ ] All existing formatters work correctly
- [ ] Documentation updated
- [ ] All tests pass (`metta ci`)
- [ ] Code is properly formatted

---

## Related Files

**Implementation:**

- `metta/setup/tools/formatter.py` - Main formatter module
- `metta/setup/metta_cli.py` - CLI registration
- `.claude/tasks/metta-format-command.md` - Detailed task plan

**Formatter Scripts:**

- `devops/tools/format_json.sh`
- `devops/tools/format_md.sh`
- `devops/tools/format_sh.sh`
- `devops/tools/format_toml.sh`
- `devops/tools/format_yml.sh`
- `packages/mettagrid/Makefile` - C++ formatting (optional)

**Documentation:**

- `CLAUDE.md` - Development guide
- `.cursor/commands.md` - Quick test commands
- `/tmp/makefile-to-metta-cli-migration.md` - Full migration plan

---

## Progress Log

### 2025-10-23 - Initial Setup

- Created feature branch `feature/metta-format-multi-file-support`
- Created task plan at `.claude/tasks/metta-format-command.md`
- Created WORKPLAN.md for tracking
- Implemented `metta/setup/tools/formatter.py` with:
  - FORMATTERS dict for all file types
  - format_code() function with --all, --type, --check flags
  - Dynamic C++ support detection
  - Alias support for common abbreviations
- **Next:** Register command in metta_cli.py and test

### 2025-10-23 - Implementation Complete ✅

- Registered formatter app in `metta/setup/metta_cli.py`
- Fixed command structure using `invoke_without_command=True` pattern
- Tested all formatters successfully:
  - ✅ Python (ruff) - 963 files formatted
  - ✅ JSON - all files formatted
  - ✅ Markdown - all files formatted
  - ✅ Shell scripts - all files formatted
  - ✅ TOML - all files formatted
  - ✅ YAML - all files formatted
  - ✅ C++ - gracefully skipped (not available in main branch)
- Updated documentation:
  - ✅ CLAUDE.md: Added format command examples to Code Quality section
  - ✅ .cursor/commands.md: Added comprehensive Code Formatting section
- Ran `metta lint` - all checks pass
- **Status**: Implementation complete and ready for PR

---

## Notes

- Root Makefile was removed in commit `68127b081d`
- This feature restores the multi-format functionality via metta CLI
- Maintains backward compatibility (Python-only by default)
- C++ formatting is optional based on mettagrid Makefile existence
- See `/tmp/makefile-to-metta-cli-migration.md` for complete migration plan
