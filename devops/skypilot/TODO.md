# SkyPilot Python Refactor - Follow-up Work

This document tracks the remaining low-priority improvements to be addressed in follow-up PRs after the initial Python refactor (PR #2384) is merged.

**Status:** 2 items remaining - all low priority
**Last Updated:** 2025-10-29

---

## Overview

The shell-to-Python refactor (PR #2384) successfully completed all high and medium priority items:
- ✅ 10/10 substantive issues resolved
- ✅ All technical improvements complete
- ✅ Zero blocking concerns
- ✅ 5/7 follow-up items completed during PR review

Remaining items are code organization improvements, UX enhancements, and design questions that require team discussion.

---

## Phase 1: Code Organization (1-2 PRs)

**Priority:** Low | **Effort:** ~4 hours | **Impact:** Developer Experience

### 1.1 Client/Server Code Separation
**From:** @berekuk's review

**Current State:**
- `job_helpers` (client-side code) mixed with `skypilot/utils/` and `skypilot/launch/` (server-side)
- No clear distinction between what runs locally vs remotely

**Proposed:**
- Clear directory structure: `client/` vs `server/` or similar convention
- Makes code organization more intuitive

**Effort:** 2-4 hours (mostly moving files, updating imports)

---

### 1.2 Rename Configuration Files ✅ COMPLETED
**From:** @berekuk's review

**Completed:** 2025-10-29

**Implementation:**
- Renamed `skypilot_run.yaml` to `job.yaml`
- Updated references in:
  - `devops/skypilot/launch.py`
  - `devops/skypilot/connect.py`
  - `devops/skypilot/README.md`

---

## Phase 2: Environment Configuration (1 PR)

**Priority:** Low | **Effort:** ~6 hours | **Impact:** Maintainability

### 2.1 Environment Variable Management
**From:** @daveey's review

**Questions to resolve:**
- Should `configure_environment.py` use a .env file instead of reading env vars directly?
- Should it receive configuration from call site rather than reading environment?
- What's the preferred configuration pattern for the team?

**Current State:**
- `configure_environment.py` reads environment variables directly
- Works but could be more explicit about configuration

**Next Steps:**
1. **Requires team discussion** about preferred approach
2. Implement chosen pattern
3. Update documentation

**Effort:** 4-6 hours (depends on chosen approach)

---

### 2.2 Make Configure Environment Executable ✅ COMPLETED
**From:** @daveey's review

**Completed:** 2025-10-29

**Implementation:**
- Added shebang: `#!/usr/bin/env -S uv run python3`
- Made file executable: `chmod +x configure_environment.py`
- Updated YAML files to call directly: `./devops/skypilot/utils/configure_environment.py`

---

## Phase 3: Design Clarifications (After Discussion)

**Priority:** Low | **Effort:** 2-6 hours | **Impact:** Code Quality

### 3.1 Restart Count Tracking ✅ COMPLETED
**From:** @daveey's review

**Completed:** 2025-10-29

**Decision:** Document rationale (Option A)

**Implementation:**
Added comprehensive docstring to `_setup_job_metadata()` explaining:
- Restart count needed for runtime monitoring and accumulated runtime tracking
- Persisted to shared storage (DATA_DIR) to survive job restarts
- Used by runtime monitors to make timeout and testing decisions
- SkyPilot's max_restarts_on_errors controls policy, our count tracks actual restarts

---

### 3.2 NCCL Test Necessity ✅ COMPLETED
**From:** @daveey's review
**Location:** `devops/skypilot/utils/nccl_tests.py`

**Completed:** 2025-10-29

**Decision:** Document value (Option A)

**Implementation:**
Added module-level docstring explaining:
- Validates GPU communication infrastructure before training starts
- Catches network configuration issues in cloud environments
- Detects GPU driver/firmware incompatibilities
- Identifies incorrect NCCL environment variable settings
- Catches hardware issues on specific instance types
- Saves hours of debugging when training mysteriously fails mid-run

---

## Phase 4: UX Improvements (1 PR)

**Priority:** Low | **Effort:** ~2 hours | **Impact:** Developer Experience

### 4.1 Launch Script Verbosity Control ✅ COMPLETED
**From:** @daveey's review
**Location:** `devops/skypilot/launch.py`

**Completed:** 2025-10-29

**Implementation:**
- Added `--verbose` flag to launch.py
- Modified `launch_task()` to accept `verbose` parameter
- Request IDs and log instructions only printed when `--verbose` is passed
- Default is now quiet mode for better batch operation UX

---

## Recommended Sequencing

### Follow-up PR #1: Code Organization
**Items:** 1.1, 1.2
**Effort:** ~4 hours
**Dependencies:** None
**Impact:** Better code organization

### Follow-up PR #2: Environment Config
**Items:** 2.1, 2.2
**Effort:** ~6 hours
**Dependencies:** Requires team discussion first
**Impact:** Cleaner configuration pattern

### Follow-up PR #3: Design Clarifications
**Items:** 3.1, 3.2
**Effort:** 2-6 hours (varies by decisions)
**Dependencies:** Requires team discussion first
**Impact:** Code clarity and maintenance

### Follow-up PR #4: UX Polish
**Items:** 4.1
**Effort:** ~2 hours
**Dependencies:** None
**Impact:** Better CLI experience

---

## Effort Summary

- **Total effort:** 15-22 hours across 3-4 PRs
- **Timeline:** 2-3 weeks if done incrementally
- **Required discussions:** 2
  1. Environment configuration approach (#2.1)
  2. Restart count and NCCL test strategy (#3.1, #3.2)
- **Risk:** Low - all items are improvements, not fixes

---

## Quick Reference

**Can start immediately (no discussion needed):**
- 🔄 Client/Server separation (#1.1)

**Completed during PR review:**
- ✅ Rename skypilot_run.yaml (#1.2) - COMPLETED
- ✅ Configure environment executable (#2.2) - COMPLETED
- ✅ Launch script verbosity (#4.1) - COMPLETED
- ✅ Restart count tracking (#3.1) - COMPLETED
- ✅ NCCL test necessity (#3.2) - COMPLETED

**Requires discussion first:**
- ⏸️ Environment variable management (#2.1)

---

## Notes

- All items are **low priority** - none are blocking or correctness issues
- Items emerged from code review feedback after core refactor was complete
- Can be addressed incrementally without impacting system functionality
- Consider combining related items into single PRs for efficiency
