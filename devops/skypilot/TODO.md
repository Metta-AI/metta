# SkyPilot Python Refactor - Follow-up Work

This document tracks the remaining low-priority improvements to be addressed in follow-up PRs after the initial Python refactor (PR #2384) is merged.

**Status:** 7 items remaining - all low priority
**Last Updated:** 2025-10-23

---

## Overview

The shell-to-Python refactor (PR #2384) successfully completed all high and medium priority items:
- ✅ 10/10 substantive issues resolved
- ✅ All technical improvements complete
- ✅ Zero blocking concerns

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

### 1.2 Rename Configuration Files
**From:** @berekuk's review

**Current State:**
- `skypilot_run.yaml` - not immediately obvious what this configures

**Proposed:**
- Rename to `job.yaml` for clarity
- Update references in documentation and launch scripts

**Effort:** 1 hour

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

### 2.2 Make Configure Environment Executable
**From:** @daveey's review

**Current State:**
- Must call via `uv run python devops/skypilot/utils/configure_environment.py`

**Proposed:**
- Add shebang: `#!/usr/bin/env -S uv run --script`
- Make file executable: `chmod +x configure_environment.py`
- Call directly: `./devops/skypilot/utils/configure_environment.py`

**Effort:** 30 minutes

---

## Phase 3: Design Clarifications (After Discussion)

**Priority:** Low | **Effort:** 2-6 hours | **Impact:** Code Quality

### 3.1 Restart Count Tracking
**From:** @daveey's review

**Question:**
Why track restart count manually? SkyPilot has native restart limiting via `job_recovery.max_restarts_on_errors`.

**Current State:**
- `configure_environment.py` tracks restart count
- Unclear why this is needed vs SkyPilot's native mechanism

**Options:**
a) **Document rationale** - Explain why manual tracking is needed and keep current approach
b) **Switch to native** - Use SkyPilot's restart management exclusively
c) **Remove** - If not providing value beyond SkyPilot's feature

**Next Steps:**
1. **Requires team discussion** about requirements and rationale
2. Implement chosen approach

**Effort:** 2-4 hours (depending on decision)

---

### 3.2 NCCL Test Necessity
**From:** @daveey's review
**Location:** `devops/skypilot/utils/nccl_tests.py`

**Question:**
What's the failure mode these tests protect against? Has NCCL ever failed in practice?

**Current State:**
- NCCL tests run as part of job setup
- Unclear if they've caught real issues

**Options:**
a) **Document value** - Explain what failures these catch and when they're useful (2 hours)
b) **Remove** - If not catching real issues in practice (1 hour)

**Next Steps:**
1. **Requires team discussion** about testing strategy
2. Check if tests have caught issues historically
3. Document rationale or remove

**Effort:** 1-2 hours

---

## Phase 4: UX Improvements (1 PR)

**Priority:** Low | **Effort:** ~2 hours | **Impact:** Developer Experience

### 4.1 Launch Script Verbosity Control
**From:** @daveey's review
**Location:** `devops/skypilot/launch.py`

**Issue:**
- Script always prints cluster ID
- Interrupts workflow when pasting multiple commands

**Proposed Solution:**
- Add `--verbose` or `--print-id` flag
- Default to quiet mode
- Print cluster ID only when flag is set

**Benefit:** Better UX for batch operations

**Effort:** 1-2 hours

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
- ✅ Client/Server separation (#1.1)
- ✅ Rename skypilot_run.yaml (#1.2)
- ✅ Configure environment executable (#2.2)
- ✅ Launch script verbosity (#4.1)

**Requires discussion first:**
- ⏸️ Environment variable management (#2.1)
- ⏸️ Restart count tracking (#3.1)
- ⏸️ NCCL test necessity (#3.2)

---

## Notes

- All items are **low priority** - none are blocking or correctness issues
- Items emerged from code review feedback after core refactor was complete
- Can be addressed incrementally without impacting system functionality
- Consider combining related items into single PRs for efficiency
