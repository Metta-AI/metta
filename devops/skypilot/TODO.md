# SkyPilot Python Refactor - Follow-up Work

This document tracks remaining follow up work after PR #2384.

### 1. Client/Server Code Separation

**From:** @berekuk's review **Priority:** Low

**Current:** `job_helpers` (client-side) mixed with `skypilot/utils/` and `skypilot/launch/` (server-side)

**Proposed:** Clear directory structure (`client/` vs `server/`) to distinguish what runs locally vs remotely

**Effort:** 2-4 hours (mostly moving files, updating imports)

### 2. Environment Variable Management

**From:** @daveey's review **Priority:** Low **Status:** Requires team discussion

**Questions:**

- Should `configure_environment.py` use a .env file instead of reading env vars directly?
- Should it receive configuration from call site?
- What's the preferred configuration pattern?

**Current:** `configure_environment.py` reads environment variables directly

**Effort:** 4-6 hours (depends on chosen approach)
