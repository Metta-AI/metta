# SkyPilot Python Refactor - Follow-up Work

This document tracks remaining follow up work after PR #2384.

### 1. Security Threat Model Review

**From:** @berekuk's review **Priority:** Medium **Status:** Requires security review

**Issue:** Secrets leak in SkyPilot dashboard despite log redaction

**Current:**
- `to_filtered_dict()` redacts `github_pat` and `discord_webhook_url` from logs
- However, these secrets are still visible in SkyPilot job entrypoints in the dashboard UI
- Example: https://skypilot-api.softmax-research.net/dashboard/jobs/3255

**Questions:**
- What is our security threat model for SkyPilot jobs?
- Are published logs a greater risk than the SkyPilot dashboard?
- Should we use SkyPilot's secrets management instead of environment variables?
- Should we restrict dashboard access or redact secrets from the UI?

**Proposed:**
- Review and document security threat model for secrets in SkyPilot jobs
- Either use SkyPilot secrets management or ensure secrets don't appear in dashboard

**Effort:** 4-8 hours (requires discussion + implementation)

### 2. Client/Server Code Separation

**From:** @berekuk's review **Priority:** Low

**Current:** `job_helpers` (client-side) mixed with `skypilot/utils/` and `skypilot/launch/` (server-side)

**Proposed:** Clear directory structure (`client/` vs `server/`) to distinguish what runs locally vs remotely

**Effort:** 2-4 hours (mostly moving files, updating imports)

### 3. Environment Variable Management

**From:** @daveey's review **Priority:** Low **Status:** Requires team discussion

**Questions:**

- Should `configure_environment.py` use a .env file instead of reading env vars directly?
- Should it receive configuration from call site?
- What's the preferred configuration pattern?

**Current:** `configure_environment.py` reads environment variables directly

**Effort:** 4-6 hours (depends on chosen approach)
