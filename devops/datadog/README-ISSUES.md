# Datadog Collector Issues & Tasks

This directory contains local issue tracking files for Datadog collector development work.

## Work Plan

See **[WORKPLAN.md](WORKPLAN.md)** for the comprehensive execution strategy that coordinates all issues below.

**Key Insight**: Don't migrate architecture until we've validated the metric naming pattern and finalized metric selection.

**Timeline**: 13 days (2.5 weeks) across 4 phases
1. Foundation & Research (2-3 days)
2. Team Alignment & Quick Wins (3-4 days)
3. Deployment & Verification (2-3 days)
4. Documentation & PR Finalization (1-2 days)

## Active Issues

### [ISSUE-pr-review-feedback.md](ISSUE-pr-review-feedback.md)
**Status**: 📋 Planned | **Priority**: High

Address review feedback from PR #3384 before merging to main.

**Key Concerns**:
- Validate metric naming topology (`{service}.{category}.{metric_name}` vs tags)
- Replace Makefile with Typer CLI for better composability
- Review and refine metric selection with team input
- Document metric rationale and KPIs

**Reviewer**: @nishu-builder

---

### [ISSUE-migrate-github-collector.md](ISSUE-migrate-github-collector.md)
**Status**: 📋 Planned | **Priority**: High

Migrate GitHub metrics from current location (`softmax/dashboard/`) to the new modular collector architecture (`devops/datadog/collectors/github/`).

**Key Tasks**:
- Fix code statistics (lines added/deleted/changed)
- Add CI duration distribution metrics (min, p50, p75, p90, p95, max)
- Migrate to modular architecture with BaseCollector
- Update deployment to use new structure

**Metrics**: 23 total (17 implemented + 6 new duration stats)

---

### [ISSUE-github-stats-collection.md](ISSUE-github-stats-collection.md)
**Status**: 🐛 Bug | **Priority**: Medium

Fix GitHub code statistics metrics that currently return 0.

**Problem**: Three metrics always return 0:
- `github.code.lines_added_7d`
- `github.code.lines_deleted_7d`
- `github.code.files_changed_7d`

**Root Cause**: GitHub list commits endpoint doesn't include stats field

**Solution**: Fetch individual commits to get stats (91 API calls, well within rate limits)

---

## Why Local Markdown Files?

We track internal development tasks in local markdown files rather than GitHub issues because:

1. **Development Workflow**: These are internal implementation details, not user-facing issues
2. **Detailed Planning**: Markdown files allow more detailed technical documentation with code samples
3. **Version Control**: Issues evolve with the code on the same branch
4. **Less Noise**: Keeps GitHub issues focused on user-facing problems and features
5. **Easier Collaboration**: Team members working on the branch can see and update tasks directly

## Creating a New Issue

Use this template for new local issues:

```markdown
# Issue Title

**Status**: 📋 Planned | 🐛 Bug | ✅ Done
**Priority**: High | Medium | Low
**Created**: YYYY-MM-DD
**Branch**: branch-name

## Problem

Clear description of the problem or task.

## Background

Context and investigation results.

## Proposed Solution

How to fix it.

## Tasks

- [ ] Task 1
- [ ] Task 2

## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Related

- Other files or documentation
```

## Status Indicators

- 📋 **Planned** - Not started, ready to implement
- 🚧 **In Progress** - Currently being worked on
- 🐛 **Bug** - Something broken that needs fixing
- ✅ **Done** - Completed and verified
- 🔄 **Blocked** - Waiting on dependencies

## Priority Levels

- **High** - Critical for deployment, blocks other work
- **Medium** - Important but not blocking
- **Low** - Nice to have, can be deferred

## Workflow

1. **Create** issue file: `ISSUE-descriptive-name.md`
2. **Work** on tasks, update checkboxes
3. **Commit** changes: Document progress in git commits
4. **Complete** when all success criteria met
5. **Reference** in related PRs and documentation

## Current Branch

All current issues are being tracked on branch: `robb/1022-datadog`

## Questions?

See main documentation:
- [Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)
- [Main README](README.md)
