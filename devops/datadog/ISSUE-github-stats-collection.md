# Fix GitHub Code Statistics Collection

**Status**: 🐛 Bug
**Priority**: Medium
**Created**: 2025-10-22
**Branch**: `robb/1022-datadog`
**Related**: [ISSUE-migrate-github-collector.md](ISSUE-migrate-github-collector.md)

## Problem

Three GitHub code statistics metrics always return 0:
- `github.code.lines_added_7d`: 0 (should be thousands)
- `github.code.lines_deleted_7d`: 0 (should be thousands)
- `github.code.files_changed_7d`: 0 (should be hundreds)

## Root Cause

The GitHub `/repos/{owner}/{repo}/commits` list endpoint doesn't include the `stats` field in its response.

**Current Code** (`softmax/src/softmax/dashboard/metrics.py`):
```python
@system_health_metric(metric_key="code.lines_added_7d")
def get_lines_added_7d() -> int:
    """Total lines of code added in the last 7 days."""
    commits = get_commits(repo=repo, branch="main", since=since, ...)

    total_additions = 0
    for commit in commits:
        stats = commit.get("stats", {})  # ❌ stats field doesn't exist
        total_additions += stats.get("additions", 0)

    return total_additions  # Always returns 0
```

## Investigation Results

### Endpoint Comparison

| Endpoint | Has Stats? | Reliability | Cost |
|----------|-----------|-------------|------|
| `/repos/{owner}/{repo}/commits` (list) | ❌ No | N/A | 1 API call |
| `/repos/{owner}/{repo}/commits/{sha}` (individual) | ✅ Yes | High | 1 call per commit |
| `/repos/{owner}/{repo}/stats/code_frequency` | ✅ Yes | Low (202) | 1 API call |
| `/repos/{owner}/{repo}/stats/commit_activity` | ✅ Yes | Low (202) | 1 API call |

### Test Results

**Individual Commit Fetch** (works):
```bash
GET /repos/Metta-AI/metta/commits/7eaa1d2e
```
```json
{
  "sha": "7eaa1d2e...",
  "stats": {
    "total": 180,
    "additions": 121,
    "deletions": 59
  },
  "files": [
    {
      "filename": "path/to/file.py",
      "additions": 10,
      "deletions": 5,
      "changes": 15
    }
  ]
}
```

**Statistics API** (unreliable):
```bash
GET /repos/Metta-AI/metta/stats/code_frequency
```
```
Status: 202 Accepted
{
  "message": "Computing statistics..."
}
```
- Returns 202 "Computing" status
- Can take minutes/hours to compute
- Not suitable for real-time collection

## Fix Options

### Option A: Individual Commit Fetches ✅ Recommended

**Approach**: Fetch each commit individually to get stats

**Pros**:
- Guaranteed to work
- Accurate per-commit stats
- Can get file-level details

**Cons**:
- More API calls (91 commits = 91 calls)
- Slightly slower collection

**API Cost Analysis**:
- Current: ~15 API calls per collection
- With fix: ~106 API calls per collection (+91)
- Rate limit: 5000 requests/hour
- Collection frequency: Every 15 minutes = 4x/hour
- Total usage: 424 calls/hour (well within limit)
- Remaining headroom: 4576 calls/hour (91% available)

**Implementation**:

```python
# Add to gitta library
def get_commit_with_stats(
    repo: str,
    sha: str,
    token: str | None = None,
    **headers: str,
) -> dict[str, Any]:
    """
    Get individual commit with stats.

    Args:
        repo: Repository in format "owner/repo"
        sha: Commit SHA
        token: GitHub token for authentication
        **headers: Additional headers

    Returns:
        Commit object with stats field
    """
    with github_client(repo, token=token, **headers) as client:
        resp = client.get(f"/commits/{sha}")
        resp.raise_for_status()
        return resp.json()

# Update metrics.py
@system_health_metric(metric_key="code.lines_added_7d")
def get_lines_added_7d() -> int:
    """Total lines of code added in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    # Get commit list (lightweight)
    commits = get_commits(
        repo=repo,
        branch="main",
        since=since,
        per_page=100,
        Authorization=_get_github_auth_header(),
    )

    # Fetch each commit with stats (heavier)
    total_additions = 0
    for commit in commits:
        try:
            full_commit = get_commit_with_stats(
                repo=repo,
                sha=commit["sha"],
                Authorization=_get_github_auth_header(),
            )
            stats = full_commit.get("stats", {})
            total_additions += stats.get("additions", 0)
        except Exception as e:
            logger.warning(f"Failed to get stats for commit {commit['sha'][:8]}: {e}")
            continue

    return total_additions
```

**Optimization**: Could cache commit stats if needed, but probably not necessary.

### Option B: PR-Based Stats (Alternative)

**Approach**: Calculate from merged PRs instead of commits

**Pros**:
- Fewer API calls (97 PRs vs 91 commits)
- More meaningful (measures reviewed/merged changes)
- Already fetching PRs for other metrics

**Cons**:
- Misses direct commits to main (hotfixes, reverts)
- Different semantic meaning than "commits"
- Need to ensure PRs have stats

**Implementation**:

```python
@system_health_metric(metric_key="code.lines_added_7d")
def get_lines_added_7d() -> int:
    """Total lines added from merged PRs in last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    prs = get_pull_requests(
        repo=repo,
        state="closed",
        since=since,
        Authorization=_get_github_auth_header(),
    )

    total_additions = 0
    for pr in prs:
        if pr.get("merged_at"):
            # PRs already have additions/deletions/changed_files
            total_additions += pr.get("additions", 0)

    return total_additions
```

**Note**: Need to verify PR objects include stats. If not, this option doesn't work.

### Option C: Leave As-Is (Not Recommended)

**Approach**: Document as known limitation

**Pros**:
- No code changes
- No API cost

**Cons**:
- Metrics are misleading (show 0)
- Missing useful visibility into code changes
- Incomplete metric set

**Only choose if**: Code stats are truly not important for monitoring.

## Recommendation

**Go with Option A: Individual Commit Fetches**

**Rationale**:
- We have plenty of API headroom (91% available after this fix)
- Guaranteed to work (tested and confirmed)
- Accurate representation of commit-level changes
- Straightforward implementation
- Consistent with current metric semantics

**Estimated Effort**: 2-3 hours
1. Add `get_commit_with_stats()` to gitta (30 min)
2. Update 3 metrics functions (1 hour)
3. Test and verify (1 hour)

## Tasks

- [ ] **Decision**: Confirm Option A (or choose B/C)
- [ ] Add `get_commit_with_stats()` function to `packages/gitta/src/gitta/github.py`
- [ ] Add export to `packages/gitta/src/gitta/__init__.py`
- [ ] Update `get_lines_added_7d()` in `softmax/src/softmax/dashboard/metrics.py`
- [ ] Update `get_lines_deleted_7d()` in `softmax/src/softmax/dashboard/metrics.py`
- [ ] Update `get_files_changed_7d()` to count unique files from commit stats
- [ ] Add error handling for individual commit fetches (log warnings, continue)
- [ ] Test locally: `metta softmax-system-health report`
  - Verify `code.lines_added_7d` > 0
  - Verify `code.lines_deleted_7d` > 0
  - Verify `code.files_changed_7d` > 0
  - Check values are reasonable (thousands of lines, hundreds of files)
- [ ] Test with push: `metta softmax-system-health report --push`
- [ ] Verify metrics in Datadog (check last 5 minutes)
- [ ] Monitor API rate limit usage (should be <500 calls/hour)
- [ ] Update `docs/CI_CD_METRICS.md` to remove "known limitation" note

## Test Cases

### Expected Values (Last 7 Days)

Based on recent activity:
- Commits: ~91
- Lines added: ~5,000-20,000 (varies widely)
- Lines deleted: ~2,000-10,000 (varies widely)
- Files changed: ~200-500 (varies)

### Edge Cases

1. **No commits in last 7 days**: Should return 0 (correctly)
2. **Commit fetch fails**: Log warning, skip commit, continue
3. **Commit has no stats**: Shouldn't happen, but handle gracefully (treat as 0)
4. **Very large commits**: Should handle correctly (no size limit on stats)

## Success Criteria

- [ ] `github.code.lines_added_7d` returns non-zero value
- [ ] `github.code.lines_deleted_7d` returns non-zero value
- [ ] `github.code.files_changed_7d` returns non-zero value
- [ ] Values are reasonable (within expected ranges)
- [ ] Metrics update in Datadog
- [ ] Collection completes within 2 minutes
- [ ] No API rate limit errors
- [ ] Documentation updated

## Related

- Main migration issue: [ISSUE-migrate-github-collector.md](ISSUE-migrate-github-collector.md)
- CI/CD Metrics: [docs/CI_CD_METRICS.md](docs/CI_CD_METRICS.md)
- GitHub Collector: [collectors/github/README.md](collectors/github/README.md)

## Notes

- This fix should be done as part of Phase 1 in the main migration
- Once fixed in current location, carry forward to new architecture
- Consider adding caching if API usage becomes a concern (unlikely)
- Monitor Datadog rate limit metrics after deployment
