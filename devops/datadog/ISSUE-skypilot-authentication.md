# Issue: Optimize SkyPilot Collector with Remote API Authentication

**Date**: 2025-10-25
**Priority**: Low (optimization, not blocking)
**Status**: Proposed

## Problem Statement

The SkyPilot collector currently works but takes 38 seconds because it:
1. Attempts to connect to the remote SkyPilot API server at `https://skypilot-api.softmax-research.net`
2. Fails (no authentication)
3. Falls back to starting a local SkyPilot API server
4. Collects metrics successfully

**Current behavior:**
```
[2mFailed to connect to SkyPilot API server at http://127.0.0.1:46580. Starting a local server.[0m
[33mYour SkyPilot API server machine only has 2.0GB memory available...[0m
[0m[32m✓ SkyPilot API server started. [0m[0m
✅ skypilot: 30 metrics collected in 38.20s
```

**Desired behavior:**
- Authenticate with remote API server
- Collect metrics in ~5-10 seconds (faster, no local server startup)
- Reduce memory pressure (no local server process)

## Reference Implementation

The `monitoring` repo successfully authenticates with SkyPilot:

**File:** `monitoring/skypilot_monitor/initialize_skypilot_sdk.py`
```python
async def initialize_skypilot_sdk(config) -> bool:
    """Authenticate with SkyPilot API server once."""
    token = config.SKYPILOT_API_KEY

    cmd = ["sky", "api", "login", "-e", "https://skypilot-api.softmax-research.net", "--token", token]
    result = await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        logger.info("✅ SkyPilot authentication successful!")
        return True
    else:
        raise ConnectionError(...)
```

**Key components:**
1. `SKYPILOT_API_KEY` environment variable
2. `sky api login` command with endpoint and token
3. `/root/.sky` volume mount (emptyDir) for state persistence
4. 4GB memory limit (vs our 2GB)

## Proposed Solution

### Architecture

**Defensive Authentication Pattern:**
- Try to authenticate at startup
- If authentication fails, continue with fallback (current behavior)
- Don't break the job if SkyPilot auth is unavailable
- Log clear warnings for debugging

### Implementation Approach

**Option A: Startup Authentication (Recommended)**

Add authentication at the beginning of `run_all_collectors.py`:

```python
def initialize_skypilot_auth() -> bool:
    """
    Attempt to authenticate with SkyPilot API server.

    Returns True if successful, False if authentication unavailable.
    Does not raise exceptions - logs warnings instead.
    """
    api_key = os.environ.get("SKYPILOT_API_KEY")

    if not api_key:
        logger.warning("SKYPILOT_API_KEY not set - will use local API server")
        return False

    try:
        cmd = [
            "sky", "api", "login",
            "-e", "https://skypilot-api.softmax-research.net",
            "--token", api_key
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # Short timeout for auth
        )

        if result.returncode == 0:
            logger.info("✅ SkyPilot API authentication successful")
            return True
        else:
            logger.warning(f"SkyPilot auth failed (code {result.returncode}): {result.stderr}")
            logger.warning("Falling back to local API server")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("SkyPilot auth timed out - falling back to local API server")
        return False
    except Exception as e:
        logger.warning(f"SkyPilot auth error: {e} - falling back to local API server")
        return False

def main():
    """Run all collectors and report results."""
    # ... existing setup ...

    # Try to authenticate with SkyPilot (optional optimization)
    initialize_skypilot_auth()

    # Continue with collectors as normal
    for name in COLLECTORS:
        # ... existing collector loop ...
```

**Benefits:**
- ✅ Defensive: Falls back gracefully if auth unavailable
- ✅ DRY: Authenticates once per job (not per collector)
- ✅ Simple: Minimal code changes
- ✅ Backward compatible: Works without SKYPILOT_API_KEY

**Option B: Collector-Level Authentication**

Add authentication in the SkyPilot collector's `__init__`:

```python
class SkypilotCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self._authenticated = False
        self._try_authenticate()

    def _try_authenticate(self):
        """Try to authenticate with remote API server."""
        # Same logic as Option A
        pass
```

**Downsides:**
- ❌ Authenticates every 15 minutes (wasteful)
- ❌ Tight coupling to one collector
- ❌ Not reusable if we add more SkyPilot-dependent code

**Option C: Docker Entrypoint**

Add authentication in container startup script:

```bash
#!/bin/bash
# Authenticate with SkyPilot if key available
if [ -n "$SKYPILOT_API_KEY" ]; then
    sky api login -e https://skypilot-api.softmax-research.net --token "$SKYPILOT_API_KEY" || true
fi

# Run collectors
python /app/devops/datadog/scripts/run_all_collectors.py
```

**Downsides:**
- ❌ Adds complexity to container lifecycle
- ❌ Harder to test locally
- ❌ Auth runs even if SkyPilot collector is disabled

### Infrastructure Changes

#### 1. Helm Chart Updates

**File:** `devops/charts/dashboard-cronjob/templates/cronjob.yaml`

Add environment variable:
```yaml
env:
  # ... existing env vars ...

  # SkyPilot authentication (optional optimization)
  - name: SKYPILOT_API_KEY
    valueFrom:
      secretKeyRef:
        name: softmax-secrets  # Or dashboard-cronjob-secrets
        key: skypilot-api-key
        optional: true  # Don't fail if secret missing
```

Add volume mount:
```yaml
volumeMounts:
  - name: sky-config
    mountPath: /root/.sky

volumes:
  - name: sky-config
    emptyDir: {}  # Persists during pod lifetime
```

#### 2. AWS Secrets Manager

Add new secret:
```bash
aws secretsmanager create-secret \
    --name skypilot/api-key \
    --description "SkyPilot API authentication token" \
    --secret-string "YOUR_API_KEY_HERE" \
    --region us-east-1
```

**Note**: The existing IAM role policy for the dashboard-cronjob service account already allows `secretsmanager:GetSecretValue` on `skypilot/api-key*` via IRSA. No policy update needed.

#### 3. Secret Access Code

The existing `get_credential()` function in `devops/datadog/scripts/run_collector.py` already implements the correct pattern:

```python
# In run_all_collectors.py or similar
from devops.datadog.scripts.run_collector import get_credential

def initialize_skypilot_auth() -> bool:
    """Try to authenticate with SkyPilot API server."""
    # Uses existing get_credential() which checks env var then Secrets Manager
    api_key = get_credential("SKYPILOT_API_KEY", "skypilot/api-key", required=False)

    if not api_key:
        logger.warning("SKYPILOT_API_KEY not set - will use local API server")
        return False

    # ... rest of authentication logic
```

**Pattern**: The `get_credential()` helper already:
- Checks environment variable first (`SKYPILOT_API_KEY`)
- Falls back to AWS Secrets Manager (`skypilot/api-key`)
- Returns None gracefully if not found (when `required=False`)
- Uses the existing `get_secretsmanager_secret()` utility with caching

### Testing Plan

#### Phase 1: Local Testing
```bash
# 1. Get SkyPilot API key from monitoring repo deployment
kubectl get secret softmax-monitor-secrets -n softmax-monitor -o json | \
    jq -r '.data["skypilot-api-key"]' | base64 -d

# 2. Test authentication locally
export SKYPILOT_API_KEY="<key-from-step-1>"
sky api login -e https://skypilot-api.softmax-research.net --token "$SKYPILOT_API_KEY"

# 3. Test collector performance
time uv run python devops/datadog/scripts/run_collector.py skypilot --verbose

# Expected: Should complete in <10s instead of 38s
```

#### Phase 2: Dev Environment Testing
```bash
# 1. Add secret to AWS Secrets Manager
aws secretsmanager create-secret --name skypilot/api-key --secret-string "$SKYPILOT_API_KEY"

# 2. Update Helm chart with env var and volume
# 3. Deploy to dev
cd devops/charts
helmfile apply -l name=dashboard-cronjob-dev

# 4. Trigger test job
kubectl create job --from=cronjob/dashboard-cronjob-dev-dashboard-cronjob test-auth-$(date +%s) -n monitoring

# 5. Check logs for authentication success
kubectl logs -n monitoring job/test-auth-<timestamp> | grep -i "skypilot"

# Expected:
# ✅ SkyPilot API authentication successful
# ✅ skypilot: 30 metrics collected in ~8s
```

#### Phase 3: Fallback Testing
```bash
# Verify graceful degradation when auth unavailable

# 1. Remove SKYPILOT_API_KEY from environment
# 2. Deploy and test
# 3. Verify logs show warning and fallback
# Expected:
# ⚠️  SKYPILOT_API_KEY not set - will use local API server
# ✅ skypilot: 30 metrics collected in ~38s (slower but works)
```

### Performance Expectations

**Before (current):**
- SkyPilot: 38.20s (starts local server)
- Total job: ~99s

**After (with auth):**
- SkyPilot: ~8-10s (uses remote API)
- Total job: ~70s (30% faster)

**Memory usage:**
- Before: 2GB pod + local SkyPilot server overhead
- After: 2GB pod, no local server

### Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| SkyPilot API server down | Collector fails | Graceful fallback to local server |
| Wrong API key | Auth fails | Same - fall back to local server |
| Secret not in Secrets Manager | Key unavailable | Optional secret - continues without auth |
| Breaking existing deployments | Production impact | Make all changes optional, test in dev first |
| Authentication adds latency | Slower startup | Cache auth in /root/.sky, authenticate once |

### Decision Criteria

**Implement now if:**
- ✅ We want faster collector runs (38s → 8s)
- ✅ We want to reduce memory pressure
- ✅ We have access to SKYPILOT_API_KEY
- ✅ We have time to test thoroughly in dev

**Defer if:**
- ❌ Current 38s performance is acceptable
- ❌ More urgent priorities exist
- ❌ Risk of breaking production outweighs benefit

### Recommendation

**Priority: Low - Nice to Have**

The timeout protection (multiprocessing) solved the critical issue - jobs no longer hang. SkyPilot authentication is an optimization that:
- Saves ~30 seconds per job (38s → 8s)
- Reduces memory pressure slightly
- Requires careful testing and secret management

**Suggested Timeline:**
1. **Immediate**: Monitor current dev deployment for 24-48 hours to verify timeout protection is stable
2. **Next sprint**: Implement SkyPilot authentication if time permits
3. **Validate**: Test in dev for 1 week before production

### Implementation Checklist

When we decide to implement:

**Phase 1: Secret Setup**
- [ ] Get SKYPILOT_API_KEY from monitoring repo deployment
  ```bash
  kubectl get secret softmax-monitor-secrets -n softmax-monitor -o json | \
    jq -r '.data["skypilot-api-key"]' | base64 -d
  ```
- [ ] Add secret to AWS Secrets Manager as `skypilot/api-key`
  ```bash
  aws secretsmanager create-secret \
    --name skypilot/api-key \
    --secret-string "<key>" \
    --region us-east-1
  ```
- [ ] Verify IAM role policy (should already allow `skypilot/api-key*` access via IRSA)

**Phase 2: Code Implementation**
- [ ] Implement `initialize_skypilot_auth()` in run_all_collectors.py using existing `get_credential()` pattern
- [ ] Add volume mount for /root/.sky in Helm chart (emptyDir)
- [ ] Test authentication locally with env var
- [ ] Test fallback behavior (without key) to ensure graceful degradation

**Phase 3: Deployment & Validation**
- [ ] Deploy to dev environment
- [ ] Verify <10s SkyPilot collection time (vs current 38s)
- [ ] Confirm fallback works if secret unavailable
- [ ] Monitor dev for 1 week for stability
- [ ] Deploy to production
- [ ] Document in DEPLOYMENT_GUIDE.md

## Related Files

- `devops/datadog/scripts/run_all_collectors.py` - Main orchestration (add `initialize_skypilot_auth()` here)
- `devops/datadog/scripts/run_collector.py` - Contains `get_credential()` helper for secret retrieval
- `devops/datadog/utils/secrets.py` - AWS Secrets Manager utilities (already implemented)
- `devops/datadog/collectors/skypilot/collector.py` - SkyPilot collector
- `devops/charts/dashboard-cronjob/templates/cronjob.yaml` - Kubernetes deployment (add volume mount)
- `monitoring/skypilot_monitor/initialize_skypilot_sdk.py` - Reference implementation from monitoring repo

## References

- Monitoring repo: `/Users/rwalters/GitHub/monitoring`
- SkyPilot API docs: https://skypilot-api.softmax-research.net (requires auth)
- Original timeout issue: `devops/datadog/ISSUE-skypilot-collector-timeout.md`
