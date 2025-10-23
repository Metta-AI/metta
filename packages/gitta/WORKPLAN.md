# WORKPLAN: Add Optional AWS Secrets Manager Support to Gitta

## Problem Statement

Gitta currently requires API keys and tokens to be provided via environment variables (e.g., `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`). In production AWS environments, it would be more secure and convenient to fetch these secrets from AWS Secrets Manager, following the pattern already used in the `softmax` package.

## Goals

1. **Optional AWS Integration**: Add AWS Secrets Manager fallback without making boto3 a required dependency
2. **Consistent Pattern**: Follow the existing pattern from `softmax/src/softmax/aws/secrets_manager.py`
3. **Backward Compatible**: Existing functionality must work exactly as before
4. **Production Ready**: Support typical AWS deployment scenarios

## Non-Goals

- Making AWS mandatory for gitta usage
- Changing the standalone package philosophy
- Supporting other secret managers (HashiCorp Vault, etc.)

## Current State

### Secrets Used in Gitta

**In `split.py`:**
- `ANTHROPIC_API_KEY` (required) - AI PR splitting
- `GITHUB_TOKEN` (optional) - GitHub PR creation

**In `github.py`:**
- `GITHUB_TOKEN` - Used in 3 functions:
  - `github_client()` - optional
  - `create_pr()` - required
  - `post_commit_status()` - required

### Current Implementation

All secrets are fetched directly from environment variables:
```python
github_token = token or os.environ.get("GITHUB_TOKEN")
api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
```

## Proposed Solution

### Architecture

**Fallback Order:**
1. Explicit parameter (if provided to function)
2. Environment variable (if set)
3. AWS Secrets Manager (if boto3 available)
4. Fail or return None (depending on `required` flag)

**AWS Secret Names:**
- `GITHUB_TOKEN` → `github/token`
- `ANTHROPIC_API_KEY` → `anthropic/api-key`

### Implementation Plan

#### Phase 1: Core Infrastructure ✅ COMPLETE

**1.1 Create `gitta/secrets.py` module**
- [x] Implement `get_secret()` function with AWS fallback
- [x] Add caching with 1-hour TTL (matching softmax pattern)
- [x] Implement helper functions: `get_github_token()`, `get_anthropic_api_key()`
- [x] Handle boto3 not available gracefully
- [x] Add comprehensive docstrings

**1.2 Update `pyproject.toml`**
- [x] Add optional dependency group: `aws = ["boto3>=1.26.0"]`
- [x] Update description to mention AWS support

#### Phase 2: Integration ✅ COMPLETE (github.py)

**2.1 Update `split.py`**
- [ ] Import `get_secret()` from `gitta.secrets`
- [ ] Replace `os.environ.get("ANTHROPIC_API_KEY")` with `get_secret()`
- [ ] Replace `os.environ.get("GITHUB_TOKEN")` with `get_secret()`
- [ ] Update `_get_anthropic_client()` method
- [ ] Update `__init__()` method

_Note: split.py does not exist in main branch yet (it's in PR #3254). This will be updated after that PR merges or in a follow-up PR to rebase this work on top of that branch._

**2.2 Update `github.py`**
- [x] Import `get_github_token()` from `gitta.secrets`
- [x] Update `github_client()` function
- [x] Update `create_pr()` function
- [x] Update `post_commit_status()` function
- [x] Remove unused `os` import

**2.3 Update `__init__.py`**
- [x] Export `get_secret`, `get_github_token`, `get_anthropic_api_key`
- [x] Add to `__all__`

#### Phase 3: Testing ✅ COMPLETE

**3.1 Unit Tests**
- [x] Create `tests/test_secrets.py`
- [x] Test environment variable fallback
- [x] Test AWS fallback (with mocked boto3)
- [x] Test boto3 not available scenario
- [x] Test caching behavior
- [x] Test required vs optional secrets
- [x] Test secret not found errors

**3.2 Integration Tests**
- [x] Test `github.py` functions with AWS secrets
- [x] Test graceful degradation when AWS unavailable
- [ ] Test `split.py` with AWS secrets (deferred - split.py not in main branch yet)

#### Phase 4: Documentation

**4.1 Update README.md**
- [ ] Add "AWS Secrets Manager Integration" section
- [ ] Document optional dependency installation
- [ ] Add usage examples
- [ ] Update Configuration section
- [ ] Add troubleshooting for AWS issues

**4.2 Update split.py docstrings**
- [ ] Document AWS fallback in class docstring
- [ ] Update CLI help text

**4.3 Create migration guide**
- [ ] Document how to set up AWS secrets
- [ ] Provide AWS secret name mappings
- [ ] Add IAM policy examples

#### Phase 5: Verification

**5.1 Local Testing**
- [ ] Run all tests with pytest
- [ ] Test without boto3 installed
- [ ] Test with boto3 but no AWS credentials
- [ ] Test with mocked AWS Secrets Manager

**5.2 Code Quality**
- [ ] Run ruff linting
- [ ] Run ruff formatting
- [ ] Check test coverage
- [ ] Review for type safety

## Success Criteria

- [ ] All existing tests pass
- [ ] New tests achieve >80% coverage of `secrets.py`
- [ ] Gitta works without boto3 installed
- [ ] Gitta works with boto3 but without AWS credentials
- [ ] Gitta successfully fetches from AWS Secrets Manager when configured
- [ ] Documentation is clear and comprehensive
- [ ] No breaking changes to existing API

## Dependencies

**New optional dependency:**
- `boto3>=1.26.0` (only for AWS support)

**Pattern reference:**
- `softmax/src/softmax/aws/secrets_manager.py`

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| boto3 import adds overhead | Low | Only import when needed, graceful fallback |
| AWS credentials not configured | Medium | Clear error messages, documentation |
| Secret name conflicts | Low | Use consistent naming convention |
| Breaking backward compatibility | High | Thorough testing, no API changes |
| Cache invalidation issues | Medium | Reasonable 1-hour TTL, provide clear_cache() |

## Timeline Estimate

- Phase 1 (Infrastructure): 1-2 hours
- Phase 2 (Integration): 1-2 hours
- Phase 3 (Testing): 2-3 hours
- Phase 4 (Documentation): 1 hour
- Phase 5 (Verification): 1 hour

**Total**: 6-9 hours of work

## Related Work

- PR #3254: Add PR splitting capabilities (base for this work)
- `softmax/src/softmax/aws/secrets_manager.py`: Pattern to follow
- `softmax/src/softmax/dashboard/metrics.py`: Example usage

## Open Questions

1. Should we support other AWS regions beyond us-east-1?
   - **Decision**: Yes, respect AWS_REGION env var, default to us-east-1

2. Should we support custom secret name mappings?
   - **Decision**: Not in this PR, can be added later if needed

3. Should we add support for AWS SSM Parameter Store as well?
   - **Decision**: No, Secrets Manager is sufficient for now

4. How should we handle AWS throttling/rate limits?
   - **Decision**: Let boto3 handle retries, cache aggressively

## Implementation Updates

### Phase 1 Complete - 2025-10-23

**Summary**: Successfully implemented core AWS Secrets Manager integration with optional boto3 dependency.

**Files Created:**
- `packages/gitta/src/gitta/secrets.py` (227 lines) - Core secrets management module

**Files Modified:**
- `packages/gitta/pyproject.toml` - Added `[project.optional-dependencies]` with boto3
- `packages/gitta/src/gitta/__init__.py` - Exported secrets functions

**Key Features Implemented:**
- ✅ `get_secret()` - Generic secret retrieval with env var → AWS fallback
- ✅ `get_github_token()` - GitHub token helper with fallback to `github/token`
- ✅ `get_anthropic_api_key()` - Anthropic API key helper with fallback to `anthropic/api-key`
- ✅ `clear_cache()` - Manual cache clearing for testing/rotation
- ✅ 1-hour TTL caching (matching softmax pattern)
- ✅ Graceful degradation when boto3 not available
- ✅ AWS_REGION env var support (default: us-east-1)
- ✅ Comprehensive docstrings with examples
- ✅ Type hints with overloads for `required` parameter

**Testing:**
- ✅ Basic functionality tests passed
- ✅ Environment variable fallback works
- ✅ Missing secrets handled correctly
- ✅ Imports work without boto3 installed
- ✅ Ruff linting and formatting passed

**Next Steps:**
- Phase 2: Integration with split.py and github.py
- Phase 3: Comprehensive unit tests

### Phase 2 Complete (github.py) - 2025-10-23

**Summary**: Successfully integrated AWS secrets fallback into github.py functions.

**Files Modified:**
- `packages/gitta/src/gitta/github.py` - Replaced `os.environ.get("GITHUB_TOKEN")` with `get_github_token()`

**Changes:**
- ✅ Imported `get_github_token()` from `.secrets`
- ✅ Updated `github_client()` - uses `get_github_token(required=False)` for optional token
- ✅ Updated `create_pr()` - uses `get_github_token(required=True)` for required token
- ✅ Updated `post_commit_status()` - uses `get_github_token(required=True)` for required token
- ✅ Removed manual ValueError checks (handled by get_github_token)
- ✅ Removed unused `os` import
- ✅ Updated docstrings to mention AWS fallback

**Testing:**
- ✅ Imports work correctly
- ✅ Functions work without token set (when optional)
- ✅ Functions use environment variable when set
- ✅ Ruff linting and formatting passed

**Note on split.py:**
The split.py file doesn't exist in the main branch yet - it's part of PR #3254. Once that PR merges, we can either:
1. Create a follow-up PR to add AWS support to split.py
2. Rebase this branch on top of the PR branch and add split.py support here

**Next Steps:**
- Phase 3: Comprehensive unit tests with mocked boto3

### Phase 3 Complete - 2025-10-23

**Summary**: Created comprehensive test suite for secrets module with excellent coverage.

**Files Created:**
- `packages/gitta/tests/test_secrets.py` (335 lines) - 25 comprehensive tests

**Files Modified:**
- `packages/gitta/pyproject.toml` - Added boto3 and botocore to dev dependencies

**Test Coverage:**
- **25 tests** organized into 5 test classes
- **90% code coverage** of secrets.py
- All tests passing ✅

**Test Classes:**
1. **TestEnvironmentVariableFallback** (4 tests)
   - Environment variable retrieval
   - Env var precedence over AWS
   - Missing secrets (required vs optional)

2. **TestAWSSecretsManagerFallback** (6 tests)
   - AWS secret retrieval
   - Default secret name conversion (ENV_VAR → env/var)
   - Custom AWS region support
   - Secret not found handling
   - Whitespace stripping

3. **TestBoto3NotAvailable** (3 tests)
   - Graceful degradation without boto3
   - Error handling when boto3 unavailable
   - Env var fallback still works

4. **TestCaching** (4 tests)
   - Cache hits for AWS secrets
   - Cache expiry after 1 hour
   - Cache misses are cached
   - Manual cache clearing

5. **TestHelperFunctions** (8 tests)
   - `get_github_token()` from env and AWS
   - `get_anthropic_api_key()` from env and AWS
   - Required vs optional behavior

**Uncovered Lines (10%):**
- Lines 37-38: ImportError exception handler in _has_boto3 (edge case)
- Lines 138-143: Some error handling paths (transient errors)

**Testing Approach:**
- Mocked boto3 using `patch.dict(sys.modules)` to simulate AWS
- Mocked botocore.exceptions.ClientError for not-found scenarios
- Clean environment fixture to ensure test isolation
- Time mocking for cache expiry tests

**Next Steps:**
- Phase 4: Documentation updates
