# Codex Review Prompt for PR #4572

Please review PR #4572: "akshay/add webhook to observatory backend"

## Context
This PR migrates the GitHub-Asana webhook service from AWS Lambda to EKS, integrating it as a route in the Observatory backend following Slava's guidance. It also fixes assignee resolution to use Asana user GIDs instead of email addresses.

## Review Focus Areas

### 1. Architecture & Integration
- Verify the webhook service is properly integrated as a route in Observatory backend
- Check that it reuses existing infrastructure (Docker image, Helm chart, IRSA role)
- Ensure no unnecessary new infrastructure was created
- Confirm the route is accessible at `/webhooks/github`

### 2. Security
- Verify GitHub webhook signature verification is properly implemented
- Check that secrets are accessed via AWS Secrets Manager (not hardcoded)
- Ensure IRSA role has correct permissions for Secrets Manager access
- Verify error handling doesn't leak sensitive information

### 3. Configuration Management
- Check that environment variables are properly set via Helm `extra_args`
- Verify `USE_AWS_SECRETS` flag is correctly used
- Ensure fallback behavior for local development is safe
- Check that required GitHub variables (ASANA_WORKSPACE_GID, ASANA_PROJECT_GID) are documented

### 4. Code Quality
- Review error handling and logging
- Check for proper type hints and type safety
- Verify retry logic is appropriate
- Check for potential race conditions or concurrency issues
- Ensure proper cleanup of resources (HTTP clients, etc.)

### 5. Asana Integration
- Verify assignee resolution using GIDs (not emails) works correctly
- Check that stale email handling is properly addressed
- Ensure task creation/updates handle edge cases
- Verify roster project mapping logic

### 6. GitHub Integration
- Check PR description update logic
- Verify GitHub API error handling
- Ensure proper handling of webhook events (opened, closed, assigned, etc.)

### 7. Terraform Changes
- Review `devops/tf/eks/observatory.tf` changes
- Verify IAM policy is correctly scoped (not too permissive)
- Check that secrets policy ARNs are correct
- Ensure `role_policy_arns` structure is correct

### 8. Dependencies
- Verify all required dependencies are in `pyproject.toml`
- Check for any missing imports or dependencies
- Ensure version constraints are appropriate

### 9. Testing & Observability
- Check that metrics are properly instrumented
- Verify logging is appropriate (not too verbose, includes necessary context)
- Check for any missing error handling paths

### 10. Migration Concerns
- Verify no Lambda-specific code remains
- Check that all Lambda Terraform was removed (in previous PR)
- Ensure the code works in a long-running process (not just Lambda)

## Specific Files to Review
- `app_backend/src/metta/app_backend/github_webhook/routes.py` - Webhook endpoint
- `app_backend/src/metta/app_backend/github_webhook/asana_integration.py` - Asana API integration
- `app_backend/src/metta/app_backend/github_webhook/pr_handler.py` - PR event handling
- `app_backend/src/metta/app_backend/github_webhook/config.py` - Configuration
- `app_backend/src/metta/app_backend/server.py` - Router integration
- `devops/tf/eks/observatory.tf` - IAM permissions
- `.github/workflows/build-app-backend-image.yml` - CI/CD configuration
- `app_backend/pyproject.toml` - Dependencies

## Questions to Answer
1. Are there any security vulnerabilities?
2. Are there any potential bugs or edge cases not handled?
3. Is the error handling robust?
4. Are there any performance concerns?
5. Is the code maintainable and well-structured?
6. Are there any missing tests or test coverage gaps?
7. Are there any configuration issues that could cause problems in production?

Please provide specific, actionable feedback with code examples where helpful.

