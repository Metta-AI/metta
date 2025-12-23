# Testing Webhook Service with Real GitHub PR

## Prerequisites

1. Set environment variables:
```bash
export ASANA_PAT='your-asana-pat'
export ASANA_WORKSPACE_GID='your-workspace-gid'
export ASANA_PROJECT_GID='your-project-gid'
export GITHUB_WEBHOOK_SECRET='your-webhook-secret'  # Optional for local testing
```

2. Install ngrok (to expose local service to GitHub):
```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/
```

## Steps

### 1. Start the webhook service locally

```bash
cd webhook_service
./test_with_real_pr.sh
```

The service will run on `http://localhost:8000`

### 2. Expose local service to internet (using ngrok)

In a **new terminal**:

```bash
ngrok http 8000
```

This will give you a public URL like: `https://abc123.ngrok.io`

### 3. Configure GitHub webhook

1. Go to your GitHub repo: `https://github.com/Metta-AI/metta/settings/hooks`
2. Click "Add webhook" (or edit existing)
3. Set:
   - **Payload URL**: `https://abc123.ngrok.io/webhooks/github` (use your ngrok URL)
   - **Content type**: `application/json`
   - **Secret**: Your `GITHUB_WEBHOOK_SECRET` (or leave empty for local testing)
   - **Events**: Select "Pull requests"
   - Click "Add webhook"

### 4. Test scenarios

#### Test 1: Create a PR
1. Create a new PR in the repo
2. Check webhook service logs - should see "opened" event
3. Check Asana - should see new task created
4. Check PR description - should have Asana task link added

#### Test 2: Assign PR to someone
1. Assign the PR to a GitHub user (e.g., yourself)
2. Check webhook service logs - should see "assigned" event
3. Check Asana task - assignee should be updated

#### Test 3: Unassign PR
1. Unassign the PR
2. Check webhook service logs - should see "unassigned" event
3. Check Asana task - should be reassigned to PR author

#### Test 4: Edit PR (change assignee)
1. Edit the PR and change the assignee
2. Check webhook service logs - should see "edited" event
3. Check Asana task - assignee should be updated

#### Test 5: Close PR
1. Close the PR (merge or close without merging)
2. Check webhook service logs - should see "closed" event
3. Check Asana task - should be marked as complete

#### Test 6: Reopen PR
1. Reopen the closed PR
2. Check webhook service logs - should see "reopened" event
3. Check Asana task - should be marked as incomplete

## Monitoring

Watch the webhook service logs to see:
- Events received
- Asana API calls
- Task creation/updates
- Any errors

## Troubleshooting

- **Webhook not receiving events**: Check ngrok is running and URL is correct in GitHub settings
- **Asana task not created**: Check ASANA_PAT, ASANA_WORKSPACE_GID, ASANA_PROJECT_GID are set
- **Assignee not updating**: Check GitHub username matches Asana email in roster project
- **Signature verification fails**: Set GITHUB_WEBHOOK_SECRET or disable verification in dev mode


