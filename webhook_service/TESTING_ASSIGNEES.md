# Testing Assignee Syncing

## Important: Assignees vs Reviewers

The webhook syncs **PR assignees** (not reviewers). In GitHub:

- **Assignees**: People responsible for the PR (set in PR sidebar)
- **Reviewers**: People requested to review the PR

The webhook only syncs **assignees** to Asana task assignees.

## How to Test Assignee Syncing

### 1. Create a new PR

- Create a new PR in your fork
- Don't assign anyone initially (or assign yourself)

### 2. Test Assign Action

- In the PR, click "Assignees" in the right sidebar
- Assign the PR to a GitHub user (e.g., yourself or another user)
- Watch webhook logs - should see `action=assigned`
- Check Asana task - assignee should update

### 3. Test Unassign Action

- Unassign the PR (remove assignee)
- Watch webhook logs - should see `action=unassigned`
- Check Asana task - should be reassigned to PR author

### 4. Test Edit Action (change assignee)

- Edit the PR and change the assignee
- Watch webhook logs - should see `action=edited`
- Check Asana task - assignee should update

## Finding Completed Tasks in Asana

Completed tasks are often hidden by default. To see them:

1. **In the project view:**
   - Click the project in Asana
   - Look for a filter or view option
   - Select "Completed" or "All tasks"

2. **Direct task URL:**
   - Use the task URL from logs:
     `https://app.asana.com/1/1209016784099267/project/1210348820405981/task/1212561279132368`
   - Even if completed, the URL should still work

3. **Search:**
   - Use Asana search to find the task by name or PR number

## What to Watch in Logs

When you assign/unassign:

- `action=assigned` or `action=unassigned` or `action=edited`
- `Updated Asana task assignee`
- `assignTo: <github-username>`
- `success: true`
