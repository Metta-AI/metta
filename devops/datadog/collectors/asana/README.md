# Asana Collector

Collects project management and task velocity metrics from Asana API.

**Status**: 📋 **Planned** (Low Priority)

## Metrics to Collect

### Task Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `asana.tasks.open` | Currently open tasks | count | Medium |
| `asana.tasks.completed_7d` | Tasks completed in last 7 days | count | Medium |
| `asana.tasks.created_7d` | Tasks created in last 7 days | count | Low |
| `asana.tasks.overdue` | Tasks past due date | count | High |
| `asana.tasks.blocked` | Tasks marked as blocked | count | Medium |
| `asana.tasks.in_progress` | Tasks currently in progress | count | Low |

### Completion Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `asana.tasks.completion_rate_7d` | Task completion rate (%) | percent | Medium |
| `asana.tasks.avg_time_to_complete_days` | Average time from creation to completion | days | Low |
| `asana.tasks.velocity_7d` | Tasks completed per week | count | Medium |

### Project Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `asana.projects.active` | Projects with activity in last 7 days | count | Low |
| `asana.projects.completion_percent` | Overall project completion % | percent | Low |
| `asana.projects.on_track` | Projects on schedule | count | Medium |
| `asana.projects.at_risk` | Projects at risk or behind | count | High |

### Priority Distribution

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `asana.tasks.high_priority` | High priority open tasks | count | High |
| `asana.tasks.medium_priority` | Medium priority open tasks | count | Low |
| `asana.tasks.low_priority` | Low priority open tasks | count | Low |

### Team Activity

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `asana.users.active_7d` | Users who completed tasks in 7 days | count | Low |
| `asana.tasks.per_user_7d` | Average tasks completed per user | count | Low |
| `asana.tasks.unassigned` | Tasks without assignee | count | Medium |

### Sprint/Milestone Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `asana.sprints.current_completion_percent` | Current sprint completion % | percent | Medium |
| `asana.milestones.upcoming` | Milestones in next 30 days | count | Low |
| `asana.milestones.overdue` | Milestones past due date | count | High |

## Configuration

### Required Secrets (AWS Secrets Manager)

- `asana/personal-access-token` - Asana Personal Access Token

### Environment Variables

- `ASANA_WORKSPACE` - Asana workspace ID
- `ASANA_PROJECT_IDS` - Comma-separated project IDs to track (optional, can query all)

### Collection Schedule

**Recommended**: Every 6 hours (`0 */6 * * *`)

- Tasks don't change as frequently as code/infra
- Reduces API calls to Asana
- Still provides daily visibility

## API Endpoints

Asana REST API v1:

```
GET /tasks                          # List tasks with filters
GET /tasks/{task_id}               # Task details
GET /projects                      # List projects
GET /projects/{project_id}/tasks   # Tasks in project
GET /workspaces/{workspace_id}/users # Workspace users
```

Reference: https://developers.asana.com/reference/rest-api-reference

## Implementation Notes

### Sample Metric Implementation

```python
from datetime import datetime, timedelta, timezone
import httpx

from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret


def _get_asana_client() -> httpx.Client:
    """Create authenticated Asana API client."""
    token = get_secret("asana/personal-access-token")

    return httpx.Client(
        base_url="https://app.asana.com/api/1.0",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        timeout=30.0,
    )


@metric("asana.tasks.open", unit="count")
def get_open_tasks() -> int:
    """Number of currently open tasks."""
    with _get_asana_client() as client:
        workspace_id = os.environ.get("ASANA_WORKSPACE")

        response = client.get(
            "/tasks",
            params={
                "workspace": workspace_id,
                "completed_since": "now",  # Only incomplete tasks
                "opt_fields": "name,completed",
            },
        )
        response.raise_for_status()
        return len(response.json()["data"])


@metric("asana.tasks.completed_7d", unit="count")
def get_completed_tasks_7d() -> int:
    """Tasks completed in last 7 days."""
    with _get_asana_client() as client:
        workspace_id = os.environ.get("ASANA_WORKSPACE")
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        response = client.get(
            "/tasks",
            params={
                "workspace": workspace_id,
                "completed_since": since,
                "opt_fields": "name,completed,completed_at",
            },
        )
        response.raise_for_status()

        # Filter for completed tasks
        tasks = response.json()["data"]
        completed = [t for t in tasks if t.get("completed")]
        return len(completed)


@metric("asana.tasks.overdue", unit="count")
def get_overdue_tasks() -> int:
    """Tasks past their due date."""
    with _get_asana_client() as client:
        workspace_id = os.environ.get("ASANA_WORKSPACE")

        response = client.get(
            "/tasks",
            params={
                "workspace": workspace_id,
                "completed_since": "now",  # Only incomplete
                "opt_fields": "name,due_on,completed",
            },
        )
        response.raise_for_status()

        now = datetime.now(timezone.utc).date()
        overdue_count = 0

        for task in response.json()["data"]:
            due_on = task.get("due_on")
            if due_on:
                due_date = datetime.fromisoformat(due_on).date()
                if due_date < now and not task.get("completed"):
                    overdue_count += 1

        return overdue_count
```

## Dashboard Widgets

Recommended visualizations:

1. **Task Overview**
   - Query value: Open tasks
   - Query value: Overdue tasks
   - Timeseries: Tasks created vs completed

2. **Team Velocity**
   - Query value: Completion rate (7d)
   - Timeseries: Tasks completed per week
   - Query value: Average time to complete

3. **Project Health**
   - Query value: Projects at risk
   - Pie chart: Tasks by priority
   - Query value: Blocked tasks

4. **Sprint Progress** (if using sprints)
   - Query value: Current sprint completion %
   - Timeseries: Burndown chart
   - Query value: Tasks remaining

## Alerting Recommendations

```yaml
# High overdue task count
Query: avg(last_24h):sum:asana.tasks.overdue{} > 20
Alert: More than 20 overdue tasks

# Low completion rate
Query: avg(last_7d):avg:asana.tasks.completion_rate_7d{} < 50
Alert: Task completion rate below 50%

# Many blocked tasks
Query: avg(last_24h):sum:asana.tasks.blocked{} > 10
Alert: More than 10 blocked tasks

# Projects at risk
Query: avg(last_24h):sum:asana.projects.at_risk{} > 3
Alert: 3 or more projects marked at risk
```

## Dependencies

- `httpx` - HTTP client
- `boto3` - AWS Secrets Manager access

## Challenges & Considerations

1. **API Rate Limits**: Asana has rate limits (150 requests/minute)
2. **Workspace Access**: Need appropriate permissions in workspace
3. **Custom Fields**: Projects may use custom fields for priority/status
4. **Large Projects**: Projects with 1000s of tasks may need pagination
5. **Sprint Tracking**: Asana doesn't have native sprint concept, may need custom fields

## Authentication

Requires Personal Access Token with permissions:
- Read tasks
- Read projects
- Read users

Generate token: https://app.asana.com/0/my-apps

## Related Documentation

- [Asana API Documentation](https://developers.asana.com/docs)
- [Asana REST API Reference](https://developers.asana.com/reference/rest-api-reference)
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)

## Next Steps

1. Determine if Asana tracking is needed (vs GitHub Issues)
2. Obtain Personal Access Token
3. Store secret in AWS Secrets Manager
4. Identify workspace and project IDs
5. Implement collector following template
6. Test with actual workspace data
7. Deploy to production
8. Create dashboard

## Alternative: GitHub Issues

**Note**: Consider using GitHub Issues instead of Asana if most project tracking happens there. This would avoid needing another integration.

## Maintenance

- **Owner**: Product/Project Management Team
- **Priority**: Low (optional integration)
- **API Version**: Asana API v1.0
- **Estimated Effort**: 2-3 hours implementation
- **Last Updated**: 2025-10-22
