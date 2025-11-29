Sample local data sources for the training/eval collectors. Point the collectors at these (or any other) files by
setting:

- `TRAINING_HEALTH_FILE=/app/devops/datadog/data/training_health.json`
- `EVAL_HEALTH_FILE=/app/devops/datadog/data/eval_health.json`

Each record must include:

```json
{
  "metric": "metta.training.hearts",
  "workflow_name": "multigpu_arena_basic_easy_shaped",
  "task": "hearts",
  "check": "reward_threshold",
  "condition": "> 0.5",
  "value": 0.92,
  "status": "pass",
  "timestamp": "2025-01-14T15:00:00Z",
  "tags": {
    "commit": "abc123",
    "workflow:training": "arena"
  }
}
```
