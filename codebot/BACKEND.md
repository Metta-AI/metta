# Backend Architecture (Future)

This document outlines the planned backend architecture for remote execution of Codebot commands.

## Overview

The backend enables:
- Subscription to repository events (push, PR, schedule)
- Remote execution of commands and workflows
- Monitoring and debugging of executions
- Result storage and retrieval
- Non-interactive only

## Core Components

```python
@dataclass
class Subscription:
    id: str
    repo: str
    branch_pattern: str = "main"
    paths: List[str]
    command_or_workflow: str
    trigger: Literal["push", "pr", "schedule"]
    config: Dict[str, Any]

@dataclass
class Execution:
    id: str
    subscription_id: str
    trigger_event: Dict[str, Any]
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime
    completed_at: Optional[datetime]
    output: Optional[CommandOutput]
    logs: List[LogEntry]
```

## Event Processing

```python
class EventProcessor:
    async def handle_github_webhook(self, event: GitHubEvent):
        # Find matching subscriptions
        subscriptions = await self.find_matching_subscriptions(
            repo=event.repository,
            paths=event.changed_files,
            trigger=event.event_type
        )

        # Queue executions
        for sub in subscriptions:
            await self.queue_execution(sub, event)

    async def execute_subscription(self, subscription: Subscription, event: Dict):
        # Clone repository at specific commit
        repo_path = await self.clone_repo(event["repository"], event["commit"])

        # Build execution context
        context = ExecutionContext(
            git_diff=await self.get_event_diff(event),
            clipboard="",  # No clipboard in remote execution
            relevant_files=await self.gather_files(subscription.paths),
            working_directory=repo_path,
            mode=ExecutionMode.ONESHOT  # Remote always uses fastest mode
        )

        # Execute command/workflow
        if self.is_workflow(subscription.command_or_workflow):
            output = await self.workflow_engine.execute(
                subscription.command_or_workflow,
                context
            )
        else:
            output = await self.command_executor.execute(
                subscription.command_or_workflow,
                context
            )

        # Store results
        await self.store_execution_result(subscription, event, output)
```

## CLI Interface

```bash
# Subscribe to events
remotebot subscribe test src/ tests/          # Test on any change to src/ or tests/
remotebot subscribe lint --paths "*.py"       # Lint all Python files
remotebot subscribe review --trigger pr       # Review on PR creation/update

# Manage subscriptions
remotebot list                                # List all subscriptions
remotebot list --repo myorg/myrepo           # List for specific repo
remotebot show <subscription-id>              # Show subscription details
remotebot pause <subscription-id>             # Pause subscription
remotebot resume <subscription-id>            # Resume subscription
remotebot delete <subscription-id>            # Delete subscription

# View execution history
remotebot logs <subscription-id>              # View recent executions
remotebot logs <subscription-id> --tail 50    # View last 50 log entries
remotebot result <execution-id>               # View execution result

# Manual trigger
remotebot trigger <subscription-id>           # Manually trigger execution
```

## Security Considerations

1. **Authentication**: API keys or GitHub App authentication
2. **Authorization**: Verify user has access to repository
3. **Isolation**: Execute in sandboxed environments
4. **Rate Limiting**: Prevent abuse of LLM resources
5. **Secrets**: Never log or expose sensitive information

## Deployment Options

### Self-Hosted
```yaml
# docker-compose.yml
version: '3.8'
services:
  codebot-server:
    image: codebot/server:latest
    environment:
      - DATABASE_URL=postgresql://...
      - GITHUB_WEBHOOK_SECRET=...
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
```

### Cloud Deployment
- Kubernetes with horizontal scaling
- Managed queue service (SQS, Cloud Tasks)
- Object storage for results (S3, GCS)
- Observability (logs, metrics, traces)

## Implementation Priority

1. **Phase 1**: Basic webhook processing and execution
2. **Phase 2**: Subscription management and CLI
3. **Phase 3**: Monitoring and debugging tools
