# SkyDeck Development Guidelines

## SkyPilot Integration

**IMPORTANT**: Always use the SkyPilot API for accessing job and cluster information. Never query the SkyPilot SQLite databases directly (e.g., `~/.sky/jobs.db`, `~/.sky/state.db`).

### API Access

- API endpoint: `https://skypilot-api.softmax-research.net`
- Authentication: OAuth2 cookies stored in `~/.sky/cookies.txt`
- Configuration: `~/.sky/config.yaml`

### Why Use the API

1. **Centralized data**: The API provides access to managed jobs across all users and clusters
2. **Up-to-date information**: The API reflects the current state of the jobs controller
3. **Proper abstractions**: The API provides structured data with proper types
4. **Security**: Direct database access bypasses authentication and auditing

## Data Model

### Experiment Groups
- **Created by**: User (via UI)
- **Purpose**: Organize experiments into logical groups
- **Contains**: Ordered list of experiments (many-to-many relationship)
- **Fields**: id, name, flags (columns to display), order, collapsed

### Experiments
- **Created by**: User (via "Create" button or "Duplicate")
- **Purpose**: Configuration template that defines what to run
- **Key fields**:
  - `id`: Auto-increment integer (internal)
  - `name`: Unique string identifier (user-facing, used for matching jobs/checkpoints)
  - `desired_state`: RUNNING or STOPPED
  - `current_state`: Reflects latest job status
  - `flags`: Configuration key-value pairs
  - `nodes`, `gpus`: Resource requirements

### Jobs
- **Created by**: Synced from SkyPilot API (poller)
- **Purpose**: Track actual job executions
- **Matching**: Jobs display under experiments where `job.experiment_id == experiment.name`
- **Key fields**: id, experiment_id (matches experiment.name), status, command, git_ref, nodes, gpus

### Checkpoints
- **Created by**: Synced from S3 (syncer)
- **Purpose**: Track model checkpoints and replay files
- **S3 path**: `s3://softmax-public/policies/{experiment.name}/`
- **Key fields**: experiment_id (references experiment.id), epoch, model_path, replay_paths, policy_version

### Key Relationships
- **Jobs** match to experiments by **name**: `job.experiment_id == experiment.name`
- **Checkpoints** are stored by **id**: `checkpoint.experiment_id == experiment.id`
- **S3 paths** use experiment **name** (e.g., `s3://.../{experiment.name}/`)
- If no experiment exists with matching name, jobs appear as "orphaned"

## Database

**Database Location**: SkyDeck uses SQLite for persistent storage.

- **Default location**: `~/.skydeck/skydeck.db`
- **Configuration**: Can be overridden with `--db-path` flag or `SKYDECK_DB_PATH` environment variable
- **Schema**: Defined in `skydeck/database.py` with automatic migrations on startup

### Database Scripts

When working with the database directly:

```bash
# Backfill checkpoint versions (example)
uv run python -c "
import asyncio
from pathlib import Path
from skydeck.backfill_versions import backfill_checkpoint_versions
db_path = str(Path.home() / '.skydeck' / 'skydeck.db')
asyncio.run(backfill_checkpoint_versions(db_path))
"

# Query database directly
sqlite3 ~/.skydeck/skydeck.db "SELECT COUNT(*) FROM experiments;"
```

### Code Style

- Always use `uv` for pip and python operations
- Imports should go at the top of the file if possible
- Follow existing patterns in the codebase for consistency
- **NEVER add fallbacks** - fix the underlying problem instead
- When making backend changes, restart the server: the user must restart skydeck for changes to take effect

## Development Workflow

After making backend changes (Python), restart the server to pick up changes:
```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 2
nohup uv run skydeck --port 8000 > /tmp/skydeck.log 2>&1 &
sleep 3
curl -s http://localhost:8000/api/health | head -c 100  # Verify it's running
```

After making frontend changes (TypeScript/React):
```bash
cd packages/skydeck/frontend
npm run build  # Builds to ../skydeck/static/
```

**Important**: Always restart the backend yourself to test changes. Do not ask the user to restart.
