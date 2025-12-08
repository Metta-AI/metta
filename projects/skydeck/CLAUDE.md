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
