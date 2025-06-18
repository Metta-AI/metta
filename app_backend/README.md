# Metta App Backend

A FastAPI-based backend server for the Metta AI framework, containerized with Docker and managed with uv.

## Structure

```
app_backend/
├── src/                    # Source code
│   ├── server.py          # Main FastAPI application
│   ├── config.py          # Configuration settings
│   ├── metta_repo.py      # Database repository
│   ├── schema_manager.py  # Database migration utilities
│   └── routes/            # API route handlers
├── Dockerfile             # Docker configuration
├── pyproject.toml         # Project dependencies (uv)
├── build.sh              # Build script
└── .dockerignore         # Docker ignore file
```

## Building the Docker Image

1. Make sure you have Docker and uv installed
2. Run the build script:
   ```bash
   ./build.sh
   ```

This will:

- Generate a `uv.lock` file with pinned dependencies
- Build the Docker image tagged as `metta-app-backend:latest`

## Running the Container

```bash
docker run -p 8000:8000 \
  -e STATS_DB_URI="postgres://user:password@host:port/db" \
  metta-app-backend:latest
```

## Environment Variables

- `STATS_DB_URI`: PostgreSQL connection string (default: `postgres://postgres:password@127.0.0.1/postgres`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)

## Development

To run locally without Docker:

```bash
cd app_backend
uv sync
uv run python src/server.py
```

## API Endpoints

- `/dashboard/*` - Dashboard-related endpoints
- `/stats/*` - Statistics and data recording endpoints
