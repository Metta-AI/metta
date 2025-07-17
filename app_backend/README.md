# Metta App Backend

This is the backend for the app deployed at https://observatory.softmax-research.net/

## Building the Docker Image

In the parent directory run the command

```
docker build -t metta-app-backend:latest -f app_backend/Dockerfile .
```

This will build the Docker image tagged as `metta-app-backend:latest`. The command must be run from the parent directory
because we are using the parent `uv.lock` file.

## Running the Container

```bash
docker run -p 8000:8000 \
  -e STATS_DB_URI="postgres://user:password@host:port/db" \
  metta-app-backend:latest
```

If you are running a postgres instance locally, use `host.docker.internal` as host

## Environment Variables

- `STATS_DB_URI`: PostgreSQL connection string (default: `postgres://postgres:password@127.0.0.1/postgres`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)
- `DEBUG_USER_EMAIL` : In production, your auth token is managed by oauth2_proxy. This is a way to run locally without
  the oauth2_proxy.

## Development

To run locally without Docker:

```bash
cd app_backend
DEBUG_USER_EMAIL=localdev@stem.ai uv run python server.py
```

## API Endpoints

- `/dashboard/*` - Dashboard-related endpoints
- `/stats/*` - Statistics and data recording endpoints
