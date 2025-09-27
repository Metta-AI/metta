# Softmax Orchestrator

Softmax's orchestration service powers evaluation scheduling, container lifecycle
management, and backend APIs (served at https://observatory.softmax-research.net/).

## Building the Docker Image

Run from the repository root:

```bash
docker build -t softmax-orchestrator:latest -f packages/softmax-orchestrator/Dockerfile .
```

## Running the Container Locally

```bash
docker run -p 8000:8000 \
  -e STATS_DB_URI="postgres://user:password@host:port/db" \
  softmax-orchestrator:latest
```

### Postgres

The orchestrator relies on PostgreSQL. When running locally, use
`host.docker.internal` as the host if your database is on the host machine.

## Environment Variables

- `STATS_DB_URI`: PostgreSQL connection string (default:
  `postgres://postgres:password@127.0.0.1/postgres`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)
- `DEBUG_USER_EMAIL`: Local override to bypass OAuth proxies when developing
  against the API directly.

## Development

To run without Docker:

```bash
cd packages/softmax-orchestrator
DEBUG_USER_EMAIL=localdev@softmax.ai uv run python server.py
```

## API Endpoints

- `/dashboard/*` – Dashboard-related endpoints
- `/stats/*` – Statistics and data recording endpoints

## Policy Evaluator

The service dispatches evaluation jobs (locally or via Kubernetes). Ensure the
orchestrator is running before invoking CLI helpers such as
`softmax.training.tools.sim` or `softmax.training.tools.sweep`.
