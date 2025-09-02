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

### Postgres

The app_backend service relies on postgres.

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

## Policy Evaluator

This service evaluates policies on-demand. It will be deployed on Kubernetes.

### Local development

Ensure that you have app_backend running. If it is running anywhere except for your `localhost:8000`, then you will need
to provide `BACKEND_URL` as an env var in the following sections.

#### Local docker

Getting the service running

- `metta local build-policy-evaluator-img` to build the `metta-policy-evaluator-local:latest` image. This will serve as
  the base for both the orchestrator and the workers
- `WANDB_API_KEY=your-key-here docker compose -f app_backend/src/metta/app_backend/docker-compose.yml up`

Viewing logs

- Orchestrator: `docker compose -f app_backend/src/metta/app_backend/docker-compose.yml logs`
- Workers: `docker ps` to find the worker id, and `docker logs {id} --follow`

#### Local kind

Kind is a tool for running local Kubernetes clusters using Docker container nodes. We use it for testing our Kubernetes
deployment locally.

- Ensure `metta status` shows that you are connected to `observatory-local-key` and `wandb`
- Run your local app_backend server, which should be serving on `localhost:8000`
- Run your local frontend from `observatory/`
- Build orchestrator and worker images: `metta local build-policy-evaluator-img`
- Set up helm: `metta local kind build`
- Launch: `metta local kind up`
- Monitor: `metta local kind get-pods` and `metta local kind pods {pod-name}`
