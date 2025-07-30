# Metta Docker Sandbox

A Docker-based testing environment for distributed sweeps that mimics SkyPilot's multi-node setup locally.

## Quick Start

### 1. Setup (one-time)

```bash
./devops/setup_docker_sandbox.sh
```

This will:

- Build the Docker environment
- Start master, worker, and dummy API containers
- Mount your credentials (`.netrc`, `.metta`, `.config`, `.ssh`, `.aws`)
- Set up virtual environments automatically

### 2. Run sweeps

```bash
./devops/docker_launch.py sweep run=test trainer.total_timesteps=50 trainer.curriculum=/env/mettagrid/curriculum/arena/random +hardware=macbook
```

### 3. Other commands

```bash
./devops/docker_launch.py status    # Check environment status
./devops/docker_launch.py shell     # Open shell in master container
./devops/docker_launch.py logs      # View logs
./devops/docker_launch.py stop      # Stop environment
```

## What it provides

- **Multi-node distributed training**: Master and worker containers with proper PyTorch DDP setup
- **Production API compatibility**: Dummy API servers for Observatory/Cogweb endpoints
- **Credential mounting**: Your local auth files are mounted exactly like SkyPilot does
- **Automatic dependencies**: Virtual environment setup handled automatically
- **S3-free testing**: Uses arena curricula that don't require S3 downloads

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  metta-sweep-   │    │  metta-sweep-   │    │  metta-dummy-   │
│     master      │    │     worker      │    │      api        │
│                 │    │                 │    │                 │
│ • Rank 0        │    │ • Rank 1        │    │ • Observatory   │
│ • Orchestration │◄───┤ • Training      │    │ • Cogweb        │
│ • Training      │    │ • Coordination  │    │ • Mock APIs     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration

The sandbox environment:

- Uses `+hardware=macbook` to avoid NCCL GPU requirements
- Mounts credentials from your local system
- Runs arena simulations (no S3 dependencies)
- Automatically handles virtual environment setup

## Debugging

```bash
# Check container status
docker ps | grep metta-sweep

# View logs from specific containers
docker logs metta-sweep-master
docker logs metta-sweep-worker
docker logs metta-dummy-api

# Shell into containers
docker exec -it metta-sweep-master bash
docker exec -it metta-sweep-worker bash

# Clean restart
docker-compose -f metta/metta/sweep/docker/docker-compose.yml down
./devops/setup_docker_sandbox.sh
```

## Files

- `metta/metta/sweep/docker/` - Core Docker environment
  - `Dockerfile` - Container image definition
  - `docker-compose.yml` - Multi-container orchestration
  - `scripts/start-test-env.sh` - Container initialization
  - `scripts/dummy-api-server.py` - Mock API endpoints
- `devops/setup_docker_sandbox.sh` - One-command setup
- `devops/docker_launch.py` - Main entry point (mirrors `skypilot/launch.py`)

## Testing distributed coordination

This environment is specifically designed to test the sweep-to-trainer handoff in a distributed setting, allowing you
to:

1. Debug distributed training coordination issues
2. Test protein suggestions and W&B integration
3. Validate multi-node sweep orchestration
4. Iterate on distributed fixes without SkyPilot overhead

The environment successfully demonstrates that distributed training starts correctly and the sweep orchestration works
as expected.
