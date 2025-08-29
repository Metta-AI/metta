# Datadog Container Setup for SkyPilot

## Overview

This setup runs Datadog Agent as a Docker container alongside your application container on SkyPilot EC2 instances.

## Architecture

```
EC2 Host Instance
├── Datadog Agent Container (dd-agent)
│   ├── Monitors all containers via Docker socket
│   ├── Collects host metrics via /proc and /sys mounts
│   └── Sends metrics to Datadog
└── Application Container (metta:latest)
    └── Your training job
```

## How It Works

1. **Setup Phase**: Just checks out code (no Datadog installation needed)

2. **Run Phase**: 
   - Starts Datadog agent as a Docker container
   - Agent monitors all containers on the host
   - Collects metrics and logs
   - Tags everything with SkyPilot metadata

## Implementation

### Files

- `start_datadog_container.sh` - Launches Datadog container with proper configuration
- `skypilot_run.sh` - Modified to call the startup script
- `setup.sh` - Simplified (no agent installation)

### Container Configuration

The Datadog container runs with:
- `--network host` - Access to host network
- `--pid host` - See all processes
- Docker socket mounted for container discovery
- `/proc` and `/sys/fs/cgroup` mounted for metrics
- Environment variables for API key and tags

## Monitoring

Metrics appear in Datadog within 1-2 minutes of job start. Look for:
- Container metrics (CPU, memory, network, disk)
- Host metrics
- Docker daemon metrics
- All tagged with `metta_run_id` and `skypilot_task_id`

## Troubleshooting

Check if container is running:
```bash
docker ps | grep dd-agent
```

View container logs:
```bash
docker logs dd-agent
```

## Benefits

- **Simple**: One Docker run command
- **No installation**: Nothing to install during setup
- **Reliable**: Docker manages the agent lifecycle
- **Complete monitoring**: Full visibility into all containers and host