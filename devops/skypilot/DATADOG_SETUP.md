# Datadog Integration for SkyPilot

## Overview

This setup installs Datadog Agent on the EC2 host to monitor Docker containers launched by SkyPilot. The agent runs on the host VM and automatically discovers and monitors all containers.

## Architecture

```
EC2 Host Instance
├── Datadog Agent (installed during setup phase)
│   ├── Monitors Docker daemon via /var/run/docker.sock
│   ├── Collects host metrics
│   └── Auto-discovers containers
└── Docker Containers
    └── metta:latest (your application container)
```

## How It Works

1. **Setup Phase** (runs on host):
   - Installs Datadog Agent on EC2 instance
   - Configures Docker integration
   - Sets up container auto-discovery
   - Applies SkyPilot tags (run_id, task_id, node_rank)

2. **Run Phase** (runs in container):
   - Datadog Agent on host automatically detects new container
   - Collects container metrics (CPU, memory, network, disk I/O)
   - Forwards container logs to Datadog
   - Tags all metrics with container metadata

## Configuration

### API Key Management

The Datadog API key is retrieved in this order:
1. `DD_API_KEY` environment variable (if set locally)
2. AWS Secrets Manager: `datadog/api-key` secret
3. Skips installation if no key found (non-blocking)

### Automatic Tags

All metrics and logs are tagged with:
- `env:production`
- `service:skypilot-worker`
- `metta_run_id:<run_id>`
- `skypilot_task_id:<task_id>`
- `skypilot_node_rank:<rank>`
- `skypilot_num_nodes:<total_nodes>`
- `git_ref:<commit_hash>`
- Container-specific tags (image, name, etc.)

### Features Enabled

- **Container Monitoring**: Auto-discovery and metrics for all containers
- **Log Collection**: Collects stdout/stderr from containers
- **APM/Tracing**: Accepts traces from containers (port 8126)
- **Process Monitoring**: Tracks running processes
- **Docker Daemon Metrics**: Monitor Docker itself

## Files Modified

1. **`devops/skypilot/config/lifecycle/install_datadog.sh`**
   - New script that installs and configures Datadog Agent on host
   - Configures Docker integration and container discovery
   - Sets up proper tagging from SkyPilot environment

2. **`devops/skypilot/config/lifecycle/setup.sh`**
   - Updated to call install_datadog.sh during setup phase
   - Runs on host VM before container starts

3. **`devops/skypilot/utils.py`**
   - Added DD_API_KEY to secrets passed to SkyPilot
   - Retrieves key from environment or AWS Secrets Manager

4. **`devops/skypilot/config/skypilot_run.yaml`**
   - Added DD_API_KEY to secrets section

5. **`metta/setup/components/datadog_agent.py`**
   - Updated to skip installation when running inside Docker container
   - Prevents accidental installation in wrong context

## Monitoring in Datadog

Once deployed, you can monitor your SkyPilot jobs in Datadog:

1. **Infrastructure List**: See all EC2 hosts and containers
2. **Container Map**: Visualize container resource usage
3. **Logs**: Search logs by `metta_run_id` or other tags
4. **Metrics Explorer**: Query metrics like:
   - `docker.cpu.usage{metta_run_id:your_run_id}`
   - `docker.mem.rss{service:skypilot-worker}`
   - `system.gpu.utilization{skypilot_task_id:your_task}`

## Troubleshooting

### Check Agent Status on Host
```bash
# SSH to the EC2 instance, then:
sudo systemctl status datadog-agent
sudo -u dd-agent datadog-agent status
```

### View Agent Logs
```bash
sudo journalctl -u datadog-agent -n 100 -f
```

### Verify Docker Integration
```bash
sudo -u dd-agent datadog-agent check docker
```

### Common Issues

1. **No metrics showing up**:
   - Verify DD_API_KEY is set correctly
   - Check agent status on host
   - Ensure Docker socket is accessible

2. **Container not discovered**:
   - Check if dd-agent user is in docker group
   - Verify Docker integration config exists
   - Restart agent: `sudo systemctl restart datadog-agent`

3. **Missing tags**:
   - Check environment variables in setup phase
   - Verify tags in `/etc/datadog-agent/datadog.yaml`

## Testing

To test the setup locally:

```bash
# Launch a job with Datadog monitoring
./devops/skypilot/launch.py experiments.recipes.arena.train \
  --run test-datadog-$(date +%s) \
  --dry-run  # Remove to actually launch

# The agent will be installed during setup phase
# Metrics should appear in Datadog within 1-2 minutes
```

## Benefits

- **No container modification needed**: Works with existing Docker images
- **Comprehensive monitoring**: Host + container metrics in one place  
- **Automatic discovery**: New containers monitored instantly
- **Low overhead**: Single agent per host, not per container
- **SkyPilot aware**: Automatic tagging with job metadata