# Datadog Monitoring

This chart deploys the Datadog Agent as a DaemonSet across all nodes in the Kubernetes cluster.

## Overview

The Datadog Agent runs on every node and collects metrics, traces, and logs from all pods running on that node. This
centralized deployment allows any service in the cluster to send telemetry data to Datadog.

## How Services Connect to Datadog

Services that need to send data to Datadog should configure the following environment variables:

```yaml
env:
  # Point to the Datadog agent running on the same node
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP

  # Enable tracing (optional)
  - name: DD_TRACE_ENABLED
    value: 'true'

  # Set the environment tag
  - name: DD_ENV
    value: 'production' # or staging, development, etc.

  # Set the service name
  - name: DD_SERVICE
    value: 'your-service-name'

  # Set the version
  - name: DD_VERSION
    value: '1.0.0' # typically from your image tag
```

## Example: Skypilot Integration

For Skypilot workers or any other service, add the above environment variables to your deployment/pod specification. The
`DD_AGENT_HOST` configuration ensures that the pod connects to the Datadog agent running on the same node, which then
forwards the data to Datadog's servers.

## Configuration

The Datadog agent is configured with:

- APM (Application Performance Monitoring) enabled on port 8126
- Log collection enabled for all containers
- Process monitoring enabled
- Kubernetes orchestrator explorer enabled

## Secret Management

The Datadog API key must be stored in a Kubernetes secret named `datadog-secret` in the monitoring namespace:

```bash
kubectl create secret generic datadog-secret \
  --from-literal=api-key=YOUR_DATADOG_API_KEY \
  -n monitoring
```
