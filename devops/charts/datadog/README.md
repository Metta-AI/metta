The Datadog Agent runs on every node and collects metrics, traces, and logs from all pods running on that node. This
centralized deployment allows any service in the cluster to send telemetry data to Datadog.

## How to connect

Services that need to send data to Datadog should configure the following environment variables:

```yaml
env:
  # Point to the Datadog agent running on the same node
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP

  - name: DD_TRACE_ENABLED
    value: 'true'

  - name: DD_ENV
    value: 'production'

  # Set the service name (used within datadog to group data)
  - name: DD_SERVICE
    value: 'your-service-name'

  # Set the version (used within datadog to group data)
  - name: DD_VERSION
    value: '1.0.0' # image tag or git hash associated with image
```
