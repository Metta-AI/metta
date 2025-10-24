// Preset Demo Dashboard
// Demonstrates how to use preset widgets for common patterns

local layouts = import '../lib/layouts.libsonnet';
local presets = import '../lib/presets.libsonnet';

layouts.grid(
  'Preset Widgets Demo',
  std.flattenArrays([
    // Row 1: Section header
    [layouts.fullWidth(0, presets.sectionHeader(
      'Infrastructure Metrics',
      'System-level monitoring with smart defaults'
    ), height=1)],

    // Row 2: Four infrastructure gauges
    layouts.row(1, [
      presets.diskUsageGauge('Disk Usage', 'avg:system.disk.in_use{*}'),
      presets.activeUsersGauge('Active Users', 'sum:users.active{*}'),
      presets.requestCountGauge('Total Requests', 'sum:requests.total{*}'),
      presets.latencyGauge('Avg Latency', 'avg:request.latency{*}'),
    ], height=2),

    // Row 3: CPU and Memory timeseries with SLO markers
    layouts.row(3, [
      presets.cpuTimeseries('CPU Usage', 'avg:system.cpu.user{*}'),
      presets.memoryTimeseries('Memory Usage', 'avg:system.mem.pct_usable{*}'),
    ], height=3),

    // Row 4: Section header
    [layouts.fullWidth(6, presets.sectionHeader(
      'Application Performance',
      'Response times, error rates, and throughput'
    ), height=1)],

    // Row 5: Application metrics
    layouts.row(7, [
      presets.errorRateGauge('Error Rate', 'avg:error.rate{*}'),
      presets.successRateTimeseries('Success Rate', '(sum:requests{*} - sum:errors{*}) / sum:requests{*} * 100'),
    ], height=3),

    // Row 6: Latency and requests
    layouts.row(10, [
      presets.latencyTimeseries('P95 Latency', 'p95:request.latency{*}'),
      presets.requestRateTimeseries('Request Rate', 'sum:requests{*}.as_rate()'),
    ], height=3),

    // Row 7: Section header
    [layouts.fullWidth(13, presets.sectionHeader(
      'Top Lists',
      'Ranked views of system resources'
    ), height=1)],

    // Row 8: Top lists
    layouts.row(14, [
      presets.topServicesByRequests('Busiest Services', 'sum:requests{*} by {service}'),
      presets.topHostsByCPU('Hosts by CPU', 'avg:system.cpu.user{*} by {host}'),
    ], height=3),

    // Row 9: Section header
    [layouts.fullWidth(17, presets.sectionHeader(
      'Health Tables',
      'Multi-metric service and host health'
    ), height=1)],

    // Row 10: Health tables
    layouts.row(18, [
      presets.serviceHealthTable('Service Health Dashboard'),
      presets.hostHealthTable('Host Health Dashboard'),
    ], height=4),

    // Row 11: Info note
    [layouts.fullWidth(22, presets.infoNote(
      'All widgets use preset configurations with smart defaults. Customize by passing different queries.'
    ), height=1)],
  ]),
  {
    id: 'rd5-3wh-9s2',  // Dashboard ID from Datadog
    description: 'Demonstrates preset widgets with common patterns and smart defaults',
  }
)
