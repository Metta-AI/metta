// Preset widget patterns for common use cases
// Built on top of primitive widgets with smart defaults and common configurations
//
// This library provides pre-configured widgets for common metrics:
// - Infrastructure: CPU, memory, disk, network
// - Application: Error rates, latency, throughput
// - Business: User counts, conversion rates, revenue
//
// @philosophy: Progressive disclosure with sensible defaults
// @layer: Level 2 - Presets (built on primitives)

local widgets = import 'widgets.libsonnet';

{
  // ========== INFRASTRUCTURE PRESETS ==========
  //
  // Common system metrics with appropriate display styles

  // CPU usage timeseries with warning/critical thresholds
  // @preset: cpuTimeseries
  // @use_case: System CPU monitoring
  // @thresholds: 80% warning, 95% critical
  cpuTimeseries(title='CPU Usage (%)', query='avg:system.cpu.user{*}')::
    widgets.timeseries(title, query, {
      display_type: 'area',
      palette: 'warm',
      markers: [
        { label: 'Warning: 80%', value: 'y = 80', display_type: 'warning dashed' },
        { label: 'Critical: 95%', value: 'y = 95', display_type: 'error dashed' },
      ],
    }),

  // Memory usage timeseries with warning/critical thresholds
  // @preset: memoryTimeseries
  // @use_case: System memory monitoring
  // @thresholds: 80% warning, 90% critical
  memoryTimeseries(title='Memory Usage (%)', query='avg:system.mem.used{*} / avg:system.mem.total{*} * 100')::
    widgets.timeseries(title, query, {
      display_type: 'area',
      palette: 'purple',
      markers: [
        { label: 'Warning: 80%', value: 'y = 80', display_type: 'warning dashed' },
        { label: 'Critical: 90%', value: 'y = 90', display_type: 'error dashed' },
      ],
    }),

  // Disk usage gauge
  // @preset: diskUsageGauge
  // @use_case: Current disk utilization
  diskUsageGauge(title='Disk Usage', query='avg:system.disk.used{*}')::
    widgets.queryValue(title, query, {
      precision: 0,
      custom_unit: '%',
      autoscale: false,
      aggregator: 'last',
    }),

  // Network throughput timeseries
  // @preset: networkThroughput
  // @use_case: Network bytes in/out
  networkThroughput(title='Network Throughput', query='avg:system.net.bytes_rcvd{*}')::
    widgets.timeseries(title, query, {
      display_type: 'line',
      palette: 'cool',
      show_legend: true,
    }),

  // ========== APPLICATION PRESETS ==========
  //
  // Common application metrics with appropriate thresholds

  // Error rate timeseries with SLO threshold
  // @preset: errorRateTimeseries
  // @use_case: Application error monitoring
  // @threshold: 1% error rate SLO
  errorRateTimeseries(title='Error Rate (%)', query='avg:app.errors.rate{*}')::
    widgets.timeseries(title, query, {
      display_type: 'bars',
      palette: 'warm',
      markers: [
        { label: 'SLO: < 1%', value: 'y = 1', display_type: 'error dashed' },
        { label: 'Target: < 0.1%', value: 'y = 0.1', display_type: 'ok dashed' },
      ],
    }),

  // Error rate gauge (current value)
  // @preset: errorRateGauge
  // @use_case: Current error rate display
  errorRateGauge(title='Current Error Rate', query='avg:app.errors.rate{*}')::
    widgets.queryValue(title, query, {
      precision: 2,
      custom_unit: '%',
      autoscale: false,
      aggregator: 'avg',
    }),

  // Latency timeseries with percentiles and SLO
  // @preset: latencyTimeseries
  // @use_case: Application response time monitoring
  // @threshold: 200ms SLO
  latencyTimeseries(title='Response Time (ms)', query='avg:app.latency{*}')::
    widgets.timeseries(title, query, {
      display_type: 'line',
      palette: 'purple',
      line_width: 'thick',
      markers: [
        { label: 'SLO: 200ms', value: 'y = 200', display_type: 'ok dashed' },
        { label: 'Warning: 500ms', value: 'y = 500', display_type: 'warning dashed' },
      ],
    }),

  // Latency gauge (current value)
  // @preset: latencyGauge
  // @use_case: Current latency display
  latencyGauge(title='Avg Latency', query='avg:app.latency{*}')::
    widgets.queryValue(title, query, {
      precision: 0,
      custom_unit: 'ms',
      autoscale: false,
      aggregator: 'avg',
    }),

  // Request rate timeseries
  // @preset: requestRateTimeseries
  // @use_case: Throughput monitoring
  requestRateTimeseries(title='Request Rate', query='sum:app.requests{*}')::
    widgets.timeseries(title, query, {
      display_type: 'bars',
      palette: 'cool',
    }),

  // Request count gauge
  // @preset: requestCountGauge
  // @use_case: Total requests display
  requestCountGauge(title='Total Requests', query='sum:app.requests{*}')::
    widgets.queryValue(title, query, {
      precision: 0,
      autoscale: true,
      aggregator: 'sum',
    }),

  // Success rate timeseries with SLO
  // @preset: successRateTimeseries
  // @use_case: Application availability monitoring
  // @threshold: 99% SLO
  successRateTimeseries(title='Success Rate (%)', query='(sum:app.requests{*} - sum:app.errors{*}) / sum:app.requests{*} * 100')::
    widgets.timeseries(title, query, {
      display_type: 'line',
      palette: 'cool',
      line_width: 'thick',
      markers: [
        { label: 'SLO: 99%', value: 'y = 99', display_type: 'ok dashed' },
        { label: 'Critical: 95%', value: 'y = 95', display_type: 'error dashed' },
      ],
    }),

  // ========== BUSINESS PRESETS ==========
  //
  // Common business metrics

  // Active user count gauge
  // @preset: activeUsersGauge
  // @use_case: Current active user count
  activeUsersGauge(title='Active Users', query='sum:users.active{*}')::
    widgets.queryValue(title, query, {
      precision: 0,
      autoscale: true,
      aggregator: 'last',
    }),

  // User growth change widget
  // @preset: userGrowthChange
  // @use_case: Day-over-day user growth
  userGrowthChange(title='User Growth', query='sum:users.active{*}')::
    widgets.change(title, query, {
      compare_to: 'day_before',
      increase_good: true,
      show_present: true,
    }),

  // Conversion rate gauge
  // @preset: conversionRateGauge
  // @use_case: Conversion funnel monitoring
  conversionRateGauge(title='Conversion Rate', query='sum:conversions{*} / sum:visitors{*} * 100')::
    widgets.queryValue(title, query, {
      precision: 1,
      custom_unit: '%',
      autoscale: false,
      aggregator: 'avg',
    }),

  // Revenue timeseries
  // @preset: revenueTimeseries
  // @use_case: Revenue tracking
  revenueTimeseries(title='Revenue', query='sum:revenue{*}')::
    widgets.timeseries(title, query, {
      display_type: 'bars',
      palette: 'cool',
    }),

  // ========== TOP LISTS ==========
  //
  // Common ranked lists

  // Top services by request count
  // @preset: topServicesByRequests
  topServicesByRequests(title='Top Services by Requests', query='sum:requests{*} by {service}')::
    widgets.toplist(title, query),

  // Top endpoints by latency
  // @preset: topEndpointsByLatency
  topEndpointsByLatency(title='Slowest Endpoints', query='avg:latency{*} by {endpoint}')::
    widgets.toplist(title, query),

  // Top hosts by CPU
  // @preset: topHostsByCPU
  topHostsByCPU(title='Top Hosts by CPU', query='avg:system.cpu.user{*} by {host}')::
    widgets.toplist(title, query),

  // Top hosts by memory
  // @preset: topHostsByMemory
  topHostsByMemory(title='Top Hosts by Memory', query='avg:system.mem.used{*} by {host}')::
    widgets.toplist(title, query),

  // ========== SECTION HEADERS ==========
  //
  // Common section headers for dashboard organization

  // Blue section header
  // @preset: sectionHeader
  sectionHeader(title, description='')::
    widgets.note(
      '## ' + title + (if description != '' then '\n\n' + description else ''),
      {
        background_color: 'blue',
        font_size: '18',
        text_align: 'center',
        vertical_align: 'center',
      }
    ),

  // Gray subsection header
  // @preset: subsectionHeader
  subsectionHeader(title)::
    widgets.note(
      '### ' + title,
      {
        background_color: 'gray',
        font_size: '16',
        text_align: 'left',
        vertical_align: 'center',
      }
    ),

  // Alert/warning note
  // @preset: alertNote
  alertNote(message)::
    widgets.note(
      '**Alert:** ' + message,
      {
        background_color: 'red',
        font_size: '16',
        text_align: 'center',
      }
    ),

  // Info note
  // @preset: infoNote
  infoNote(message)::
    widgets.note(
      message,
      {
        background_color: 'white',
        font_size: '14',
        text_align: 'left',
      }
    ),

  // ========== COMPARISON PATTERNS ==========
  //
  // Common comparison widgets

  // Before/after comparison (2 timeseries)
  // @preset: beforeAfterTimeseries
  beforeAfterTimeseries(title, beforeQuery, afterQuery)::
    widgets.timeseries(title, beforeQuery + ', ' + afterQuery, {
      display_type: 'line',
      show_legend: true,
    }),

  // Service health table
  // @preset: serviceHealthTable
  serviceHealthTable(title='Service Health')::
    widgets.table(title, [
      { query: 'sum:requests{*} by {service}', alias: 'Requests', aggregator: 'sum' },
      { query: 'avg:latency{*} by {service}', alias: 'Latency (ms)', aggregator: 'avg' },
      { query: 'sum:errors{*} by {service}', alias: 'Errors', aggregator: 'sum' },
    ], {
      has_search_bar: 'auto',
    }),

  // Host health table
  // @preset: hostHealthTable
  hostHealthTable(title='Host Health')::
    widgets.table(title, [
      { query: 'avg:system.cpu.user{*} by {host}', alias: 'CPU %', aggregator: 'avg' },
      { query: 'avg:system.mem.used{*} by {host}', alias: 'Memory %', aggregator: 'avg' },
      { query: 'avg:system.disk.used{*} by {host}', alias: 'Disk %', aggregator: 'avg' },
    ], {
      has_search_bar: 'auto',
    }),
}
