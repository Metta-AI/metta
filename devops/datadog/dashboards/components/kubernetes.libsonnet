// Kubernetes resource efficiency and health components
// Widgets for cluster optimization, pod health monitoring, and cost reduction

local widgets = import '../lib/widgets.libsonnet';

{
  // ========== Resource Efficiency Metrics ==========

  // CPU waste in cores
  cpuWasteWidget()::
    widgets.queryValue(
      title='CPU Waste (cores)',
      query='avg:k8s.resources.cpu_waste_cores{*}',
      options={
        precision: 2,
        aggregator: 'last',
        custom_unit: 'cores',
      }
    ),

  // Memory waste in GB
  memoryWasteWidget()::
    widgets.queryValue(
      title='Memory Waste (GB)',
      query='avg:k8s.resources.memory_waste_gb{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: 'GB',
      }
    ),

  // CPU efficiency percentage
  cpuEfficiencyWidget()::
    widgets.queryValue(
      title='CPU Efficiency',
      query='avg:k8s.resources.cpu_efficiency_pct{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: '%',
      }
    ),

  // Memory efficiency percentage
  memoryEfficiencyWidget()::
    widgets.queryValue(
      title='Memory Efficiency',
      query='avg:k8s.resources.memory_efficiency_pct{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: '%',
      }
    ),

  // Overallocated pods count
  overallocatedPodsWidget()::
    widgets.queryValue(
      title='Overallocated Pods',
      query='avg:k8s.resources.overallocated_pods{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Resource efficiency trend over time
  resourceEfficiencyTrendWidget()::
    widgets.timeseries(
      title='Resource Efficiency Trends',
      query='avg:k8s.resources.cpu_efficiency_pct{*}, avg:k8s.resources.memory_efficiency_pct{*}',
      options={
        show_legend: true,
        palette: 'cool',
        display_type: 'line',
      }
    ),

  // ========== Pod Health Metrics ==========

  // Crash looping pods
  crashLoopingPodsWidget()::
    widgets.queryValue(
      title='Crash Looping Pods',
      query='avg:k8s.pods.crash_looping{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Failed pods
  failedPodsWidget()::
    widgets.queryValue(
      title='Failed Pods',
      query='avg:k8s.pods.failed{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Pending pods
  pendingPodsWidget()::
    widgets.queryValue(
      title='Pending Pods',
      query='avg:k8s.pods.pending{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // OOMKilled pods in last 24h
  oomkilledPodsWidget()::
    widgets.queryValue(
      title='OOMKilled (24h)',
      query='avg:k8s.pods.oomkilled_24h{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Pods with high restarts
  highRestartsWidget()::
    widgets.queryValue(
      title='High Restart Pods',
      query='avg:k8s.pods.high_restarts{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Image pull errors
  imagePullErrorsWidget()::
    widgets.queryValue(
      title='Image Pull Errors',
      query='avg:k8s.pods.image_pull_errors{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Pod health trend over time
  podHealthTrendWidget()::
    widgets.timeseries(
      title='Pod Health Issues Over Time',
      query='avg:k8s.pods.crash_looping{*}, avg:k8s.pods.failed{*}, avg:k8s.pods.oomkilled_24h{*}',
      options={
        show_legend: true,
        palette: 'warm',
        display_type: 'line',
      }
    ),

  // ========== Underutilization Metrics ==========

  // Idle pods
  idlePodsWidget()::
    widgets.queryValue(
      title='Idle Pods',
      query='avg:k8s.pods.idle_count{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Low CPU usage pods
  lowCpuPodsWidget()::
    widgets.queryValue(
      title='Low CPU Usage Pods',
      query='avg:k8s.pods.low_cpu_usage{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Low memory usage pods
  lowMemoryPodsWidget()::
    widgets.queryValue(
      title='Low Memory Usage Pods',
      query='avg:k8s.pods.low_memory_usage{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Deployments with zero replicas
  zeroReplicasWidget()::
    widgets.queryValue(
      title='Zero Replica Deployments',
      query='avg:k8s.deployments.zero_replicas{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Underutilization trend over time
  underutilizationTrendWidget()::
    widgets.timeseries(
      title='Underutilization Trends',
      query='avg:k8s.pods.idle_count{*}, avg:k8s.pods.low_cpu_usage{*}, avg:k8s.pods.low_memory_usage{*}',
      options={
        show_legend: true,
        palette: 'purple',
        display_type: 'area',
      }
    ),
}
