// Infrastructure component library
// Widgets for system, container, and Kubernetes metrics

local widgets = import '../lib/widgets.libsonnet';

{
  // CPU usage by host
  cpuUsageByHostWidget()::
    widgets.toplist(
      title='CPU Usage by Host (Top 10)',
      query='top(avg:system.cpu.user{*} by {host}, 10, "mean", "desc")',
      options={}
    ),

  // Memory usage by host
  memoryUsageByHostWidget()::
    widgets.toplist(
      title='Memory Usage by Host (Top 10)',
      query='top(avg:system.mem.used{*} by {host}, 10, "mean", "desc")',
      options={}
    ),

  // Container CPU usage timeseries
  containerCpuWidget()::
    widgets.timeseries(
      title='Container CPU Usage',
      query='avg:container.cpu.usage{*} by {container_name}',
      options={
        display_type: 'area',
        show_legend: true,
      }
    ),

  // Container memory usage timeseries
  containerMemoryWidget()::
    widgets.timeseries(
      title='Container Memory Usage',
      query='avg:container.memory.usage{*} by {container_name}',
      options={
        display_type: 'area',
        show_legend: true,
      }
    ),

  // Kubernetes pod count
  kubernetesPodCountWidget()::
    widgets.queryValue(
      title='Total Kubernetes Pods',
      query='sum:kubernetes.pods.running{*}',
      options={
        aggregator: 'sum',
        precision: 0,
      }
    ),

  // System load average
  systemLoadWidget()::
    widgets.timeseries(
      title='System Load Average',
      query='avg:system.load.1{*} by {host}',
      options={
        display_type: 'line',
        show_legend: true,
      }
    ),

  // Disk usage by host
  diskUsageWidget()::
    widgets.timeseries(
      title='Disk Usage %',
      query='avg:system.disk.in_use{*} by {host,device}',
      options={
        display_type: 'line',
        show_legend: true,
      }
    ),

  // Network traffic
  networkTrafficWidget()::
    widgets.timeseries(
      title='Network Traffic (bytes/sec)',
      query='avg:system.net.bytes_rcvd{*} by {host}',
      options={
        display_type: 'area',
        show_legend: true,
      }
    ),

  // Kubernetes nodes status
  kubernetesNodesWidget()::
    widgets.queryValue(
      title='Kubernetes Nodes',
      query='sum:kubernetes_state.node.count{*}',
      options={
        aggregator: 'sum',
        precision: 0,
      }
    ),

  // Container restarts
  containerRestartsWidget()::
    widgets.timeseries(
      title='Container Restarts',
      query='sum:kubernetes.containers.restarts{*}.as_count()',
      options={
        display_type: 'bars',
        palette: 'warm',
      }
    ),
}
