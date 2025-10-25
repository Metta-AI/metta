// Kubernetes Resource Efficiency and Health Dashboard
// Monitors cluster costs, pod health issues, and optimization opportunities

local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';
local k8s = import '../components/kubernetes.libsonnet';

layouts.grid(
  'Kubernetes Cluster Health',
  std.flattenArrays([
    // Header
    [layouts.fullWidth(0, widgets.note(
      '# Kubernetes Cluster Health\n\n' +
      'Track resource efficiency, pod health issues, and cost optimization opportunities. ' +
      'Metrics collected every 15 minutes.'
    ), height=1)],

    // Section 1: Resource Efficiency
    [layouts.fullWidth(1, widgets.note(
      '## Resource Efficiency',
      {
        background_color: 'gray',
        font_size: '16',
        text_align: 'left',
      }
    ), height=1)],

    // Row 1: Efficiency metrics (5 widgets)
    layouts.row(2, [
      k8s.cpuWasteWidget(),
      k8s.memoryWasteWidget(),
      k8s.cpuEfficiencyWidget(),
      k8s.memoryEfficiencyWidget(),
    ], height=2),
    layouts.row(4, [
      k8s.overallocatedPodsWidget(),
      k8s.resourceEfficiencyTrendWidget(),
    ], height=3),

    // Section 2: Pod Health
    [layouts.fullWidth(7, widgets.note(
      '## Pod Health Issues',
      {
        background_color: 'gray',
        font_size: '16',
        text_align: 'left',
      }
    ), height=1)],

    // Row 2: Health metrics (6 widgets)
    layouts.row(8, [
      k8s.crashLoopingPodsWidget(),
      k8s.failedPodsWidget(),
      k8s.pendingPodsWidget(),
      k8s.oomkilledPodsWidget(),
    ], height=2),
    layouts.row(10, [
      k8s.highRestartsWidget(),
      k8s.imagePullErrorsWidget(),
      k8s.podHealthTrendWidget(),
    ], height=3),

    // Section 3: Underutilization
    [layouts.fullWidth(13, widgets.note(
      '## Underutilization & Cost Optimization',
      {
        background_color: 'gray',
        font_size: '16',
        text_align: 'left',
      }
    ), height=1)],

    // Row 3: Underutilization metrics (4 widgets)
    layouts.row(14, [
      k8s.idlePodsWidget(),
      k8s.lowCpuPodsWidget(),
      k8s.lowMemoryPodsWidget(),
      k8s.zeroReplicasWidget(),
    ], height=2),
    [layouts.fullWidth(16, k8s.underutilizationTrendWidget(), height=3)],
  ]),
  {
    id: '687-i5n-ncf',  // Dashboard ID from Datadog
    description: 'Kubernetes cluster resource efficiency, pod health monitoring, and cost optimization opportunities',
  }
)
