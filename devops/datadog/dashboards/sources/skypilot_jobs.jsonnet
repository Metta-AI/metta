// Skypilot Jobs Dashboard
// Job tracking, GPU utilization, and cluster health monitoring
//
// Rewritten using the Jsonnet component system

local layouts = import '../lib/layouts.libsonnet';
local presets = import '../lib/presets.libsonnet';
local skypilot = import '../components/skypilot.libsonnet';

layouts.grid(
  'Skypilot Jobs Dashboard',
  std.flattenArrays([
    // Row 1: Key metrics (4 equal-width widgets)
    layouts.row(0, [
      skypilot.runningJobsWidget(),
      skypilot.queuedJobsWidget(),
      skypilot.failedJobsWidget(),
      skypilot.activeClustersWidget(),
    ], height=2),

    // Row 2: Job status trends (2 half-width widgets)
    layouts.row(2, [
      skypilot.jobStatusTimeseriesWidget(),
      skypilot.jobSuccessRateWidget(),
    ], height=3),

    // Row 3: Section header
    [layouts.fullWidth(5, presets.sectionHeader(
      'Runtime Analysis',
      'Job execution times and duration distribution'
    ), height=1)],

    // Row 4: Runtime metrics (2 half-width widgets)
    layouts.row(6, [
      skypilot.runtimePercentilesWidget(),
      skypilot.runtimeBucketsWidget(),
    ], height=3),

    // Row 5: Section header
    [layouts.fullWidth(9, presets.sectionHeader(
      'Resource Utilization',
      'GPU allocation and instance type distribution'
    ), height=1)],

    // Row 6: GPU metrics (custom widths: 3, 3, 6)
    layouts.rowCustom(
      10,
      [
        skypilot.totalGPUsWidget(),
        skypilot.activeUsersWidget(),
        skypilot.gpuTypeDistributionWidget(),
      ],
      [3, 3, 6],
      height=3
    ),

    // Row 7: Resource distribution (2 half-width widgets)
    layouts.row(13, [
      skypilot.spotVsOnDemandWidget(),
      skypilot.regionalDistributionWidget(),
    ], height=3),

    // Row 8: Section header
    [layouts.fullWidth(16, presets.sectionHeader(
      'Reliability',
      'Job recovery and failure analysis'
    ), height=1)],

    // Row 9: Reliability metrics (2 half-width widgets)
    layouts.row(17, [
      skypilot.jobsWithRecoveriesWidget(),
      skypilot.avgRecoveryCountWidget(),
    ], height=3),
  ]),
  {
    id: 'mtw-y2p-4ed',  // Dashboard ID from Datadog
    description: 'Track Skypilot job status, GPU utilization, runtime metrics, and cluster health across regions.',
  }
)
