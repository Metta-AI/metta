// Skypilot Jobs Dashboard
// Job tracking, GPU utilization, and cluster health monitoring

local skypilot = import '../components/skypilot.libsonnet';

{
  id: 'skypilot-jobs',
  title: 'Skypilot Jobs Dashboard',
  description: 'Track Skypilot job status, GPU utilization, runtime metrics, and cluster health across regions.',
  layout_type: 'ordered',
  template_variables: [],
  widgets: [
    // Row 1: Key metrics
    skypilot.runningJobsWidget() + { layout: { x: 0, y: 0, width: 3, height: 2 } },
    skypilot.queuedJobsWidget() + { layout: { x: 3, y: 0, width: 3, height: 2 } },
    skypilot.failedJobsWidget() + { layout: { x: 6, y: 0, width: 3, height: 2 } },
    skypilot.activeClustersWidget() + { layout: { x: 9, y: 0, width: 3, height: 2 } },

    // Row 2: Job status trends
    skypilot.jobStatusTimeseriesWidget() + { layout: { x: 0, y: 2, width: 6, height: 3 } },
    skypilot.jobSuccessRateWidget() + { layout: { x: 6, y: 2, width: 6, height: 3 } },

    // Row 3: Section header
    skypilot.sectionNote(
      'Runtime Analysis',
      'Job execution times and duration distribution'
    ) + { layout: { x: 0, y: 5, width: 12, height: 1 } },

    // Row 4: Runtime metrics
    skypilot.runtimePercentilesWidget() + { layout: { x: 0, y: 6, width: 6, height: 3 } },
    skypilot.runtimeBucketsWidget() + { layout: { x: 6, y: 6, width: 6, height: 3 } },

    // Row 5: Section header
    skypilot.sectionNote(
      'Resource Utilization',
      'GPU allocation and instance type distribution'
    ) + { layout: { x: 0, y: 9, width: 12, height: 1 } },

    // Row 6: GPU metrics
    skypilot.totalGPUsWidget() + { layout: { x: 0, y: 10, width: 3, height: 2 } },
    skypilot.activeUsersWidget() + { layout: { x: 3, y: 10, width: 3, height: 2 } },
    skypilot.gpuTypeDistributionWidget() + { layout: { x: 6, y: 10, width: 6, height: 3 } },

    // Row 7: Resource distribution
    skypilot.spotVsOnDemandWidget() + { layout: { x: 0, y: 13, width: 6, height: 3 } },
    skypilot.regionalDistributionWidget() + { layout: { x: 6, y: 13, width: 6, height: 3 } },

    // Row 8: Section header
    skypilot.sectionNote(
      'Reliability',
      'Job recovery and failure analysis'
    ) + { layout: { x: 0, y: 16, width: 12, height: 1 } },

    // Row 9: Reliability metrics
    skypilot.jobsWithRecoveriesWidget() + { layout: { x: 0, y: 17, width: 6, height: 3 } },
    skypilot.avgRecoveryCountWidget() + { layout: { x: 6, y: 17, width: 6, height: 3 } },
  ],
}
