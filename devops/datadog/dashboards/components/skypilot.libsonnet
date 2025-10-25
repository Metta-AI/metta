// Skypilot metrics component library
// Widgets for job tracking, GPU utilization, and cluster health

local widgets = import '../lib/widgets.libsonnet';

{
  // ========== Job Status Metrics ==========

  // Running jobs count
  runningJobsWidget()::
    widgets.queryValue(
      title='Running Jobs',
      query='avg:skypilot.jobs.running{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Queued jobs count
  queuedJobsWidget()::
    widgets.queryValue(
      title='Queued Jobs',
      query='avg:skypilot.jobs.queued{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Failed jobs (7 days)
  failedJobsWidget()::
    widgets.queryValue(
      title='Failed Jobs (7d)',
      query='sum:skypilot.jobs.failed_7d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Active clusters
  activeClustersWidget()::
    widgets.queryValue(
      title='Active Clusters',
      query='avg:skypilot.clusters.active{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Job status over time
  jobStatusTimeseriesWidget()::
    widgets.timeseries(
      title='Job Status Over Time',
      query='avg:skypilot.jobs.running{*}, avg:skypilot.jobs.queued{*}, avg:skypilot.jobs.failed{*}',
      options={
        line_width: 'normal',
        show_legend: true,
      }
    ),

  // Job success rate
  jobSuccessRateWidget()::
    widgets.timeseries(
      title='Job Success Rate (%)',
      query='sum:skypilot.jobs.succeeded{*} / (sum:skypilot.jobs.succeeded{*} + sum:skypilot.jobs.failed_7d{*}) * 100',
      options={
        line_width: 'thick',
        markers: [
          {
            label: ' Target: 95% success ',
            value: 'y = 95',
            display_type: 'ok dashed',
          },
        ],
      }
    ),

  // ========== Runtime Metrics ==========

  // Runtime percentiles
  runtimePercentilesWidget()::
    widgets.timeseries(
      title='Job Runtime Percentiles (seconds)',
      query='avg:skypilot.jobs.runtime_seconds.p50{*}, avg:skypilot.jobs.runtime_seconds.p90{*}, avg:skypilot.jobs.runtime_seconds.p99{*}',
      options={
        line_width: 'normal',
        show_legend: true,
      }
    ),

  // Runtime buckets distribution
  runtimeBucketsWidget()::
    widgets.timeseries(
      title='Job Runtime Distribution',
      query='avg:skypilot.jobs.runtime_buckets.0_1h{*}, avg:skypilot.jobs.runtime_buckets.1_4h{*}, avg:skypilot.jobs.runtime_buckets.4_24h{*}, avg:skypilot.jobs.runtime_buckets.over_24h{*}',
      options={
        line_width: 'normal',
        show_legend: true,
        display_type: 'bars',
      }
    ),

  // ========== Resource Utilization ==========

  // Total GPU count
  totalGPUsWidget()::
    widgets.queryValue(
      title='Total GPUs in Use',
      query='avg:skypilot.resources.gpus.total_count{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // GPU type distribution
  gpuTypeDistributionWidget()::
    widgets.timeseries(
      title='GPU Type Distribution',
      query='avg:skypilot.resources.gpus.l4_count{*}, avg:skypilot.resources.gpus.a10g_count{*}, avg:skypilot.resources.gpus.h100_count{*}',
      options={
        line_width: 'normal',
        show_legend: true,
        display_type: 'area',
      }
    ),

  // Spot vs on-demand
  spotVsOnDemandWidget()::
    widgets.timeseries(
      title='Spot vs On-Demand Jobs',
      query='avg:skypilot.resources.spot_jobs{*}, avg:skypilot.resources.ondemand_jobs{*}',
      options={
        line_width: 'normal',
        show_legend: true,
      }
    ),

  // ========== Regional Distribution ==========

  // Regional distribution
  regionalDistributionWidget()::
    widgets.timeseries(
      title='Jobs by Region',
      query='avg:skypilot.regions.us_east_1{*}, avg:skypilot.regions.us_west_2{*}, avg:skypilot.regions.other{*}',
      options={
        line_width: 'normal',
        show_legend: true,
        display_type: 'area',
      }
    ),

  // ========== Reliability Metrics ==========

  // Jobs with recoveries
  jobsWithRecoveriesWidget()::
    widgets.timeseries(
      title='Jobs with Recoveries',
      query='avg:skypilot.jobs.with_recoveries{*}',
      options={
        line_width: 'normal',
      }
    ),

  // Average recovery count
  avgRecoveryCountWidget()::
    widgets.timeseries(
      title='Average Recoveries per Job',
      query='avg:skypilot.jobs.recovery_count.avg{*}',
      options={
        line_width: 'normal',
      }
    ),

  // ========== Team Activity ==========

  // Active users
  activeUsersWidget()::
    widgets.queryValue(
      title='Active Users',
      query='avg:skypilot.users.active_count{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // ========== Sections ==========

  // Section header note
  sectionNote(title, description='')::
    widgets.note(
      content='## ' + title + '\n\n' + description,
      options={
        background_color: 'gray',
        font_size: '16',
        text_align: 'center',
        vertical_align: 'center',
        has_padding: true,
      }
    ),
}
