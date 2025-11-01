// Weights & Biases (WandB) training metrics components
// Widgets for ML training runs, model performance, and resource usage

local widgets = import '../lib/widgets.libsonnet';

{
  // ========== Training Run Status ==========

  // Active training runs
  activeRunsWidget()::
    widgets.queryValue(
      title='Active Training Runs',
      query='avg:wandb.runs.active{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Completed runs in last 24 hours
  completedRuns7dWidget()::
    widgets.queryValue(
      title='Completed Runs (24h)',
      query='avg:wandb.runs.completed_24h{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Failed runs in last 24 hours
  failedRuns7dWidget()::
    widgets.queryValue(
      title='Failed Runs (24h)',
      query='avg:wandb.runs.failed_24h{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Total recent runs (24h + active)
  totalRunsWidget()::
    widgets.queryValue(
      title='Total Recent Runs',
      query='avg:wandb.runs.total_recent{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Run status trend over time
  runStatusTrendWidget()::
    widgets.timeseries(
      title='Training Run Activity (24h window)',
      query='avg:wandb.runs.active{*}, avg:wandb.runs.completed_24h{*}, avg:wandb.runs.failed_24h{*}',
      options={
        show_legend: true,
        palette: 'cool',
        display_type: 'line',
      }
    ),

  // ========== Model Performance Metrics ==========

  // Latest training throughput (steps per second)
  bestAccuracyWidget()::
    widgets.queryValue(
      title='Training Throughput (SPS)',
      query='avg:wandb.metrics.latest_sps{*}',
      options={
        precision: 0,
        aggregator: 'last',
        custom_unit: 'steps/s',
      }
    ),

  // Latest SkyPilot queue latency
  latestLossWidget()::
    widgets.queryValue(
      title='Queue Latency',
      query='avg:wandb.metrics.latest_queue_latency_s{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: 's',
      }
    ),

  // Average heart amount (agent survival metric)
  avgAccuracy7dWidget()::
    widgets.queryValue(
      title='Avg Heart Amount (24h)',
      query='avg:wandb.metrics.avg_heart_amount_24h{*}',
      options={
        precision: 3,
        aggregator: 'last',
      }
    ),

  // Training throughput trend over time
  performanceTrendWidget()::
    widgets.timeseries(
      title='Training Throughput Trend',
      query='avg:wandb.metrics.latest_sps{*}',
      options={
        show_legend: true,
        palette: 'green',
        display_type: 'line',
      }
    ),

  // Heart amount trend over time
  lossTrendWidget()::
    widgets.timeseries(
      title='Agent Survival Trend (Heart Amount)',
      query='avg:wandb.metrics.avg_heart_amount_24h{*}',
      options={
        palette: 'warm',
        display_type: 'line',
      }
    ),

  // ========== Resource Usage Metrics ==========

  // Average training duration
  avgDurationWidget()::
    widgets.queryValue(
      title='Avg Training Duration',
      query='avg:wandb.training.avg_duration_hours{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: 'hours',
      }
    ),

  // GPU utilization average
  gpuUtilizationWidget()::
    widgets.queryValue(
      title='Avg GPU Utilization',
      query='avg:wandb.training.gpu_utilization_avg{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: '%',
      }
    ),

  // Total GPU hours in last 24 hours
  totalGpuHours7dWidget()::
    widgets.queryValue(
      title='Total GPU Hours (24h)',
      query='avg:wandb.training.total_gpu_hours_24h{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: 'hours',
      }
    ),

  // Resource usage trend over time
  resourceUsageTrendWidget()::
    widgets.timeseries(
      title='GPU Resource Usage Trends (24h window)',
      query='avg:wandb.training.gpu_utilization_avg{*}, avg:wandb.training.total_gpu_hours_24h{*}',
      options={
        show_legend: true,
        palette: 'purple',
        display_type: 'area',
      }
    ),
}
