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

  // Completed runs in last 7 days
  completedRuns7dWidget()::
    widgets.queryValue(
      title='Completed Runs (7d)',
      query='avg:wandb.runs.completed_7d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Failed runs in last 7 days
  failedRuns7dWidget()::
    widgets.queryValue(
      title='Failed Runs (7d)',
      query='avg:wandb.runs.failed_7d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Total runs
  totalRunsWidget()::
    widgets.queryValue(
      title='Total Runs',
      query='avg:wandb.runs.total{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Run status trend over time
  runStatusTrendWidget()::
    widgets.timeseries(
      title='Training Run Activity',
      query='avg:wandb.runs.active{*}, avg:wandb.runs.completed_7d{*}, avg:wandb.runs.failed_7d{*}',
      options={
        show_legend: true,
        palette: 'cool',
        display_type: 'line',
      }
    ),

  // ========== Model Performance Metrics ==========

  // Best accuracy achieved
  bestAccuracyWidget()::
    widgets.queryValue(
      title='Best Model Accuracy',
      query='avg:wandb.metrics.best_accuracy{*}',
      options={
        precision: 3,
        aggregator: 'last',
        custom_unit: '%',
      }
    ),

  // Latest training loss
  latestLossWidget()::
    widgets.queryValue(
      title='Latest Training Loss',
      query='avg:wandb.metrics.latest_loss{*}',
      options={
        precision: 4,
        aggregator: 'last',
      }
    ),

  // Average accuracy from last 7 days
  avgAccuracy7dWidget()::
    widgets.queryValue(
      title='Avg Accuracy (7d)',
      query='avg:wandb.metrics.avg_accuracy_7d{*}',
      options={
        precision: 3,
        aggregator: 'last',
        custom_unit: '%',
      }
    ),

  // Model performance trend over time
  performanceTrendWidget()::
    widgets.timeseries(
      title='Model Performance Trends',
      query='avg:wandb.metrics.best_accuracy{*}, avg:wandb.metrics.avg_accuracy_7d{*}',
      options={
        show_legend: true,
        palette: 'green',
        display_type: 'line',
      }
    ),

  // Loss trend over time
  lossTrendWidget()::
    widgets.timeseries(
      title='Training Loss Trend',
      query='avg:wandb.metrics.latest_loss{*}',
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

  // Total GPU hours in last 7 days
  totalGpuHours7dWidget()::
    widgets.queryValue(
      title='Total GPU Hours (7d)',
      query='avg:wandb.training.total_gpu_hours_7d{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: 'hours',
      }
    ),

  // Resource usage trend over time
  resourceUsageTrendWidget()::
    widgets.timeseries(
      title='GPU Resource Usage Trends',
      query='avg:wandb.training.gpu_utilization_avg{*}, avg:wandb.training.total_gpu_hours_7d{*}',
      options={
        show_legend: true,
        palette: 'purple',
        display_type: 'area',
      }
    ),
}
