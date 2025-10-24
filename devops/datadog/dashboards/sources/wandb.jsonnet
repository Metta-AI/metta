// Weights & Biases Training Metrics Dashboard
// Monitors ML training runs, model performance, and GPU resource usage

local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';
local wandb = import '../components/wandb.libsonnet';

layouts.grid(
  'WandB Training Metrics',
  std.flattenArrays([
    // Header
    [layouts.fullWidth(0, widgets.note(
      '# WandB Training Metrics\n\n' +
      'Track ML training runs, model performance, and GPU resource usage. ' +
      'Metrics collected every 15 minutes from Weights & Biases.'
    ), height=1)],

    // Section 1: Training Run Status
    [layouts.fullWidth(1, widgets.note(
      '## Training Run Status',
      {
        background_color: 'gray',
        font_size: '16',
        text_align: 'left',
      }
    ), height=1)],

    // Row 1: Run status metrics (4 widgets)
    layouts.row(2, [
      wandb.activeRunsWidget(),
      wandb.completedRuns7dWidget(),
      wandb.failedRuns7dWidget(),
      wandb.totalRunsWidget(),
    ], height=2),
    [layouts.fullWidth(4, wandb.runStatusTrendWidget(), height=3)],

    // Section 2: Model Performance
    [layouts.fullWidth(7, widgets.note(
      '## Model Performance',
      {
        background_color: 'gray',
        font_size: '16',
        text_align: 'left',
      }
    ), height=1)],

    // Row 2: Performance metrics (3 widgets)
    layouts.row(8, [
      wandb.bestAccuracyWidget(),
      wandb.avgAccuracy7dWidget(),
      wandb.latestLossWidget(),
    ], height=2),
    layouts.row(10, [
      wandb.performanceTrendWidget(),
      wandb.lossTrendWidget(),
    ], height=3),

    // Section 3: Resource Usage
    [layouts.fullWidth(13, widgets.note(
      '## GPU Resource Usage',
      {
        background_color: 'gray',
        font_size: '16',
        text_align: 'left',
      }
    ), height=1)],

    // Row 3: Resource metrics (3 widgets)
    layouts.row(14, [
      wandb.avgDurationWidget(),
      wandb.gpuUtilizationWidget(),
      wandb.totalGpuHours7dWidget(),
    ], height=2),
    [layouts.fullWidth(16, wandb.resourceUsageTrendWidget(), height=3)],
  ]),
  {
    id: 'dr3-pdj-rrw',  // Dashboard ID from Datadog
    description: 'WandB ML training metrics: run status, model performance, and GPU resource usage tracking',
  }
)
