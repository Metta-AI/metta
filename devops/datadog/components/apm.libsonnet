// APM/Tracing component library
// Widgets for application performance monitoring

local widgets = import '../lib/widgets.libsonnet';

{
  // Orchestrator run cycle performance
  orchestratorRunCycleWidget()::
    widgets.timeseries(
      title='Orchestrator Run Cycle Duration',
      query='avg:trace.orchestrator.run_cycle{*}',
      options={
        display_type: 'line',
      }
    ),

  // Orchestrator errors
  orchestratorErrorsWidget()::
    widgets.timeseries(
      title='Orchestrator Run Cycle Errors',
      query='sum:trace.orchestrator.run_cycle.errors{*}',
      options={
        display_type: 'bars',
        palette: 'warm',
      }
    ),

  // Orchestrator hit rate
  orchestratorHitsWidget()::
    widgets.timeseries(
      title='Orchestrator Run Cycle Hits',
      query='sum:trace.orchestrator.run_cycle.hits{*}.as_rate()',
      options={
        display_type: 'area',
      }
    ),

  // Worker task execution
  workerExecuteTaskWidget()::
    widgets.timeseries(
      title='Worker Task Execution Duration',
      query='avg:trace.worker.execute_task{*}',
      options={
        display_type: 'line',
      }
    ),

  // Worker execution errors
  workerErrorsWidget()::
    widgets.timeseries(
      title='Worker Task Execution Errors',
      query='sum:trace.worker.execute_task.errors{*}',
      options={
        display_type: 'bars',
        palette: 'warm',
      }
    ),

  // Worker task hits
  workerHitsWidget()::
    widgets.timeseries(
      title='Worker Task Execution Rate',
      query='sum:trace.worker.execute_task.hits{*}.as_rate()',
      options={
        display_type: 'area',
      }
    ),

  // Orchestrator startup time
  orchestratorStartupWidget()::
    widgets.queryValue(
      title='Orchestrator Startup Duration (avg)',
      query='avg:trace.orchestrator.startup{*}',
      options={
        precision: 2,
        custom_unit: 'ms',
      }
    ),

  // Worker status update performance
  workerUpdateStatusWidget()::
    widgets.timeseries(
      title='Worker Status Update Duration',
      query='avg:trace.worker.update_status{*}',
      options={
        display_type: 'line',
      }
    ),
}
