// GitHub metrics component library
// Widgets for development velocity, code quality, and CI/CD health

local widgets = import '../lib/widgets.libsonnet';

{
  // ========== Pull Request Metrics ==========

  // Open PRs count
  openPRsWidget()::
    widgets.queryValue(
      title='Open PRs',
      query='avg:github.prs.open{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // PRs merged in last 7 days
  mergedPRsWidget()::
    widgets.queryValue(
      title='PRs Merged (7 days)',
      query='sum:github.prs.merged_7d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Stale PRs (> 14 days old)
  stalePRsWidget()::
    widgets.timeseries(
      title='Stale PRs (> 14 days)',
      query='avg:github.prs.stale_count_14d{*}',
      options={
        line_width: 'thick',
      }
    ),

  // PR cycle time
  prCycleTimeWidget()::
    widgets.timeseries(
      title='PR Cycle Time (Hours)',
      query='avg:github.prs.cycle_time_hours{*}',
      options={
        line_width: 'thick',
      }
    ),

  // ========== Developer Metrics ==========

  // Active developers
  activeDevelopersWidget()::
    widgets.queryValue(
      title='Active Developers (7d)',
      query='avg:github.developers.active_7d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Commits per developer
  commitsPerDeveloperWidget()::
    widgets.timeseries(
      title='Commits per Developer (7d avg)',
      query='avg:github.commits.per_developer_7d{*}',
      options={
        line_width: 'normal',
      }
    ),

  // ========== CI/CD Metrics ==========

  // Tests passing on main
  testsPassingWidget()::
    widgets.queryValue(
      title='Tests on Main',
      query='avg:github.ci.tests_passing_on_main{*}',
      options={
        precision: 0,
        aggregator: 'last',
        autoscale: false,
      }
    ),

  // Failed workflows
  failedWorkflowsWidget()::
    widgets.timeseries(
      title='Failed Workflows (7 days)',
      query='avg:github.ci.failed_workflows_7d{*}',
      options={
        line_width: 'thick',
        markers: [
          {
            label: ' Ideal: 0 failures ',
            value: 'y = 0',
            display_type: 'ok dashed',
          },
        ],
      }
    ),

  // CI duration percentiles
  ciDurationPercentilesWidget()::
    widgets.timeseries(
      title='CI Duration Percentiles (Minutes)',
      query='avg:github.ci.duration_p50_minutes{*}, avg:github.ci.duration_p90_minutes{*}, avg:github.ci.duration_p99_minutes{*}',
      options={
        line_width: 'normal',
        show_legend: true,
      }
    ),

  // Workflow success rate
  workflowSuccessRateWidget()::
    widgets.timeseries(
      title='Workflow Success Rate',
      query='(avg:github.ci.workflow_runs_7d{*} - avg:github.ci.failed_workflows_7d{*}) / avg:github.ci.workflow_runs_7d{*} * 100',
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

  // ========== Code Quality Metrics ==========

  // Hotfixes
  hotfixesWidget()::
    widgets.timeseries(
      title='Hotfixes (7 days)',
      query='avg:github.commits.hotfix{*}',
      options={
        line_width: 'thick',
        markers: [
          {
            label: ' Ideal: 0 hotfixes ',
            value: 'y = 0',
            display_type: 'warning dashed',
          },
        ],
      }
    ),

  // Reverts
  revertsWidget()::
    widgets.timeseries(
      title='Reverts (7 days)',
      query='avg:github.commits.reverts{*}',
      options={
        line_width: 'thick',
        markers: [
          {
            label: ' Ideal: 0 reverts ',
            value: 'y = 0',
            display_type: 'warning dashed',
          },
        ],
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
