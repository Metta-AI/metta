// Asana project management and bugs tracking components
// Widgets for project health, bug tracking, and team velocity

local widgets = import '../lib/widgets.libsonnet';

{
  // ========== Project Health Metrics ==========

  // Active projects count
  activeProjectsWidget()::
    widgets.queryValue(
      title='Active Projects',
      query='avg:asana.projects.active{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Projects on track
  projectsOnTrackWidget()::
    widgets.queryValue(
      title='Projects On Track',
      query='avg:asana.projects.on_track{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Projects at risk
  projectsAtRiskWidget()::
    widgets.queryValue(
      title='Projects At Risk',
      query='avg:asana.projects.at_risk{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Projects off track
  projectsOffTrackWidget()::
    widgets.queryValue(
      title='Projects Off Track',
      query='avg:asana.projects.off_track{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Project health trend over time
  projectHealthWidget()::
    widgets.timeseries(
      title='Project Health Status',
      query='avg:asana.projects.on_track{*}',
      options={
        show_legend: true,
        palette: 'green',
        display_type: 'area',
      }
    ),

  // ========== Bug Tracking Metrics ==========

  // Total open bugs across all projects
  totalOpenBugsWidget()::
    widgets.queryValue(
      title='Total Open Bugs',
      query='avg:asana.projects.bugs.total_open{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Bugs awaiting triage
  bugsInTriageWidget()::
    widgets.queryValue(
      title='Bugs in Triage',
      query='avg:asana.projects.bugs.triage_count{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Bugs actively being worked on
  activeBugsWidget()::
    widgets.queryValue(
      title='Active Bugs',
      query='avg:asana.projects.bugs.active_count{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Bugs in backlog
  bugsInBacklogWidget()::
    widgets.queryValue(
      title='Bugs in Backlog',
      query='avg:asana.projects.bugs.backlog_count{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Bug trend over time
  bugTrendWidget()::
    widgets.timeseries(
      title='Bug Trend (Total Open)',
      query='avg:asana.projects.bugs.total_open{*}',
      options={
        show_legend: true,
        palette: 'warm',
      }
    ),

  // ========== Bug Velocity Metrics ==========

  // Bugs created in last 7 days
  bugsCreated7dWidget()::
    widgets.queryValue(
      title='New Bugs (7 days)',
      query='avg:asana.projects.bugs.created_7d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Bugs completed in last 7 days
  bugsCompleted7dWidget()::
    widgets.queryValue(
      title='Bugs Completed (7 days)',
      query='avg:asana.projects.bugs.completed_7d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Bugs completed in last 30 days
  bugsCompleted30dWidget()::
    widgets.queryValue(
      title='Bugs Completed (30 days)',
      query='avg:asana.projects.bugs.completed_30d{*}',
      options={
        precision: 0,
        aggregator: 'last',
      }
    ),

  // Bug velocity trend (created vs completed)
  bugVelocityWidget()::
    widgets.timeseries(
      title='Bug Velocity (Created vs Completed)',
      query='avg:asana.projects.bugs.completed_7d{*}',
      options={
        show_legend: true,
        palette: 'cool',
      }
    ),

  // ========== Bug Age Metrics ==========

  // Average age of open bugs
  avgBugAgeWidget()::
    widgets.queryValue(
      title='Average Bug Age (days)',
      query='avg:asana.projects.bugs.avg_age_days{*}',
      options={
        precision: 1,
        aggregator: 'last',
        custom_unit: 'days',
      }
    ),

  // Oldest open bug age
  oldestBugAgeWidget()::
    widgets.queryValue(
      title='Oldest Bug Age (days)',
      query='avg:asana.projects.bugs.oldest_bug_days{*}',
      options={
        precision: 0,
        aggregator: 'last',
        custom_unit: 'days',
      }
    ),
}
