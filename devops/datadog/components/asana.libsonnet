// Asana project management and bugs tracking components
// Reusable widgets for Asana metrics

local widgets = import '../lib/widgets.libsonnet';

{
  // Project health widgets
  activeProjectsWidget():: widgets.queryValue(
    title='Active Projects',
    query='avg:asana.projects.active{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  projectsOnTrackWidget():: widgets.queryValue(
    title='Projects On Track',
    query='avg:asana.projects.on_track{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  projectsAtRiskWidget():: widgets.queryValue(
    title='Projects At Risk',
    query='avg:asana.projects.at_risk{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  projectsOffTrackWidget():: widgets.queryValue(
    title='Projects Off Track',
    query='avg:asana.projects.off_track{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  // Bugs tracking widgets
  totalOpenBugsWidget():: widgets.queryValue(
    title='Total Open Bugs',
    query='avg:asana.projects.bugs.total_open{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  bugsInTriageWidget():: widgets.queryValue(
    title='Bugs in Triage',
    query='avg:asana.projects.bugs.triage_count{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  activeBugsWidget():: widgets.queryValue(
    title='Active Bugs',
    query='avg:asana.projects.bugs.active_count{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  bugsInBacklogWidget():: widgets.queryValue(
    title='Bugs in Backlog',
    query='avg:asana.projects.bugs.backlog_count{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  // Bugs velocity widgets
  bugsCompleted7dWidget():: widgets.queryValue(
    title='Bugs Completed (7 days)',
    query='avg:asana.projects.bugs.completed_7d{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  bugsCompleted30dWidget():: widgets.queryValue(
    title='Bugs Completed (30 days)',
    query='avg:asana.projects.bugs.completed_30d{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  bugsCreated7dWidget():: widgets.queryValue(
    title='New Bugs (7 days)',
    query='avg:asana.projects.bugs.created_7d{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  // Bug age metrics
  avgBugAgeWidget():: widgets.queryValue(
    title='Average Bug Age (days)',
    query='avg:asana.projects.bugs.avg_age_days{*}',
    options={
      precision: 1,
      aggregator: 'last',
      custom_unit: 'days',
    }
  ),

  oldestBugAgeWidget():: widgets.queryValue(
    title='Oldest Bug Age (days)',
    query='avg:asana.projects.bugs.oldest_bug_days{*}',
    options={
      precision: 0,
      aggregator: 'last',
      custom_unit: 'days',
    }
  ),

  // Timeseries widgets
  bugTrendWidget():: widgets.timeseries(
    title='Bug Trend (Total Open)',
    query='avg:asana.projects.bugs.total_open{*}',
    options={
      show_legend: true,
      palette: 'warm',
    }
  ),

  bugVelocityWidget():: widgets.timeseries(
    title='Bug Velocity (Created vs Completed)',
    query='avg:asana.projects.bugs.completed_7d{*}',
    options={
      show_legend: true,
      palette: 'cool',
    }
  ),

  projectHealthWidget():: widgets.timeseries(
    title='Project Health Status',
    query='avg:asana.projects.on_track{*}',
    options={
      show_legend: true,
      palette: 'green',
      display_type: 'area',
    }
  ),
}
