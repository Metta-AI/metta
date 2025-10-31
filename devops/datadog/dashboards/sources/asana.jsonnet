// Asana Project Management Dashboard
// Project health, bugs tracking, and team velocity metrics
//
// Rewritten using the Jsonnet component system

local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';
local presets = import '../lib/presets.libsonnet';
local asana = import '../components/asana.libsonnet';

layouts.grid(
  'Asana Project Management',
  [
    // Header
    widgets.note(
      '# Asana Project Management Dashboard\n\n' +
      'Tracks project health status and bugs lifecycle across all active projects.\n\n' +
      '**Metrics Categories:**\n' +
      '- **Project Health**: Active projects, status distribution (on track, at risk, off track)\n' +
      '- **Bug Tracking**: Open bugs, triage queue, active work, backlog\n' +
      '- **Bug Velocity**: Bugs created/completed over time, average age\n\n' +
      '_Updated every 6 hours via automated collector_',
      {
        background_color: 'blue',
        font_size: '14',
        text_align: 'left',
        has_padding: true,
      }
    ),

    // Section 1: Project Health
    widgets.group(
      'Project Health',
      [
        asana.activeProjectsWidget(),
        asana.projectsOnTrackWidget(),
        asana.projectsAtRiskWidget(),
        asana.projectsOffTrackWidget(),
        asana.projectHealthWidget(),
      ],
      {
        background_color: 'vivid_green',
      }
    ),

    // Section 2: Bug Status
    widgets.group(
      'Bug Status',
      [
        asana.totalOpenBugsWidget(),
        asana.bugsInTriageWidget(),
        asana.activeBugsWidget(),
        asana.bugsInBacklogWidget(),
        asana.bugTrendWidget(),
      ],
      {
        background_color: 'vivid_orange',
      }
    ),

    // Section 3: Bug Velocity
    widgets.group(
      'Bug Velocity',
      [
        asana.bugsCreated7dWidget(),
        asana.bugsCompleted7dWidget(),
        asana.bugsCompleted30dWidget(),
        asana.bugVelocityWidget(),
      ],
      {
        background_color: 'vivid_purple',
      }
    ),

    // Section 4: Bug Age Metrics
    widgets.group(
      'Bug Age Metrics',
      [
        asana.avgBugAgeWidget(),
        asana.oldestBugAgeWidget(),
        presets.infoNote(
          '**Bug Age Analysis**\n\n' +
          'Monitor bug aging to identify stale issues that may need prioritization or closure.\n\n' +
          '**Guidelines:**\n' +
          '- Average age < 30 days: Good\n' +
          '- Average age 30-60 days: Monitor\n' +
          '- Average age > 60 days: Action needed\n\n' +
          'Oldest bug age helps identify long-standing issues that may have been forgotten.'
        ),
      ],
      {
        background_color: 'vivid_blue',
      }
    ),

    // Footer
    presets.infoNote(
      '---\n\n' +
      '**Total Metrics**: 14 metrics across 4 sections\n\n' +
      '**Data Source**: Asana API via automated collector\n' +
      '**Collection Frequency**: Every 6 hours\n' +
      '**Component Library**: `components/asana.libsonnet`\n\n' +
      '_Dashboard source: `dashboards/sources/asana.jsonnet`_'
    ),
  ],
  {
    id: "srz-bhk-zr2",  // Dashboard ID from Datadog
    description: "Project health, bugs tracking, and team velocity metrics from Asana",
  }
)
