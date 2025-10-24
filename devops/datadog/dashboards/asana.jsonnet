// Asana Project Management Dashboard
// Project health, bugs tracking, and team velocity metrics

local asana = import '../components/asana.libsonnet';
local widgets = import '../lib/widgets.libsonnet';

{
  title: 'Asana Project Management',
  description: 'Project health, bugs tracking, and team velocity metrics from Asana',
  layout_type: 'ordered',
  template_variables: [],
  widgets: [
    // Header
    widgets.note(
      '## 📋 Asana Project Management Dashboard\n\n' +
      'Tracks project health status and bugs lifecycle across all active projects.\n\n' +
      '**Metrics Categories:**\n' +
      '- **Project Health**: Active projects, status distribution (on track, at risk, off track)\n' +
      '- **Bug Tracking**: Open bugs, triage queue, active work, backlog\n' +
      '- **Bug Velocity**: Bugs created/completed over time, average age\n\n' +
      '_Updated every 6 hours via automated collector_',
      options={
        background_color: 'blue',
        font_size: '14',
        text_align: 'left',
        has_padding: true,
      }
    ),

    // Section 1: Project Health
    widgets.group(
      title='📊 Project Health',
      widgets=[
        asana.activeProjectsWidget(),
        asana.projectsOnTrackWidget(),
        asana.projectsAtRiskWidget(),
        asana.projectsOffTrackWidget(),
        asana.projectHealthWidget(),
      ],
      options={
        background_color: 'vivid_green',
        layout_type: 'ordered',
      }
    ),

    // Section 2: Bug Status
    widgets.group(
      title='🐛 Bug Status',
      widgets=[
        asana.totalOpenBugsWidget(),
        asana.bugsInTriageWidget(),
        asana.activeBugsWidget(),
        asana.bugsInBacklogWidget(),
        asana.bugTrendWidget(),
      ],
      options={
        background_color: 'vivid_orange',
        layout_type: 'ordered',
      }
    ),

    // Section 3: Bug Velocity
    widgets.group(
      title='📈 Bug Velocity',
      widgets=[
        asana.bugsCreated7dWidget(),
        asana.bugsCompleted7dWidget(),
        asana.bugsCompleted30dWidget(),
        asana.bugVelocityWidget(),
      ],
      options={
        background_color: 'vivid_purple',
        layout_type: 'ordered',
      }
    ),

    // Section 4: Bug Age Metrics
    widgets.group(
      title='⏰ Bug Age Metrics',
      widgets=[
        asana.avgBugAgeWidget(),
        asana.oldestBugAgeWidget(),
        widgets.note(
          '### Bug Age Analysis\n\n' +
          'Monitor bug aging to identify stale issues that may need prioritization or closure.\n\n' +
          '**Guidelines:**\n' +
          '- Average age < 30 days: Good\n' +
          '- Average age 30-60 days: Monitor\n' +
          '- Average age > 60 days: Action needed\n\n' +
          'Oldest bug age helps identify long-standing issues that may have been forgotten.',
          options={
            background_color: 'gray',
            font_size: '12',
            text_align: 'left',
          }
        ),
      ],
      options={
        background_color: 'vivid_blue',
        layout_type: 'ordered',
      }
    ),

    // Footer
    widgets.note(
      '---\n\n' +
      '**Total Metrics**: 14 metrics across 4 sections\n\n' +
      '**Data Source**: Asana API via automated collector\n' +
      '**Collection Frequency**: Every 6 hours\n' +
      '**Component Library**: `components/asana.libsonnet`\n\n' +
      '_Dashboard source: `dashboards/asana.jsonnet`_',
      options={
        background_color: 'gray',
        font_size: '12',
        text_align: 'left',
      }
    ),
  ],
}
