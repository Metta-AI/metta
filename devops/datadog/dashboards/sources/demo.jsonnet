// Demo Dashboard - Showcasing Jsonnet Component System
// Demonstrates widget variety and organizational patterns with groups

local ci = import '../components/ci.libsonnet';
local apm = import '../components/apm.libsonnet';
local infrastructure = import '../components/infrastructure.libsonnet';
local widgets = import '../lib/widgets.libsonnet';

{
  title: 'Jsonnet Demo Dashboard',
  description: 'Demo dashboard showcasing the Jsonnet component system with grouped sections and a variety of widget types',
  layout_type: 'ordered',
  template_variables: [],
  widgets: [
    // Header: Overview
    widgets.note(
      '## 📊 Dashboard Overview\n\n' +
      'This dashboard demonstrates the Jsonnet-based component system with grouped sections:\n' +
      '- **Timeseries**: Line, area, and bar charts\n' +
      '- **Query Values**: Single number displays\n' +
      '- **Top Lists**: Ranked lists\n' +
      '- **Change Widgets**: Period-over-period comparison\n' +
      '- **Heatmaps**: Density visualization\n' +
      '- **Tables**: Structured data\n\n' +
      '_Built with reusable Jsonnet components_',
      options={
        background_color: 'blue',
        font_size: '14',
        text_align: 'left',
        has_padding: true,
      }
    ),

    // Section 1: CI/CD Health (as a group)
    widgets.group(
      title='🔄 CI/CD Health & Code Quality',
      widgets=[
        ci.testsPassingWidget(),
        ci.revertsCountWidget(),
        ci.hotfixCountWidget(),
        widgets.change(
          title='Test Pass Rate (Change from 1 day ago)',
          query='avg:ci.tests_passing_on_main{source:softmax-system-health}',
          options={
            compare_to: 'day_before',
            increase_good: true,
          }
        ),
      ],
      options={
        background_color: 'vivid_blue',
        layout_type: 'ordered',
      }
    ),

    // Section 2: APM Performance (as a group)
    widgets.group(
      title='⚡ Application Performance (APM)',
      widgets=[
        apm.orchestratorRunCycleWidget(),
        apm.orchestratorErrorsWidget(),
        apm.orchestratorHitsWidget(),
        apm.workerExecuteTaskWidget(),
        apm.workerErrorsWidget(),
        apm.workerHitsWidget(),
        apm.orchestratorStartupWidget(),
        widgets.queryValue(
          title='Worker Task Execution (avg ms)',
          query='avg:trace.worker.execute_task{*}',
          options={
            precision: 2,
            custom_unit: 'ms',
          }
        ),
      ],
      options={
        background_color: 'vivid_purple',
        layout_type: 'ordered',
      }
    ),

    // Section 3: Infrastructure (as a group)
    widgets.group(
      title='🏗️ Infrastructure & Resources',
      widgets=[
        infrastructure.cpuUsageByHostWidget(),
        infrastructure.memoryUsageByHostWidget(),
        infrastructure.containerCpuWidget(),
        infrastructure.containerMemoryWidget(),
        infrastructure.kubernetesPodCountWidget(),
        infrastructure.kubernetesNodesWidget(),
        infrastructure.systemLoadWidget(),
        infrastructure.containerRestartsWidget(),
      ],
      options={
        background_color: 'vivid_green',
        layout_type: 'ordered',
      }
    ),

    // Section 4: Network & Storage (as a group)
    widgets.group(
      title='💾 Network & Storage',
      widgets=[
        infrastructure.networkTrafficWidget(),
        infrastructure.diskUsageWidget(),
        widgets.heatmap(
          title='System CPU Distribution',
          query='avg:system.cpu.user{*} by {host}',
          options={
            show_legend: true,
            palette: 'blue',
          }
        ),
        widgets.table(
          title='System Metrics Summary',
          queries=[
            { query: 'avg:system.cpu.user{*} by {host}', alias: 'CPU %' },
            { query: 'avg:system.mem.used{*} by {host}', alias: 'Memory Used' },
            { query: 'avg:system.load.1{*} by {host}', alias: 'Load Avg' },
          ],
          options={
            has_search_bar: 'auto',
          }
        ),
      ],
      options={
        background_color: 'vivid_orange',
        layout_type: 'ordered',
      }
    ),

    // Footer
    widgets.note(
      '---\n\n' +
      '**Total Widgets**: 30+ widgets across 4 grouped sections\n\n' +
      '**Component Libraries Used**:\n' +
      '- `components/ci.libsonnet` - CI/CD health\n' +
      '- `components/apm.libsonnet` - Application performance\n' +
      '- `components/infrastructure.libsonnet` - System metrics\n' +
      '- `lib/widgets.libsonnet` - Widget primitives\n\n' +
      '_Dashboard source: `dashboards/demo.jsonnet`_',
      options={
        background_color: 'gray',
        font_size: '12',
        text_align: 'left',
      }
    ),
  ],
}
