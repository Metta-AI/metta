// Demo Dashboard - Comprehensive showcase of Jsonnet component system
// Demonstrates the full widget hierarchy: primitives → presets → domain components

local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';
local presets = import '../lib/presets.libsonnet';
local ci = import '../components/ci.libsonnet';
local github = import '../components/github.libsonnet';
local infrastructure = import '../components/infrastructure.libsonnet';

layouts.grid(
  'Jsonnet Component System Demo',
  std.flattenArrays([
    // Header
    [layouts.fullWidth(0, widgets.note(
      '# Dashboard Overview\n\n' +
      'This dashboard demonstrates the complete Jsonnet-based component system:\n\n' +
      '**Widget Hierarchy**:\n' +
      '- **Primitives** (`lib/widgets.libsonnet`) - Low-level widget builders\n' +
      '- **Presets** (`lib/presets.libsonnet`) - Pre-configured common patterns with smart defaults\n' +
      '- **Domain Components** (`components/*.libsonnet`) - Business-specific widgets\n' +
      '- **Layouts** (`lib/layouts.libsonnet`) - Grid positioning and organization\n\n' +
      '_All components are LLM-friendly with extensive inline documentation_',
      {
        background_color: 'blue',
        font_size: '14',
        text_align: 'left',
        has_padding: true,
      }
    ), height=2)],

    // Section 1: Preset Widgets with Smart Defaults
    [layouts.fullWidth(2, presets.sectionHeader(
      'Preset Widgets',
      'Pre-configured patterns with industry-standard thresholds'
    ), height=1)],

    // Row: Key metrics using presets
    layouts.row(3, [
      presets.activeUsersGauge('Active Users', 'sum:users.active{*}'),
      presets.requestCountGauge('Total Requests', 'sum:requests{*}'),
      presets.errorRateGauge('Error Rate', 'avg:errors.rate{*}'),
      presets.latencyGauge('Avg Latency', 'avg:latency{*}'),
    ], height=2),

    // Row: Performance with SLO markers
    layouts.row(5, [
      presets.cpuTimeseries('CPU Usage', 'avg:system.cpu.user{*}'),
      presets.memoryTimeseries('Memory Usage', 'avg:system.mem.pct_usable{*}'),
    ], height=3),

    // Section 2: Domain-Specific Components
    [layouts.fullWidth(8, presets.sectionHeader(
      'Domain Components',
      'Business-specific widgets from component libraries'
    ), height=1)],

    // Subsection: CI/CD Health
    [layouts.fullWidth(9, presets.subsectionHeader('CI/CD Health & Code Quality'), height=1)],

    layouts.row(10, [
      ci.testsPassingWidget(),
      ci.revertsCountWidget(),
      ci.hotfixCountWidget(),
    ], height=2),

    layouts.row(12, [
      github.failedWorkflowsWidget(),
      github.ciDurationPercentilesWidget(),
    ], height=3),

    // Subsection: Infrastructure
    [layouts.fullWidth(15, presets.subsectionHeader('Infrastructure & Resources'), height=1)],

    layouts.row(16, [
      infrastructure.containerCpuWidget(),
      infrastructure.containerMemoryWidget(),
    ], height=3),

    layouts.row(19, [
      infrastructure.kubernetesPodCountWidget(),
      infrastructure.kubernetesNodesWidget(),
      infrastructure.containerRestartsWidget(),
    ], height=2),

    // Section 3: Widget Variety
    [layouts.fullWidth(21, presets.sectionHeader(
      'Widget Types',
      'Full range of visualization options'
    ), height=1)],

    // Subsection: Rankings
    [layouts.fullWidth(22, presets.subsectionHeader('Top Lists'), height=1)],

    layouts.row(23, [
      presets.topServicesByRequests('Busiest Services', 'sum:requests{*} by {service}'),
      presets.topHostsByCPU('Hosts by CPU', 'avg:system.cpu.user{*} by {host}'),
    ], height=3),

    // Subsection: Comparisons
    [layouts.fullWidth(26, presets.subsectionHeader('Change Widgets'), height=1)],

    layouts.row(27, [
      presets.userGrowthChange('User Growth (DoD)', 'sum:users.active{*}'),
      widgets.change(
        'Test Pass Rate (DoD)',
        'avg:ci.tests_passing{*}',
        {
          compare_to: 'day_before',
          increase_good: true,
        }
      ),
    ], height=2),

    // Subsection: Density Plots
    [layouts.fullWidth(29, presets.subsectionHeader('Heatmaps & Distributions'), height=1)],

    layouts.row(30, [
      widgets.heatmap(
        'CPU Distribution by Host',
        'avg:system.cpu.user{*} by {host}',
        {
          show_legend: true,
          palette: 'warm',
        }
      ),
      widgets.heatmap(
        'Memory Distribution',
        'avg:system.mem.used{*} by {host}',
        {
          show_legend: true,
          palette: 'purple',
        }
      ),
    ], height=3),

    // Subsection: Tables
    [layouts.fullWidth(33, presets.subsectionHeader('Multi-Metric Tables'), height=1)],

    layouts.row(34, [
      presets.serviceHealthTable('Service Health Dashboard'),
      presets.hostHealthTable('Host Health Dashboard'),
    ], height=4),

    // Section 4: Layout Patterns
    [layouts.fullWidth(38, presets.sectionHeader(
      'Layout Patterns',
      'Grid positioning and organization helpers'
    ), height=1)],

    [layouts.fullWidth(39, presets.infoNote(
      '**Layout Helpers Demonstrated**:\n' +
      '- `layouts.row()` - Equal-width widgets across 12 columns\n' +
      '- `layouts.fullWidth()` - Full-width section headers\n' +
      '- `layouts.grid()` - 12-column grid positioning\n' +
      '- `std.flattenArrays()` - Combine multiple layout sections'
    ), height=1)],

    // Custom width demonstration
    layouts.rowCustom(
      40,
      [
        widgets.queryValue('Wide', 'avg:metric1{*}', { precision: 0 }),
        widgets.queryValue('Wider', 'avg:metric2{*}', { precision: 0 }),
        widgets.queryValue('Wide', 'avg:metric3{*}', { precision: 0 }),
      ],
      [3, 6, 3],  // 3 + 6 + 3 = 12 columns
      height=2
    ),

    // Footer with documentation links
    [layouts.fullWidth(42, widgets.note(
      '---\n\n' +
      '**Component Libraries**:\n' +
      '- `lib/widgets.libsonnet` - Primitive widget builders (timeseries, queryValue, toplist, etc.)\n' +
      '- `lib/presets.libsonnet` - Common patterns with smart defaults (CPU, memory, errors, latency)\n' +
      '- `lib/layouts.libsonnet` - Grid positioning helpers (row, column, grid2d, fullWidth)\n' +
      '- `components/ci.libsonnet` - CI/CD health metrics\n' +
      '- `components/infrastructure.libsonnet` - System and infrastructure metrics\n' +
      '- `components/github.libsonnet` - GitHub development metrics\n' +
      '- `components/skypilot.libsonnet` - Skypilot job metrics\n\n' +
      '**Documentation**:\n' +
      '- [Widget Primitives](lib/WIDGETS.md) - Low-level API reference\n' +
      '- [Presets](lib/PRESETS.md) - Pre-configured patterns\n' +
      '- [Layouts](lib/LAYOUTS.md) - Positioning and organization\n' +
      '- [Design Doc](DESIGN.md) - Overall architecture\n\n' +
      '_Dashboard source: `dashboards/sources/demo.jsonnet`_',
      {
        background_color: 'gray',
        font_size: '12',
        text_align: 'left',
      }
    ), height=3)],
  ]),
  {
    id: '2te-kvg-ja5',  // Dashboard ID from Datadog
    description: 'Comprehensive demonstration of the Jsonnet component system with primitives, presets, domain components, and layout helpers',
  }
)
