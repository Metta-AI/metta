// GitHub CI/CD Dashboard
// Development velocity, code quality, and CI/CD metrics
//
// Rewritten using the Jsonnet component system:
// - layouts.grid() for dashboard structure
// - layouts.row() for automatic positioning
// - presets.sectionHeader() for section headers
// - github.* components for domain-specific widgets

local layouts = import '../lib/layouts.libsonnet';
local presets = import '../lib/presets.libsonnet';
local github = import '../components/github.libsonnet';

layouts.grid(
  'GitHub CI/CD Dashboard',
  std.flattenArrays([
    // Row 1: Key metrics (4 equal-width widgets)
    layouts.row(0, [
      github.openPRsWidget(),
      github.mergedPRsWidget(),
      github.activeDevelopersWidget(),
      github.testsPassingWidget(),
    ], height=2),

    // Row 2: PR trends (2 half-width widgets)
    layouts.row(2, [
      github.prCycleTimeWidget(),
      github.stalePRsWidget(),
    ], height=3),

    // Row 3: Code quality (2 half-width widgets)
    layouts.row(5, [
      github.hotfixesWidget(),
      github.revertsWidget(),
    ], height=3),

    // Row 4: CI/CD health
    [layouts.halfWidth(8, github.failedWorkflowsWidget(), left=true, height=3)],

    // Row 5: Section header
    [layouts.fullWidth(11, presets.sectionHeader(
      'CI/CD Performance',
      'Workflow execution times and success rates'
    ), height=1)],

    // Row 6: CI metrics (2 half-width widgets)
    layouts.row(12, [
      github.ciDurationPercentilesWidget(),
      github.workflowSuccessRateWidget(),
    ], height=3),

    // Row 7: Developer productivity
    [layouts.fullWidth(15, github.commitsPerDeveloperWidget(), height=3)],
  ]),
  {
    id: '7gy-9ub-2sq',  // Keep existing dashboard ID for updates
    description: 'Development velocity, code quality, and CI/CD metrics from GitHub. Monitors PRs, commits, workflows, and developer activity.',
  }
)
