// GitHub CI/CD Dashboard
// Development velocity, code quality, and CI/CD metrics

local github = import '../components/github.libsonnet';

{
  id: '7gy-9ub-2sq',
  title: 'GitHub CI/CD Dashboard',
  description: 'Development velocity, code quality, and CI/CD metrics from GitHub. Monitors PRs, commits, workflows, and developer activity.',
  layout_type: 'ordered',
  template_variables: [],
  widgets: [
    // Row 1: Key metrics
    github.openPRsWidget() + { layout: { x: 0, y: 0, width: 3, height: 2 } },
    github.mergedPRsWidget() + { layout: { x: 3, y: 0, width: 3, height: 2 } },
    github.activeDevelopersWidget() + { layout: { x: 6, y: 0, width: 3, height: 2 } },
    github.testsPassingWidget() + { layout: { x: 9, y: 0, width: 3, height: 2 } },

    // Row 2: PR trends
    github.prCycleTimeWidget() + { layout: { x: 0, y: 2, width: 6, height: 3 } },
    github.stalePRsWidget() + { layout: { x: 6, y: 2, width: 6, height: 3 } },

    // Row 3: Code quality
    github.hotfixesWidget() + { layout: { x: 0, y: 5, width: 6, height: 3 } },
    github.revertsWidget() + { layout: { x: 6, y: 5, width: 6, height: 3 } },

    // Row 4: CI/CD health
    github.failedWorkflowsWidget() + { layout: { x: 0, y: 8, width: 6, height: 3 } },

    // Row 5: Section header
    github.sectionNote(
      'CI/CD Performance',
      'Workflow execution times and success rates'
    ) + { layout: { x: 0, y: 11, width: 12, height: 1 } },

    // Row 6: CI metrics
    github.ciDurationPercentilesWidget() + { layout: { x: 0, y: 12, width: 6, height: 3 } },
    github.workflowSuccessRateWidget() + { layout: { x: 6, y: 12, width: 6, height: 3 } },

    // Row 7: Developer productivity
    github.commitsPerDeveloperWidget() + { layout: { x: 0, y: 15, width: 12, height: 3 } },
  ],
}
